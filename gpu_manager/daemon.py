"""主循环与状态机。

单一主进程 + 单线程主循环（外加一个只做 IPC 的 daemon 线程）：

    ┌─ 每 tick ────────────────────────────────────────────────┐
    │ 1. nvidia-smi 采集（失败即保守处理，不判空闲）              │
    │ 2. 收割已结束的任务（退出码来自 .exit 文件，重启也不丢）      │
    │ 3. 处理停止请求（SIGTERM → grace → SIGKILL，只对自己进程组）│
    │ 4. 同步 tasks.yaml 提交（幂等）                            │
    │ 5. 连续空闲确认（idle_confirm_checks 次）                  │
    │ 6. 调度: priority + first fit，启动前二次 nvidia-smi 检查    │
    │ 7. 计算模式 IDLE/QUEUE/RUNNING 并发送通知                   │
    │ 8. 采样日志进度（供 ETA）、按 storage 上限清理日志                │
    │ 9. 持久化 state.json                                      │
    └──────────────────────────────────────────────────────────┘

模式语义:
    RUNNING  至少有一个由 Manager 启动的任务在跑（优先级最高）
    QUEUE    有 pending 任务但没有 running
    IDLE     既无 pending 也无 running

安全边界: 本模块只对「当前用户拥有且 starttime 匹配」的进程发信号；
对其他用户的进程只读观察，永不发信号、永不清理显存、永不 reset。
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from . import __version__, eta
from .bot import (
    AllowList,
    BotContext,
    CommandRouter,
    FeishuBot,
    FeishuTransport,
    GpuView,
    RateLimiter,
    TaskView,
)
from .config import Config, ConfigError, load_config
from .gpu_monitor import (
    Snapshot,
    as_dict_rows,
    collect,
    idle_gpu_ids,
    idle_report,
    still_free,
)
from .idle_tracker import IdleTracker
from .ipc import ControlServer, fail, ok
from .notifier import Notifier, build_notifier
from .retention import prune_all
from .runner import RunHandle, RunnerError, TaskRunner
from .scheduler import GpuCandidate, Plan, describe_plan, plan_assignments
from .state_store import StateStore
from .task_queue import (
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    ImportResult,
    QueueManager,
    Task,
    TaskError,
)
from .utils import (
    build_logger,
    ensure_dir,
    fmt_duration,
    fmt_ts,
    gpu_ids_text,
    process_alive,
    truncate,
)
from .watchdog import Watchdog

MODE_IDLE = "IDLE"
MODE_QUEUE = "QUEUE"
MODE_RUNNING = "RUNNING"

#: 任务终态 → (通知等级, 图标, 中文标签)。
#
# 集中定义一次并配合 .get() 默认值：以前这些映射散在 _finish_task 里，
# 而且 level 那张表的键被误写成大写字面量（"FAILED"），与 status 的小写值
# （"failed"）不匹配 → 每次任务结束都抛 KeyError，导致完成/失败通知静默丢失。
# 写成常量表 + .get() 后这类错位在结构上就不可能发生。
TERMINAL_STYLE: dict[str, tuple[str, str, str]] = {
    COMPLETED: ("success", "✅", "任务完成"),
    FAILED: ("error", "❌", "任务失败"),
    CANCELLED: ("warning", "🛑", "任务已取消"),
}

WAITING_LOG_INTERVAL = 600.0  # 等待原因最多每 10 分钟刷一条 INFO，避免日志噪声

#: 进度采样时回扫的日志行数（tqdm/Epoch 行都在文件末尾）
PROGRESS_SCAN_LINES = 200


def _file_signature(path: str | os.PathLike[str]) -> str:
    """文件的「指纹」（mtime_ns:size），用于判断日志是否增长过。"""
    try:
        info = Path(path).stat()
    except OSError:
        return ""
    return f"{info.st_mtime_ns}:{info.st_size}"


class Daemon:
    def __init__(
        self,
        cfg: Config,
        *,
        dry_run: bool = False,
        no_launch: bool = False,
        once: bool = False,
        console: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.no_launch = no_launch
        self.once = once
        self.started_at = time.time()

        self.logger = build_logger(
            "gpu_manager.daemon",
            cfg.log_file,
            level="INFO",
            rotate_mb=cfg.storage.daemon_log_max_mb,
            backups=cfg.storage.daemon_log_backups,
            console=console,
        )
        self.store = StateStore(cfg.state_file, self.logger)
        self.queue = QueueManager(cfg, self.logger)
        self.runner = TaskRunner(cfg, self.logger)
        self.notifier: Notifier = build_notifier(cfg.feishu, self.logger, dry_run=dry_run)
        self.tracker = IdleTracker(required_checks=cfg.gpu.idle_confirm_checks)
        self.watchdog = Watchdog(cfg, self.logger, self.notifier, dry_run=dry_run)

        self.state: dict[str, Any] = {}
        self.snapshot: Snapshot = Snapshot()
        self.handles: dict[str, RunHandle] = {}
        self.mode: str = MODE_IDLE
        self.mode_since: float = self.started_at
        self.last_plan: Plan = Plan()
        self.stats: dict[str, int] = {"launched": 0, "completed": 0, "failed": 0, "cancelled": 0}

        self.lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._reload_requested = False
        self._stop_deadlines: dict[str, float] = {}
        self._stop_sent: set[str] = set()
        self._stop_notified: set[str] = set()
        self._last_waiting_log = 0.0
        self._queue_import_errors_notified = 0.0
        self._last_prune = 0.0
        self._last_progress_sample = 0.0
        self.bot: FeishuBot | None = None
        self.server: ControlServer | None = None

    # ================================================================== #
    # 生命周期
    # ================================================================== #
    def run(self) -> int:
        cfg = self.cfg
        ensure_dir(cfg.state_file.parent)
        ensure_dir(cfg.task_log_dir)

        self.state = self.store.read()
        self.queue.load(self.state)
        self.tracker = IdleTracker.from_dict(
            self.state.get("idle"),
            cfg.gpu.idle_confirm_checks,
            max_age_seconds=cfg.gpu.check_interval_seconds * 2.0,
        )
        self.mode = str(self.state.get("mode") or MODE_IDLE)
        if self.mode not in (MODE_IDLE, MODE_QUEUE, MODE_RUNNING):
            self.mode = MODE_IDLE
        self.mode_since = float(self.state.get("mode_since") or self.started_at)
        self.stats = {**self.stats, **{k: int(v) for k, v in (self.state.get("stats") or {}).items() if k in self.stats}}

        self._install_signal_handlers()
        self._log_startup_banner()
        self._recover_after_restart()
        self.store.write_pid(cfg.pid_file, os.getpid())
        self.store.append_event(
            {"event": "daemon_start", "pid": os.getpid(), "version": __version__, "dry_run": self.dry_run},
            cfg.event_log,
        )

        if not self.once:
            server = ControlServer(
                cfg.socket_file, self.handle_command, self.logger, on_wake=self._wake.set
            )
            server.start()
            self.server = server
            self._start_bot()

        try:
            while not self._stopping.is_set():
                tick_start = time.time()
                try:
                    self.tick()
                except Exception:  # 单个 tick 出错绝不允许带走整个 daemon
                    self.logger.exception("tick 执行异常（daemon 继续运行）")
                if self.once:
                    break
                elapsed = time.time() - tick_start
                timeout = max(1.0, float(self.cfg.gpu.check_interval_seconds) - elapsed)
                self._wake.wait(timeout)
                self._wake.clear()
        except KeyboardInterrupt:
            self.logger.info("收到 KeyboardInterrupt，正在退出")
        finally:
            self._shutdown()
        return 0

    def request_stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def _install_signal_handlers(self) -> None:
        def on_term(signum: int, _frame: Any) -> None:
            self.logger.info("收到信号 %s，准备退出（不影响已启动的训练任务）", signum)
            self.request_stop()

        def on_hup(_signum: int, _frame: Any) -> None:
            self.logger.info("收到 SIGHUP，重新加载配置与 tasks.yaml")
            self._reload_requested = True
            self._wake.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, on_term)
        with contextlib.suppress(OSError, ValueError, AttributeError):
            signal.signal(signal.SIGHUP, on_hup)

    def _log_startup_banner(self) -> None:
        cfg = self.cfg
        self.logger.info("=" * 78)
        self.logger.info("GPU Manager v%s 启动  pid=%s", __version__, os.getpid())
        self.logger.info("服务器: %s  用户: %s", cfg.server.resolved_name(), cfg.server.resolved_user())
        self.logger.info("配置: %s", cfg.source_path or "(内置默认值)")
        self.logger.info(
            "空闲判定: util<=%s%% 且 mem<=%sMiB 且 无compute进程=%s; 连续确认 %s 次 × %ss",
            cfg.gpu.max_idle_utilization,
            cfg.gpu.max_idle_memory_mb,
            cfg.gpu.require_no_compute_process,
            cfg.gpu.idle_confirm_checks,
            cfg.gpu.check_interval_seconds,
        )
        self.logger.info(
            "GPU 范围: monitor=%s allowed=%s reserved=%s",
            cfg.gpu.monitor_gpu_ids or "all",
            cfg.gpu.scheduler_allowed_gpus or "all",
            cfg.gpu.reserved_gpu_ids or "无",
        )
        self.logger.info(
            "飞书通知: %s (enabled=%s, card=%s, 目标=%s)",
            self.notifier.name,
            cfg.feishu.enabled,
            cfg.feishu.card,
            (cfg.feishu.resolved_receive_id() or "未配置") if cfg.feishu.enabled else "-",
        )
        self.logger.info(
            "飞书交互查询: %s（app_id %s）",
            "已启用" if cfg.feishu.interactive_enabled else "已禁用",
            "已配置" if cfg.feishu.app_id else "未配置",
        )
        self.logger.info("状态文件: %s", cfg.state_file)
        self.logger.info("任务日志目录: %s", cfg.task_log_dir)
        self.logger.info(
            "看门狗: %s%s",
            cfg.watchdog.summary(),
            f"（诊断快照目录: {cfg.diagnostics_dir}）"
            if cfg.watchdog.enabled and cfg.watchdog.diagnostics_enabled
            else "",
        )
        if self.dry_run:
            self.logger.warning("--dry-run: 飞书消息只写日志，不实际发送")
        if self.no_launch:
            self.logger.warning("--no-launch: 只做调度决策，不真正启动任务")
        self.logger.info("安全边界: 不对其他用户进程发送任何信号；不清理显存；不 reset GPU")
        self.logger.info("=" * 78)

    # ================================================================== #
    # 重启恢复（只读校验，绝不发信号）
    # ================================================================== #
    def _recover_after_restart(self) -> None:
        running = self.queue.running_tasks()
        if not running:
            self.logger.info("无运行中任务，进入 %s 模式", self._compute_mode())
            return
        self.logger.info("发现 %d 个运行中任务，正在核对进程身份（只读，不发送任何信号）", len(running))
        for task in running:
            if not task.pid:
                self._finish_task(task, status=FAILED, exit_code=None, error="状态记录缺失 pid（daemon 重启前状态异常）")
                continue
            ok_identity, detail = self.runner.verify_identity(task)
            if ok_identity:
                self.logger.info(
                    "任务 %s 仍在运行: pid=%s GPU=%s 已运行 %s",
                    task.name, task.pid, task.gpu_text, fmt_duration(task.runtime_seconds()),
                )
                continue
            if process_alive(task.pid):
                # 进程活着但身份不符（PID 复用或换用户）→ 绝不触碰，仅解除跟踪
                self.logger.error(
                    "任务 %s 的 pid=%s 已不属于本进程（%s）——解除跟踪，绝不发送信号",
                    task.name, task.pid, detail,
                )
                self._finish_task(
                    task, status=FAILED, exit_code=None,
                    error=f"进程身份校验失败，已解除跟踪（{detail}）。若任务仍在跑请人工确认",
                )
                continue
            exit_code = self._read_persisted_exit_code(task)
            self.logger.warning(
                "任务 %s (pid=%s) 已不在运行，退出码=%s（daemon 重启期间结束）",
                task.name, task.pid, exit_code if exit_code is not None else "未知",
            )
            if exit_code is None:
                self._finish_task(
                    task, status=FAILED, exit_code=None,
                    error="进程已消失但没有退出码记录（daemon 曾在任务运行期间重启）",
                )
            else:
                self._finish_task(
                    task,
                    status=COMPLETED if exit_code == 0 else FAILED,
                    exit_code=exit_code,
                    error="" if exit_code == 0 else f"退出码 {exit_code}",
                    notify=False,
                )

    def _read_persisted_exit_code(self, task: Task) -> int | None:
        if not task.log_file:
            return None
        exit_path = Path(str(task.log_file)).with_suffix(".exit")
        try:
            text = exit_path.read_text(encoding="utf-8").strip()
        except OSError:
            # 也尝试 latest.exit 软链
            alt = exit_path.parent / "latest.exit"
            try:
                text = alt.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        return int(text) if text.lstrip("-").isdigit() else None

    # ================================================================== #
    # 主循环单步
    # ================================================================== #
    def tick(self) -> None:
        with self.lock:
            now = time.time()
            if self._reload_requested:
                self._reload_requested = False
                self._reload_config()

            # 1) GPU 采集
            self.snapshot = collect(timeout=15.0, include_processes=True)
            if self.snapshot.ok:
                self.logger.debug("GPU 采集成功: %d 张卡", len(self.snapshot.gpus))
            else:
                self.logger.warning("GPU 采集失败: %s", self.snapshot.error)
                self._notify_gpu_error(self.snapshot.error, now)

            # 2) 收割任务（先收割再决策，模式才准确）
            self._reap_tasks(now)

            # 3) 停止请求（用户取消 / 超时）
            self._process_stop_requests(now)

            # 4) 同步 tasks.yaml
            self._import_queue_file(now)

            # 5) 连续空闲确认
            monitored = self._monitored_gpu_ids()
            idle_ids: list[int] = []
            if self.snapshot.ok:
                idle_ids = idle_gpu_ids(
                    self.snapshot,
                    max_utilization=self.cfg.gpu.max_idle_utilization,
                    max_memory_mb=self.cfg.gpu.max_idle_memory_mb,
                    require_no_process=self.cfg.gpu.require_no_compute_process,
                    monitor_gpus=monitored,
                )
                confirmed = self.tracker.update(idle_ids, monitored, now=now)
            else:
                confirmed = []
                self.logger.debug("采集失败，本轮不推进空闲确认计数")

            # 6) 调度
            self._launch_pending(confirmed, now)

            # 7) 模式与通知
            self._update_mode(now)
            self._notify_idle_gpus(confirmed, now)
            self._maybe_notify_queue_wait(now)

            # 8) 日志轮转
            for handle in list(self.handles.values()):
                if handle.rotate_if_needed(self.cfg.runner.max_log_mb):
                    self.logger.info("任务 %s 日志已轮转（超过 %sMB）", handle.task_name, self.cfg.runner.max_log_mb)

            # 9) 周期性维护: 进度采样（供 ETA）+ 看门狗 + 日志清理（节流）
            self._sample_progress(now)
            self._run_watchdog(now)
            self._maybe_prune(now)

            # 10) 持久化
            self._persist(now)

    # ------------------------------------------------------------------ #
    def _monitored_gpu_ids(self) -> list[int]:
        return self.cfg.gpu.monitor_set(self.snapshot.all_indices())

    def _compute_mode(self) -> str:
        if any(t.is_running for t in self.queue.tasks.values()):
            return MODE_RUNNING
        if any(t.is_pending for t in self.queue.tasks.values()):
            return MODE_QUEUE
        return MODE_IDLE

    def _update_mode(self, now: float) -> None:
        new_mode = self._compute_mode()
        if new_mode == self.mode:
            return
        old_mode = self.mode
        self.mode = new_mode
        self.mode_since = now
        self.logger.info("模式切换: %s → %s", old_mode, new_mode)
        self.store.append_event(
            {"event": "mode_change", "from": old_mode, "to": new_mode}, self.cfg.event_log
        )
        if new_mode == MODE_QUEUE and self.cfg.notify.on_task_queued:
            pending = self.queue.pending_tasks()
            counts = self.queue.counts()
            body = [
                f"**服务器**: {self.cfg.server.resolved_name()}",
                f"**待执行任务**: {counts.get(PENDING, 0)} 个",
                f"**运行中任务**: {counts.get(RUNNING, 0)} 个",
                "",
                "当前无法满足资源:" if self.last_plan.waiting else "",
                self.last_plan.waiting_text(4) if self.last_plan.waiting else "",
                "",
                "需要 GPU 的任务:",
                *[
                    f"- `{t.name}` 需要 {t.gpu_num} 卡 (priority={t.priority})"
                    for t in pending[:5]
                ],
            ]
            self._notify(
                "⏳ 任务进入排队（等待空闲 GPU）",
                "\n".join(x for x in body if x is not None),
                level="warning",
            )

    # ------------------------------------------------------------------ #
    # 任务收割
    # ------------------------------------------------------------------ #
    def _check_timeout(self, task: Task, now: float) -> bool:
        """运行超时则请求停止（只针对自己启动的进程）。返回是否刚刚触发。

        提成独立方法有两个好处: 可单测；且保证「daemon 重启后接管的任务」
        也照样会被 timeout_seconds 约束。
        """
        if task.stop_requested or not task.timeout_seconds:
            return False
        runtime = task.runtime_seconds(now)
        if runtime is None or runtime <= task.timeout_seconds:
            return False
        self.logger.warning(
            "任务 %s 运行 %s 超过 timeout_seconds=%s，请求停止",
            task.name, fmt_duration(runtime), task.timeout_seconds,
        )
        task.stop_requested = True
        task.stop_requested_at = now
        task.stop_reason = "timeout"
        task.error = f"超时: 运行 {fmt_duration(runtime)} 超过 timeout_seconds={task.timeout_seconds}"
        return True

    def _reap_tasks(self, now: float) -> None:
        for task in self.queue.running_tasks():
            handle = self.handles.get(task.name)
            exit_code: int | None = None
            finished = False

            if handle is not None:
                exit_code = handle.poll()
                if exit_code is not None:
                    finished = True
                    self.handles.pop(task.name, None)
                    handle.flush_close()
                    recorded = handle.read_exit_code()
                    if recorded is not None:
                        exit_code = recorded
            elif not (task.pid and process_alive(task.pid)):
                # 没有 handle（daemon 重启过）且进程已消失 → 判定结束
                finished = True
                exit_code = self._read_persisted_exit_code(task)

            if not finished:
                # 仍在运行: 检查运行超时。
                # 注意这里不能提前 continue，否则 daemon 重启后 timeout_seconds 会失效。
                self._check_timeout(task, now)
                continue

            # 判定终态
            if task.stop_requested:
                if task.stop_reason == "timeout":
                    status = FAILED
                    error = task.error or f"超时（timeout_seconds={task.timeout_seconds}）"
                else:
                    status = CANCELLED
                    error = task.error or "用户取消"
            elif exit_code == 0:
                status = COMPLETED
                error = ""
            else:
                status = FAILED
                error = task.error or (f"退出码 {exit_code}" if exit_code is not None else "进程结束但退出码未知")

            # 自动重试（超时不重试，避免无限循环）
            retry = (
                status == FAILED
                and task.stop_reason != "timeout"
                and task.retries < task.max_retries
                and not task.stop_requested
            )
            if retry:
                task.retries += 1
                task.status = PENDING
                task.pid = None
                task.pgid = None
                task.proc_start_time = None
                task.gpu_ids = []
                task.started_at = None
                task.finished_at = None
                task.exit_code = exit_code
                task.error = ""
                task.retry_not_before = now + task.retry_delay_seconds
                self.logger.warning(
                    "任务 %s 失败（退出码 %s），将在 %ss 后第 %d/%d 次重试",
                    task.name, exit_code, task.retry_delay_seconds, task.retries, task.max_retries,
                )
                self._notify(
                    f"🔁 任务失败，将自动重试: {task.name}",
                    self._task_notify_body(task, extra=f"退出码: {exit_code}\n重试: 第 {task.retries}/{task.max_retries} 次，"
                                                         f"{task.retry_delay_seconds}s 后"),
                    level="warning",
                )
                self.store.append_event(
                    {"event": "task_retry", "task": task.name, "exit_code": exit_code, "attempt": task.retries},
                    self.cfg.event_log,
                )
                continue

            self._finish_task(task, status=status, exit_code=exit_code, error=error)

    def _finish_task(
        self,
        task: Task,
        *,
        status: str,
        exit_code: int | None,
        error: str = "",
        notify: bool = True,
    ) -> None:
        previous = task.status
        self.queue.mark_finished(task, status=status, exit_code=exit_code, error=error)
        handle = self.handles.pop(task.name, None)
        if handle is not None:
            handle.flush_close()
        self.handles.pop(task.name, None)
        self._stop_deadlines.pop(task.name, None)
        self._stop_sent.discard(task.name)
        self._stop_notified.discard(task.name)
        self.tracker.reset(task.gpu_ids)
        self.stats[status] = self.stats.get(status, 0) + 1

        duration = fmt_duration(task.runtime_seconds())
        self.logger.info(
            "任务 %s 结束: %s (退出码=%s, 用时=%s, GPU=%s)",
            task.name, status, exit_code, duration, task.gpu_text,
        )
        self.store.append_event(
            {
                "event": "task_finished", "task": task.name, "status": status,
                "exit_code": exit_code, "duration_seconds": task.runtime_seconds(),
                "gpu_ids": task.gpu_ids, "previous_status": previous,
            },
            self.cfg.event_log,
        )
        if not notify:
            return

        label = {COMPLETED: "任务完成", FAILED: "任务失败", CANCELLED: "任务已取消"}[status]
        if status == COMPLETED and not self.cfg.notify.on_task_finished:
            return
        if status == FAILED and not self.cfg.notify.on_task_failed:
            return
        if status == CANCELLED and not self.cfg.notify.on_task_finished:
            return

        level, icon, label = TERMINAL_STYLE.get(status, ("info", "ℹ️", "任务结束"))

        extra_lines = [f"退出码: {exit_code if exit_code is not None else '未知'}", f"用时: {duration}"]
        if error:
            extra_lines.append(f"原因: {error}")
        if status == FAILED:
            tail = self.runner.tail_log(task, 12)
            if tail:
                extra_lines.append("\n最后日志:\n" + truncate("\n".join(tail), 1000))
        body = self._task_notify_body(task, extra="\n".join(extra_lines), include_gpus=True)
        self._notify(
            f"{icon} {label}: {task.name}",
            body,
            level=level,
            mention_all=(status == FAILED and self.cfg.notify.mention_all_on_failure),
        )

    def _task_notify_body(self, task: Task, *, extra: str = "", include_gpus: bool = True) -> str:
        lines = [
            f"**服务器**: {self.cfg.server.resolved_name()}",
            f"**任务**: `{task.name}`",
        ]
        if include_gpus:
            gpu_text = task.gpu_text if task.gpu_ids else f"(需要 {task.gpu_num} 卡)"
            lines.append(f"**GPU**: {gpu_text}")
        if task.env:
            lines.append(f"**环境**: {task.env}")
        lines.append(f"**目录**: {task.cwd}")
        if self.mode in (MODE_IDLE, MODE_QUEUE):
            counts = self.queue.counts()
            lines.append(f"**队列**: pending={counts.get(PENDING, 0)} running={counts.get(RUNNING, 0)}")
        if extra:
            lines += ["", extra]
        if task.log_file:
            lines += ["", f"**日志**: `{task.log_file}`"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 停止请求（取消 / 超时）
    # ------------------------------------------------------------------ #
    def _process_stop_requests(self, now: float) -> None:
        for task in self.queue.running_tasks():
            if not task.stop_requested:
                continue
            if task.name not in self._stop_notified:
                self._stop_notified.add(task.name)
                self.logger.info(
                    "处理停止请求: %s (原因: %s, pid=%s)",
                    task.name, task.stop_reason or "用户取消", task.pid,
                )
            if task.name not in self._stop_sent:
                result = self.runner.send_stop_signal(task, hard=False)
                if result == "sent":
                    self._stop_sent.add(task.name)
                    self._stop_deadlines[task.name] = now + max(1.0, self.cfg.runner.term_grace_seconds)
                elif result == "gone":
                    self._stop_sent.add(task.name)
                    self._stop_deadlines[task.name] = now
                else:  # refused: 不降级、不重试，等下一轮继续校验
                    self._stop_deadlines[task.name] = now + 30.0
                continue
            deadline = self._stop_deadlines.get(task.name, 0.0)
            if now < deadline:
                continue
            if not self.cfg.runner.kill_after_grace:
                continue
            if task.pid and process_alive(task.pid):
                self.logger.warning("任务 %s 未在 grace 期内退出，发送 SIGKILL", task.name)
                self.runner.send_stop_signal(task, hard=True)
            self._stop_deadlines[task.name] = now + 10.0

    # ------------------------------------------------------------------ #
    # tasks.yaml 同步
    # ------------------------------------------------------------------ #
    def _import_queue_file(self, now: float) -> None:
        result: ImportResult = self.queue.import_queue_file()
        if result.errors and now - self._queue_import_errors_notified > 1800:
            self._queue_import_errors_notified = now
            self.logger.error("tasks.yaml 存在问题: %s", "; ".join(result.errors))
            if self.cfg.notify.on_error:
                self._notify(
                    "⚠️ tasks.yaml 解析/校验问题",
                    "\n".join(f"- {e}" for e in result.errors[:8]),
                    level="warning",
                )
        if not result.changed:
            return
        self.logger.info("tasks.yaml 同步: %s", result.summary())
        self.store.append_event({"event": "queue_import", "summary": result.summary()}, self.cfg.event_log)
        for name in result.added + result.replaced:
            task = self.queue.get(name)
            if task and self.cfg.notify.on_task_queued:
                self._notify(
                    "📥 任务已入队",
                    self._task_notify_body(task),
                    level="info",
                )

    # ------------------------------------------------------------------ #
    # 调度与启动
    # ------------------------------------------------------------------ #
    def _launch_pending(self, confirmed: list[int], now: float) -> None:
        pending = self.queue.pending_tasks()
        self.last_plan = Plan()
        if not pending:
            return
        if not self.snapshot.ok:
            self.last_plan.waiting = [(t.name, f"GPU 采集失败: {self.snapshot.error}") for t in pending]
            return

        all_ids = self.snapshot.all_indices()
        allowed = self.cfg.gpu.schedulable_set(all_ids)
        confirmed_set = set(confirmed)

        # 最近刚释放的 GPU 给一个 grace 期，避免残留内存/进程未退干净就复用
        recently_released: set[int] = set()
        grace = float(self.cfg.scheduler.grace_before_reuse_seconds)
        for task in self.queue.tasks.values():
            if task.is_terminal and task.finished_at and (now - task.finished_at) < grace:
                recently_released.update(task.gpu_ids)

        busy = self.queue.busy_gpu_ids()
        candidates: list[GpuCandidate] = []
        skip_detail: dict[int, str] = {}
        for gpu in self.snapshot.gpus:
            if gpu.index not in allowed:
                continue
            if gpu.index in busy:
                skip_detail[gpu.index] = "已分配给运行中任务"
                continue
            if gpu.index in recently_released:
                skip_detail[gpu.index] = f"刚释放 <{int(grace)}s"
                continue
            if gpu.index not in confirmed_set:
                counts = self.tracker.counts.get(gpu.index, 0)
                skip_detail[gpu.index] = f"空闲确认 {counts}/{self.cfg.gpu.idle_confirm_checks}"
                continue
            if gpu.foreign_processes:
                who = ", ".join(f"{p.name}(pid {p.pid})" for p in gpu.foreign_processes[:2])
                skip_detail[gpu.index] = f"他人占用: {who}"
                continue
            candidates.append(GpuCandidate(gpu_id=gpu.index, free_mb=gpu.memory_free_mb))

        running_count = len(self.queue.running_tasks())
        plan = plan_assignments(
            pending,
            candidates,
            running_count=running_count,
            max_concurrent=self.cfg.scheduler.max_concurrent_tasks,
            now=now,
            allow_cpu_tasks=self.cfg.scheduler.allow_cpu_tasks,
        )
        self.last_plan = plan

        if plan.waiting and now - self._last_waiting_log > WAITING_LOG_INTERVAL:
            self._last_waiting_log = now
            self.logger.info(
                "队列等待中 (%d 个任务): %s | 候选卡=%s | 排除原因=%s",
                len(plan.waiting), "; ".join(f"{n}:{r}" for n, r in plan.waiting[:4]),
                [c.gpu_id for c in candidates] or "无",
                "; ".join(f"GPU{k}:{v}" for k, v in sorted(skip_detail.items())[:6]) or "无",
            )

        if not plan.assignments:
            return
        self.logger.info("%s", describe_plan(plan))

        for assignment in plan.assignments:
            task = assignment.task
            gpu_ids = assignment.gpu_ids

            # 启动前二次检查——多人服务器必备，避免抢到刚被别人占用的卡
            if task.gpu_num > 0:
                if self.cfg.scheduler.launch_second_check:
                    frontier = collect(timeout=15.0, include_processes=True)
                    safe, detail = still_free(
                        frontier,
                        gpu_ids,
                        max_utilization=self.cfg.gpu.max_idle_utilization,
                        max_memory_mb=self.cfg.gpu.max_idle_memory_mb,
                        require_no_process=self.cfg.gpu.require_no_compute_process,
                    )
                    if not safe:
                        self.logger.warning("任务 %s 二次检查未通过，取消本次分配: %s", task.name, detail)
                        self.store.append_event(
                            {"event": "launch_aborted", "task": task.name, "gpu_ids": gpu_ids, "reason": detail},
                            self.cfg.event_log,
                        )
                        self.tracker.reset(gpu_ids)
                        continue
                    self.logger.info("任务 %s %s", task.name, detail)
                # 身份无关的最后防重：确认没有别的运行任务占着这些卡
                conflict = self.queue.busy_gpu_ids() & set(gpu_ids)
                if conflict:
                    self.logger.error(
                        "任务 %s 分配冲突（GPU %s 已被占用），跳过本次启动",
                        task.name, gpu_ids_text(conflict),
                    )
                    continue
                # 显存下限复核
                if task.min_free_memory_mb:
                    gpu_map = self.snapshot.by_index()
                    low = [
                        g for g in gpu_ids
                        if g in gpu_map and gpu_map[g].memory_free_mb < task.min_free_memory_mb
                    ]
                    if low:
                        self.logger.warning(
                            "任务 %s 要求单卡空闲显存 >= %sMiB，但 GPU %s 不满足，跳过本轮",
                            task.name, task.min_free_memory_mb, gpu_ids_text(low),
                        )
                        continue

            if self.no_launch:
                self.logger.warning(
                    "[no-launch] 本应启动 %s → GPU %s（仅决策，不执行）",
                    task.name, gpu_ids_text(gpu_ids),
                )
                self.tracker.reset(gpu_ids)
                continue

            self._start_task(task, gpu_ids)

    def _start_task(self, task: Task, gpu_ids: list[int]) -> None:
        try:
            handle = self.runner.launch(task, gpu_ids)
        except RunnerError as exc:
            self.logger.error("启动任务 %s 失败: %s", task.name, exc)
            self._finish_task(
                task, status=FAILED, exit_code=None, error=f"启动失败: {exc}", notify=True
            )
            # 启动失败多为环境/目录问题，与 GPU 无关 → 不给重试
            task.max_retries = min(task.max_retries, task.retries)
            return

        self.handles[task.name] = handle
        self.queue.mark_running(
            task,
            gpu_ids=gpu_ids,
            pid=handle.pid,
            pgid=handle.pgid,
            proc_start_time=handle.proc_start_time,
            log_file=str(handle.log_path),
        )
        self.tracker.reset(gpu_ids)
        self.stats["launched"] = self.stats.get("launched", 0) + 1
        self.store.append_event(
            {"event": "task_started", "task": task.name, "gpu_ids": gpu_ids, "pid": handle.pid},
            self.cfg.event_log,
        )
        if self.cfg.notify.on_task_started:
            body = self._task_notify_body(
                task,
                extra=f"PID: {handle.pid}\n开始时间: {fmt_ts(handle.started_at)}",
            )
            self._notify(f"🚀 任务已启动: {task.name}", body, level="success")

    # ------------------------------------------------------------------ #
    # 通知
    # ------------------------------------------------------------------ #
    def _notify(self, title: str, body: str, level: str = "info", mention_all: bool = False) -> None:
        try:
            ok_sent, detail = self.notifier.send(title, body, level, mention_all)
        except Exception as exc:  # 通知层问题绝不影响调度
            self.logger.exception("通知发送异常: %s", exc)
            return
        if not ok_sent:
            self.logger.warning("通知发送失败: %s", detail)

    def _notify_idle_gpus(self, confirmed: list[int], now: float) -> None:
        """空闲 GPU 提醒。

        IDLE 模式: 上报所有确认空闲的受监控 GPU（这就是「新 GPU 空闲」提醒）。
        QUEUE/RUNNING 模式: 只上报「调度器不会用、但你可以手动用」的卡
        （例如不在 scheduler_allowed_gpus 里的 GPU 0），避免与调度动作互相刷屏。
        """
        notify_state = self.state.setdefault("notify", {})
        if not self.cfg.notify.on_gpu_idle:
            return

        if self.mode == MODE_IDLE:
            wanted = set(confirmed)
        else:
            allowed = set(self.cfg.gpu.schedulable_set(self.snapshot.all_indices()))
            wanted = {g for g in confirmed if g not in allowed}

        if not wanted:
            notify_state["last_idle_signature"] = ""
            return
        if len(wanted) < self.cfg.notify.min_idle_gpus_to_notify:
            return

        signature = ",".join(str(g) for g in sorted(wanted))
        if signature == notify_state.get("last_idle_signature"):
            return
        last_at = float(notify_state.get("gpu_idle_at") or 0.0)
        cooldown = float(self.cfg.notify.idle_cooldown_seconds)
        if cooldown > 0 and now - last_at < cooldown:
            self.logger.debug("空闲通知冷却中（剩余 %.0fs），跳过", cooldown - (now - last_at))
            return

        rows = {r["index"]: r for r in as_dict_rows(self.snapshot, sorted(wanted))}
        lines = [
            f"**服务器**: {self.cfg.server.resolved_name()}",
            f"**空闲 GPU**: {gpu_ids_text(sorted(wanted))}（连续 {self.cfg.gpu.idle_confirm_checks} 次检查确认）",
            "",
        ]
        for gpu_id in sorted(wanted):
            row = rows.get(gpu_id)
            if not row:
                continue
            lines.append(
                f"- GPU{gpu_id}: util {row['utilization']}% · "
                f"显存 {row['memory_used_mb']}/{row['memory_total_mb']} MiB"
            )
        counts = self.queue.counts()
        lines += [
            "",
            f"**队列**: pending={counts.get(PENDING, 0)} running={counts.get(RUNNING, 0)}",
        ]
        if self.mode == MODE_IDLE:
            lines.append("当前没有待执行任务，可以提交新的训练了。")
        else:
            lines.append("这些卡不在自动调度范围内（reserved 或未列入 scheduler_allowed_gpus）。")

        self._notify(
            "🟢 GPU 空闲: " + gpu_ids_text(sorted(wanted)),
            "\n".join(lines),
            level="info",
            mention_all=self.cfg.notify.mention_all_on_idle,
        )
        notify_state["gpu_idle_at"] = now
        notify_state["last_idle_signature"] = signature

    def _maybe_notify_queue_wait(self, now: float) -> None:
        """排队久了给一次周期提醒，避免用户以为程序挂了。"""
        if self.mode != MODE_QUEUE or not self.cfg.scheduler.queue_notify:
            return
        interval = float(self.cfg.scheduler.waiting_notify_interval_seconds)
        if interval <= 0:
            return
        notify_state = self.state.setdefault("notify", {})
        last_at = float(notify_state.get("queue_wait_at") or 0.0)
        if now - last_at < interval:
            return
        if last_at == 0.0:
            # 刚进入排队时已经发过进入通知，从下一周期开始计时
            notify_state["queue_wait_at"] = now
            return

        pending = self.queue.pending_tasks()
        if not pending:
            return
        occupying = self._gpu_occupancy_summary()
        body = [
            f"**服务器**: {self.cfg.server.resolved_name()}",
            f"**已等待**: {fmt_duration(now - self.mode_since)}",
            f"**待执行**: {len(pending)} 个任务",
            "",
            "**等待原因**:",
            self.last_plan.waiting_text(6) if self.last_plan.waiting else "  资源检查中",
        ]
        if occupying:
            body += ["", "**当前 GPU 占用**:", occupying]
        self._notify(
            f"⏳ 队列仍在等待（{len(pending)} 个任务）",
            "\n".join(body),
            level="warning",
        )
        notify_state["queue_wait_at"] = now

    def _gpu_occupancy_summary(self, limit: int = 4) -> str:
        if not self.snapshot.ok:
            return ""
        lines: list[str] = []
        for gpu in self.snapshot.gpus:
            if not gpu.compute_processes:
                continue
            who = ", ".join(
                f"pid{p.pid}({'自己' if p.ours else p.min_user})" for p in gpu.compute_processes[:2]
            )
            lines.append(f"- GPU{gpu.index}: util {gpu.utilization}% · {who}")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def _notify_gpu_error(self, error: str, now: float) -> None:
        if not self.cfg.notify.on_error:
            return
        notify_state = self.state.setdefault("notify", {})
        last_at = float(notify_state.get("last_error_at") or 0.0)
        cooldown = max(3600.0, float(self.cfg.notify.idle_cooldown_seconds))
        if now - last_at < cooldown:
            return
        notify_state["last_error_at"] = now
        self._notify(
            "⚠️ GPU 状态采集异常",
            f"**服务器**: {self.cfg.server.resolved_name()}\n\n`nvidia-smi` 读取失败:\n```\n{truncate(error, 500)}\n```\n"
            "本轮不会做任何调度决策（保守处理，绝不盲目占卡）。",
            level="warning",
        )

    # ================================================================== #
    # 持久化
    # ================================================================== #
    # ------------------------------------------------------------------ #
    # 进度采样（供 ETA）
    # ------------------------------------------------------------------ #
    def _sample_progress(self, now: float) -> None:
        """定期解析运行中任务的进度，存进 task.progress 供 ETA 使用。

        节流到 ``feishu.eta_sample_interval_seconds``；只读日志尾部若干行。
        **只有在交互查询总开关打开时才采样**：关掉的功能应当零开销。

        进度来源优先级: 任务声明的 ``eta_logs``（训练脚本自己重定向的输出，
        如 console.log）→ 我们自己捕获的日志。``total_epochs`` 用于换算
        「第几轮 + 轮内进度」型日志；``stage_file`` 变化时清空旧进度。
        """
        fcfg = self.cfg.feishu
        if not (fcfg.interactive_enabled and fcfg.eta_enabled):
            return
        interval = max(5.0, float(fcfg.eta_sample_interval_seconds))
        if now - self._last_progress_sample < interval:
            return
        self._last_progress_sample = now

        patterns = tuple(fcfg.eta_patterns) or None
        for task in self.queue.running_tasks():
            self._refresh_stage(task)
            source, lines, signature = self._read_progress_source(task)
            if not lines or not source:
                continue
            # 源文件自上次采样以来没有变化 → 没有新信息，不记新样本。
            # （否则会把旧进度配上新时间戳，在 eval 等阶段用训练速度外推。）
            if task.progress_source == source and task.progress_source_sig == signature:
                continue
            task.progress_source = source
            task.progress_source_sig = signature

            sample = eta.parse_progress(
                lines, patterns, now=now, total_epochs=task.total_epochs or None
            )
            if sample is None:
                continue
            # 进度数值本身也没推进（例如日志只多了无关行）→ 同样不记
            _key, previous = eta.latest_samples(task.progress)
            if previous and sample.done is not None and previous[-1].done == sample.done:
                continue
            task.progress = eta.record_sample(task.progress, sample)
            task.progress_updated_at = now

    def _refresh_stage(self, task: Task) -> None:
        """读取用户声明的阶段标记（如 stage.status）；阶段变化时丢弃旧进度。

        训练脚本通常是「先训 8 卡、再单卡 eval」这类多阶段流程；
        若把训练阶段的进度外推到 eval 阶段，ETA 会完全错误 —— 因此阶段一变就重置。
        """
        if not task.stage_file:
            return
        raw = task.stage_file
        path = Path(raw) if os.path.isabs(raw) else Path(task.cwd) / raw
        lines = self.runner.tail_path(path, 1)
        stage = lines[-1].strip()[:32] if lines else ""
        if stage and stage != task.stage:
            self.logger.info(
                "任务 %s 阶段变化: %s → %s（丢弃旧进度样本）",
                task.name, task.stage or "-", stage,
            )
            task.stage = stage
            task.progress = {}

    def _read_progress_source(self, task: Task) -> tuple[str, list[str], str]:
        """按优先级取进度文本: eta_logs（用户声明）→ 我们捕获的日志。

        返回 ``(来源路径, 文本行, 文件指纹)``；指纹用于判断源文件是否已增长。
        """
        for raw in task.eta_logs:
            candidate = Path(raw) if os.path.isabs(raw) else Path(task.cwd) / raw
            lines = self.runner.tail_path(candidate, PROGRESS_SCAN_LINES)
            if lines:
                return str(candidate), lines, _file_signature(candidate)
        if task.log_file:
            lines = self.runner.tail_log(task, PROGRESS_SCAN_LINES)
            if lines:
                return task.log_file, lines, _file_signature(task.log_file)
        return "", [], ""

    # ------------------------------------------------------------------ #
    # 看门狗（低频；只在有任务时工作）
    # ------------------------------------------------------------------ #
    def _run_watchdog(self, now: float) -> None:
        """确认运行中任务是否还在推进；停滞/进程异常 → 告警 + 诊断快照。

        没有运行中任务时看门狗自动关闭（零开销），有新任务时自动恢复。
        判定与节流逻辑全在 watchdog.Watchdog 里（纯只读：/proc 与文件 stat）。
        """
        events = self.watchdog.update(
            self.queue.running_tasks(),
            now=now,
            snapshot=self.snapshot,
            eta_provider=lambda task: self._watchdog_eta_text(task, now),
        )
        for event in events:
            self.store.append_event(
                {
                    "event": event.event_name,
                    "task": event.task,
                    "kind": event.kind,
                    "level": event.level,
                    "title": event.title,
                    "detail": truncate(event.detail, 1200),
                    "snapshot": event.snapshot_dir,
                    "notified": event.notified,
                },
                self.cfg.event_log,
            )

    def _watchdog_eta_text(self, task: Task, now: float) -> str:
        """诊断快照里的 ETA 文本。

        只在真的要写快照时调用（告警才算），所以可以负担一次事件流读取；
        每 tick 的热路径上不做这件事。
        """
        if not self.cfg.feishu.eta_enabled:
            return ""
        history_seconds: float | None = None
        history_runs = 0
        if self.cfg.feishu.eta_history_fallback:
            history_seconds, history_runs = eta.median_history(
                StateStore.read_events(self.cfg.event_log, 500),
                task.name,
                limit=self.cfg.feishu.eta_history_max_runs,
            )
        result = eta.estimate(
            task.progress,
            now=now,
            history_seconds=history_seconds,
            history_runs=history_runs,
            started_at=task.started_at,
        )
        return eta.format_eta(result, now=now)

    # ------------------------------------------------------------------ #
    # 飞书交互式查询（只读）
    # ------------------------------------------------------------------ #
    def _bot_log_tail(self, name: str, lines: int) -> list[str] | None:
        """bot 读日志的唯一入口: 按任务名查本项目自己的日志（runner 内含路径包含校验）。"""
        task = self.queue.get(name)
        if task is None:
            return None
        return self.runner.tail_log(task, lines)

    def _bot_context(self) -> BotContext:
        """在 daemon 锁内组装一份状态快照，供 bot 线程只读使用。

        回复飞书的**网络 I/O 在锁外**发生，因此飞书变慢不会卡住调度主循环。
        """
        with self.lock:
            now = time.time()
            cfg = self.cfg
            snapshot = self.snapshot
            schedulable = set(cfg.gpu.schedulable_set(snapshot.all_indices()))
            running_tasks = [t for t in self.queue.tasks.values() if t.is_running]
            waiting = dict(self.last_plan.waiting)

            gpu_views: list[GpuView] = []
            for gpu in snapshot.gpus:
                holder = next((t.name for t in running_tasks if gpu.index in t.gpu_ids), "")
                gpu_views.append(
                    GpuView(
                        index=gpu.index,
                        utilization=gpu.utilization,
                        memory_used_mb=gpu.memory_used_mb,
                        memory_total_mb=gpu.memory_total_mb,
                        schedulable=gpu.index in schedulable,
                        idle_streak=self.tracker.counts.get(gpu.index, 0),
                        idle_required=self.tracker.required_checks,
                        own_pids=[p.pid for p in gpu.own_processes],
                        foreign_pids=[p.pid for p in gpu.foreign_processes],
                        foreign_uids=sorted(
                            {p.uid for p in gpu.foreign_processes if p.uid is not None}
                        ),
                        task_name=holder,
                    )
                )

            # 历史耗时（ETA 兜底）: 事件流读一次，避免每个任务读一遍
            history: dict[str, tuple[float | None, int]] = {}
            if cfg.feishu.eta_enabled and cfg.feishu.eta_history_fallback:
                events = StateStore.read_events(cfg.event_log, 500)
                for task in self.queue.tasks.values():
                    if task.name not in history:
                        history[task.name] = eta.median_history(
                            events, task.name, limit=cfg.feishu.eta_history_max_runs
                        )

            task_views: list[TaskView] = []
            for task in self.queue.ordered_tasks():
                eta_text = ""
                confidence = "none"
                if task.is_running and cfg.feishu.eta_enabled:
                    hist_seconds, hist_runs = history.get(task.name, (None, 0))
                    result = eta.estimate(
                        task.progress,
                        now=now,
                        history_seconds=(
                            hist_seconds if cfg.feishu.eta_history_fallback else None
                        ),
                        history_runs=hist_runs,
                        started_at=task.started_at,
                    )
                    eta_text = eta.format_eta(result, now=now)
                    confidence = result.confidence
                task_views.append(
                    TaskView(
                        name=task.name,
                        status=task.status,
                        gpu_ids=list(task.gpu_ids),
                        gpu_num=task.gpu_num,
                        priority=task.priority,
                        started_at=task.started_at,
                        finished_at=task.finished_at,
                        runtime_seconds=task.runtime_seconds(now),
                        wait_seconds=task.wait_seconds(now),
                        progress_text=eta.format_progress(task.progress) if task.progress else "",
                        eta_text=eta_text,
                        eta_confidence=confidence,
                        error=task.error,
                        waiting_reason=waiting.get(task.name, ""),
                        stage=task.stage,
                    )
                )

            return BotContext(
                server=cfg.server.resolved_name(),
                mode=self.mode,
                counts=self.queue.counts(),
                gpus=gpu_views,
                tasks=task_views,
                daemon_pid=os.getpid(),
                uptime_seconds=max(0.0, now - self.started_at),
                gpu_error=snapshot.error,
                log_reader=self._bot_log_tail,
            )

    def _bot_audit(self, record: dict[str, Any]) -> None:
        """入站指令审计（谁、何时、要了什么）。"""
        self.store.append_event(record, self.cfg.event_log)

    def _start_bot(self) -> None:
        """启动飞书交互式查询。任何问题都只记日志，绝不影响调度。"""
        fcfg = self.cfg.feishu
        if not fcfg.interactive_enabled:
            return
        if self.dry_run:
            self.logger.info("--dry-run: 不启动飞书交互查询")
            return
        for warning in fcfg.interactive_warnings():
            self.logger.warning("%s", warning)
        if not (fcfg.app_id and fcfg.app_secret):
            return
        try:
            transport = FeishuTransport(
                fcfg.app_id,
                fcfg.app_secret,
                self.logger,
                timeout=float(fcfg.timeout_seconds),
                allowed_hosts=fcfg.allowed_hosts,
                allow_any_host=fcfg.allow_any_host,
            )
            bot = FeishuBot(
                app_id=fcfg.app_id,
                app_secret=fcfg.app_secret,
                router=CommandRouter(log_lines_default=fcfg.log_lines_default),
                context_provider=self._bot_context,
                transport=transport,
                allowlist=AllowList(fcfg.allowed_users),
                rate_limiter=RateLimiter(fcfg.rate_limit_per_minute),
                logger=self.logger,
                audit=self._bot_audit,
            )
        except Exception:
            self.logger.exception("初始化飞书交互查询失败（继续运行调度）")
            return
        try:
            started = bot.start()
        except Exception:
            # 双重保险: 任何失败（含 SDK 装了但加载不了，例如 setuptools 与 Python 不兼容）
            # 都只降级记日志 —— 绝不能让一个可选功能把 daemon 拖到起不来。
            self.logger.exception("启动飞书交互查询失败（继续运行调度）")
            return
        if started:
            self.bot = bot
            self.logger.info(
                "飞书交互查询: 已授权 %d 人，限流 %d 条/分钟，ETA 采样间隔 %ss",
                len(fcfg.allowed_users),
                fcfg.rate_limit_per_minute,
                fcfg.eta_sample_interval_seconds,
            )

    def _stop_bot(self) -> None:
        if self.bot is not None:
            with contextlib.suppress(Exception):
                self.bot.stop()
            self.bot = None

    def _maybe_prune(self, now: float) -> None:
        """按 storage 配置清理日志，保护磁盘。

        节流执行（默认 5 分钟一次），且**正在运行任务的日志绝不删**。
        总量超限时优先删成功任务的旧日志，失败任务的留久一点。
        """
        interval = self.cfg.storage.prune_interval_seconds
        if interval <= 0:
            return
        if now - self._last_prune < interval:
            return
        self._last_prune = now
        try:
            active = [
                t.log_file for t in self.queue.tasks.values() if t.is_running and t.log_file
            ]
            failed = [t.safe_name for t in self.queue.tasks.values() if t.status == FAILED]
            prune_all(
                self.cfg,
                self.logger,
                active_log_paths=active,
                failed_task_names=failed,
            )
        except Exception:  # 清理失败绝不影响调度主流程
            self.logger.exception("日志清理异常（已忽略）")

    def _persist(self, now: float) -> None:
        self.state["daemon_pid"] = os.getpid()
        self.state["daemon_version"] = __version__
        self.state["started_at"] = self.started_at
        self.state["mode"] = self.mode
        self.state["mode_since"] = self.mode_since
        self.state["stats"] = self.stats
        self.state["idle"] = self.tracker.to_dict()
        self.state["watchdog"] = self.watchdog.to_dict()
        self.state["waiting"] = [
            {"name": name, "reason": reason} for name, reason in self.last_plan.waiting
        ]
        self.state["last_snapshot"] = {
            "timestamp": self.snapshot.timestamp,
            "error": self.snapshot.error,
            "gpus": as_dict_rows(self.snapshot),
        }
        self.state["idle_report"] = (
            {
                str(gid): reasons
                for gid, reasons in idle_report(
                    self.snapshot,
                    max_utilization=self.cfg.gpu.max_idle_utilization,
                    max_memory_mb=self.cfg.gpu.max_idle_memory_mb,
                    require_no_process=self.cfg.gpu.require_no_compute_process,
                    monitor_gpus=self._monitored_gpu_ids(),
                ).items()
            }
            if self.snapshot.ok
            else {}
        )
        self.queue.dump(self.state)
        self.store.write(self.state)

    # ================================================================== #
    # 配置重载
    # ================================================================== #
    def _reload_config(self) -> None:
        source = self.cfg.source_path
        if source is None:
            self.logger.warning("无配置文件来源，跳过重载")
            return
        try:
            new_cfg = load_config(source)
        except ConfigError as exc:
            self.logger.error("配置重载失败（保持原配置）: %s", exc)
            if self.cfg.notify.on_error:
                self._notify("⚠️ 配置重载失败", f"```\n{truncate(str(exc), 600)}\n```\n已继续使用原配置。", "warning")
            return

        old_cfg = self.cfg
        self.cfg = new_cfg
        self.queue.cfg = new_cfg
        self.runner.cfg = new_cfg
        self.tracker.required_checks = new_cfg.gpu.idle_confirm_checks
        self.notifier.close()
        self.notifier = build_notifier(new_cfg.feishu, self.logger, dry_run=self.dry_run)
        # 看门狗：换阈值但保留告警历史（避免重载后重复告警）；
        # 同时把引用指向新 notifier（旧对象已 close）。
        self.watchdog.reconfigure(new_cfg)
        self.watchdog.notifier = self.notifier

        # 交互查询: 只有「连接参数」变了才需要重建长连接。
        # 白名单/限流/日志行数属于「处理消息时才读」的设置，**原地更新**即可生效 ——
        # 这很重要: lark-oapi 的 ws.Client 没有公开的 stop，重建会留下旧连接
        # 继续消费事件（并被旧白名单拒绝），且旧连接无法可靠关闭。
        restart_bot = (
            old_cfg.feishu.interactive_enabled != new_cfg.feishu.interactive_enabled
            or old_cfg.feishu.app_id != new_cfg.feishu.app_id
            or old_cfg.feishu.app_secret != new_cfg.feishu.app_secret
        )
        if restart_bot:
            self._stop_bot()
            self._start_bot()
        elif self.bot is not None:
            self.bot.allowlist.replace(new_cfg.feishu.allowed_users)
            self.bot.rate_limiter.per_minute = max(1, new_cfg.feishu.rate_limit_per_minute)
            self.bot.router.log_lines_default = max(1, new_cfg.feishu.log_lines_default)
            self.logger.info(
                "交互查询设置已原地更新（不重建长连接）: 已授权 %d 人，限流 %d/分钟",
                len(self.bot.allowlist.allowed),
                self.bot.rate_limiter.per_minute,
            )
        self.logger.info("配置已重载: %s", source)
        changed = self._config_diff(old_cfg, new_cfg)
        if changed:
            self.logger.info("配置变化: %s", "; ".join(changed))
            self.store.append_event({"event": "config_reload", "changes": changed}, new_cfg.event_log)
        if str(old_cfg.log_file) != str(new_cfg.log_file):
            self.logger.warning(
                "log_file 变更（%s → %s）需要重启 daemon 才能生效", old_cfg.log_file, new_cfg.log_file
            )

    @staticmethod
    def _config_diff(old: Config, new: Config) -> list[str]:
        out: list[str] = []
        a, b = old.as_dict(), new.as_dict()
        for section in sorted(set(a) | set(b)):
            va, vb = a.get(section), b.get(section)
            if isinstance(va, dict) and isinstance(vb, dict):
                for key in sorted(set(va) | set(vb)):
                    if va.get(key) != vb.get(key):
                        out.append(f"{section}.{key}: {va.get(key)!r} → {vb.get(key)!r}")
            elif va != vb:
                out.append(f"{section}: {va!r} → {vb!r}")
        return out

    # ================================================================== #
    # IPC 命令
    # ================================================================== #
    def handle_command(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            if command == "ping":
                return ok({"pid": os.getpid(), "version": __version__, "mode": self.mode})
            if command == "status":
                return ok(self.status_payload())
            if command == "add":
                spec = params.get("spec") or {}
                if not isinstance(spec, dict):
                    return fail("add 需要 spec 参数")
                try:
                    task = self.queue.add(spec, source="cli", replace=bool(params.get("replace")))
                except TaskError as exc:
                    return fail(str(exc))
                self._persist(now)
                self.logger.info("CLI 新增任务: %s (gpu_num=%s priority=%s)", task.name, task.gpu_num, task.priority)
                self.store.append_event({"event": "task_added", "task": task.name, "source": "cli"}, self.cfg.event_log)
                if self.cfg.notify.on_task_queued:
                    self._notify("📥 任务已入队", self._task_notify_body(task), level="info")
                return ok({"task": task.to_dict()})
            if command == "cancel":
                name = str(params.get("name", ""))
                try:
                    task = self.queue.cancel(name)
                except TaskError as exc:
                    return fail(str(exc))
                task.stop_requested_at = now if task.stop_requested else 0.0
                if task.stop_requested and not task.stop_reason:
                    task.stop_reason = "user"
                self._persist(now)
                self.store.append_event({"event": "task_cancel_requested", "task": name}, self.cfg.event_log)
                msg = "已请求停止运行中的任务进程组" if task.is_running else "任务已取消（尚未运行）"
                return ok({"task": task.to_dict(), "message": msg})
            if command == "remove":
                name = str(params.get("name", ""))
                try:
                    task = self.queue.remove(name)
                except TaskError as exc:
                    return fail(str(exc))
                self._persist(now)
                return ok({"task": task.to_dict()})
            if command == "requeue":
                name = str(params.get("name", ""))
                try:
                    task = self.queue.requeue(name)
                except TaskError as exc:
                    return fail(str(exc))
                self._persist(now)
                return ok({"task": task.to_dict()})
            if command == "reload":
                # 同步重载即可；**不要**再置 _reload_requested，否则下一个 tick 会再重载一次
                # （曾因此重复启动长连接，而旧连接无法可靠关闭 → 一直按旧白名单拒消息）。
                self._reload_config()
                result = self.queue.import_queue_file()
                self._persist(now)
                return ok({"import": result.summary(), "config": "已重载"})
            if command == "stop_daemon":
                self.logger.info("收到 CLI 停止请求")
                self.request_stop()
                return ok({"message": "daemon 正在退出（运行中的训练任务不会被杀死）"})
            return fail(f"未知命令: {command}")

    def status_payload(self) -> dict[str, Any]:
        """给 CLI status 用的状态（不含任何密钥/URL）。"""
        cfg = self.cfg
        counts = self.queue.counts()
        return {
            "daemon": {
                "pid": os.getpid(),
                "version": __version__,
                "started_at": self.started_at,
                "uptime_seconds": time.time() - self.started_at,
                "dry_run": self.dry_run,
                "no_launch": self.no_launch,
            },
            "mode": self.mode,
            "mode_since": self.mode_since,
            "server": cfg.server.resolved_name(),
            "user": cfg.server.resolved_user(),
            "counts": counts,
            "stats": self.stats,
            "watchdog": self.watchdog.to_dict(),
            "tasks": [t.to_dict() for t in self.queue.ordered_tasks()],
            "idle": {
                "required_checks": self.tracker.required_checks,
                "counts": {str(k): v for k, v in self.tracker.counts.items()},
                "since": {str(k): v for k, v in self.tracker.since.items()},
                "confirmed": self.tracker.confirmed(),
            },
            "waiting": [{"name": n, "reason": r} for n, r in self.last_plan.waiting],
            "plan": {
                "assignments": [
                    {"task": a.task.name, "gpu_ids": a.gpu_ids} for a in self.last_plan.assignments
                ],
            },
            "gpu_policy": {
                "max_idle_utilization": cfg.gpu.max_idle_utilization,
                "max_idle_memory_mb": cfg.gpu.max_idle_memory_mb,
                "require_no_compute_process": cfg.gpu.require_no_compute_process,
                "check_interval_seconds": cfg.gpu.check_interval_seconds,
                "idle_confirm_checks": cfg.gpu.idle_confirm_checks,
                "monitor_gpus": cfg.gpu.monitor_gpu_ids or "all",
                "scheduler_allowed_gpus": cfg.gpu.scheduler_allowed_gpus or "all",
                "reserved_gpus": cfg.gpu.reserved_gpu_ids,
            },
            "scheduler": {
                "strategy": cfg.scheduler.strategy,
                "max_concurrent_tasks": cfg.scheduler.max_concurrent_tasks,
                "launch_second_check": cfg.scheduler.launch_second_check,
                "grace_before_reuse_seconds": cfg.scheduler.grace_before_reuse_seconds,
                "allow_cpu_tasks": cfg.scheduler.allow_cpu_tasks,
            },
            "notify": {
                "on_gpu_idle": cfg.notify.on_gpu_idle,
                "idle_cooldown_seconds": cfg.notify.idle_cooldown_seconds,
                "on_task_started": cfg.notify.on_task_started,
                "on_task_finished": cfg.notify.on_task_finished,
                "on_task_failed": cfg.notify.on_task_failed,
            },
            "feishu": {
                "enabled": cfg.feishu.enabled,
                "card": cfg.feishu.card,
                "receive_id_type": cfg.feishu.receive_id_type,
                "has_target": bool(cfg.feishu.resolved_receive_id()),
            },
            "interactive": {
                "enabled": cfg.feishu.interactive_enabled,
                "running": self.bot is not None,
                "allowed_users": len(cfg.feishu.allowed_users),
                "rate_limit_per_minute": cfg.feishu.rate_limit_per_minute,
                "eta_enabled": cfg.feishu.eta_enabled,
            },
            "paths": {
                "config": str(cfg.source_path) if cfg.source_path else "",
                "state": str(cfg.state_file),
                "log": str(cfg.log_file),
                "task_log_dir": str(cfg.task_log_dir),
                "queue_file": str(cfg.queue_file),
            },
        }

    # ================================================================== #
    # 退出
    # ================================================================== #
    def _shutdown(self) -> None:
        with self.lock:
            self.logger.info("正在退出 GPU Manager …")
            running = self.queue.running_tasks()
            if self.server is not None:
                self.server.stop()
                self.server = None
            self._stop_bot()
            for handle in list(self.handles.values()):
                handle.flush_close()
            self.handles.clear()
            self.state["daemon_pid"] = None
            self.state["stopped_at"] = time.time()
            self.state["mode"] = self._compute_mode()
            self.queue.dump(self.state)
            self.state["idle"] = self.tracker.to_dict()
            self.state["stats"] = self.stats
            self.store.write(self.state)
            self.store.clear_pid(self.cfg.pid_file)
            self.store.append_event(
                {"event": "daemon_stop", "pid": os.getpid(), "running_tasks": [t.name for t in running]},
                self.cfg.event_log,
            )
            if running:
                names = ", ".join(t.name for t in running[:5])
                self.logger.warning(
                    "%d 个训练任务仍在运行，daemon 退出不会终止它们（下次启动会自动接管监控）: %s",
                    len(running), names,
                )
                with contextlib.suppress(Exception):
                    self._notify(
                        "🛑 GPU Manager 已停止（训练任务继续运行）",
                        f"**服务器**: {self.cfg.server.resolved_name()}\n\n"
                        f"仍有 {len(running)} 个任务在运行: {names}\n"
                        "这些任务不会被终止，但在 daemon 重启前它们的状态不会被更新。",
                        level="warning",
                    )
            self.notifier.close()
            self.logger.info("已退出")


__all__ = ["Daemon", "MODE_IDLE", "MODE_QUEUE", "MODE_RUNNING"]
