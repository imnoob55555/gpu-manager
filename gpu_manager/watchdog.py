"""低频率看门狗：有训练任务时定期确认「还在推进」，持续无进展则告警并留诊断。

设计要点
--------
* **只在有任务时工作**：没有运行中任务时看门狗整体关闭（不做任何检查与统计），
  任务全部结束会清空每任务状态，下次有新任务时重新开始。
* **三类信号**（任一有动静就算「有进展」）:

    1. 进度记录 —— ``task.progress_updated_at``（eta 采样写入新样本的时间）
    2. 日志时间 —— 捕获日志 / ``task.eta_logs`` / ``task.progress_source`` 的 mtime
    3. 进程状态 —— ``/proc/<pid>`` 的存活、身份（starttime 防 PID 复用）与状态字符

* **两级告警**:

    · 停滞：连续 ``stall_threshold_seconds`` 无任何信号更新（任务启动后的
      ``startup_grace_seconds`` 宽限期内不判停滞）
    · 进程异常：消失 / 僵尸 Z / 停止 T / 不可中断 D / PID 复用 → 连续
      ``anomaly_confirm_checks`` 轮确认后**立即**告警，不等停滞阈值

* **告警去重**：同一问题首次立即告警，之后每 ``repeat_alert_interval_seconds``
  重复提醒；问题消失时按 ``notify_on_recovery`` 决定是否发恢复通知
  （恢复事件始终记入日志/事件流，只是不推送）。
* **诊断快照**：告警时把进程详情 / 日志尾部 / 进度与 ETA / 相关 GPU 状态写在
  ``<paths.diagnostics_dir>/<任务>/<时间戳>/``，保留份数与总量上限由
  ``watchdog.diagnostics_*`` 控制。

安全边界（与其他模块一致）: 全程只读 ``/proc`` 与文件，**绝不对任何进程发信号**，
不清理显存、不 reset GPU。看门狗只观察、只告警。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import retention
from .config import Config
from .utils import (
    atomic_write_text,
    ensure_dir,
    fmt_duration,
    fmt_ts,
    is_own_process,
    proc_uid,
    process_alive,
    process_start_time,
    process_state,
    truncate,
)

#: 进程异常类型 → (可读标签, 告警级别)
ANOMALY_LABELS: dict[str, tuple[str, str]] = {
    "gone": ("进程已消失", "error"),
    "zombie": ("僵尸进程（已结束但未回收退出码）", "error"),
    "identity": ("进程身份异常（疑似 PID 复用或非当前用户）", "error"),
    "stopped": ("进程已停止（T 状态）", "warning"),
    "uninterruptible": ("进程处于不可中断 I/O 等待（D 状态）", "warning"),
}

#: /proc/<pid>/stat 状态字符的可读标签
STATE_LABELS: dict[str, str] = {
    "R": "运行中",
    "S": "可中断睡眠",
    "D": "不可中断 I/O 等待",
    "T": "已停止",
    "t": "跟踪停止",
    "Z": "僵尸",
    "X": "已终止",
    "I": "空闲内核线程",
}

#: 活动来源标签
ACTIVITY_STARTUP = "任务启动"
ACTIVITY_PROGRESS = "进度记录"


def _safe_name(name: str) -> str:
    """把任务名变成可安全用作目录名的形式（尽量保留可读性）。"""
    out = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name).strip("._")
    return (out or "task")[:80]


def read_proc_cpu_seconds(pid: int) -> float | None:
    """读取进程累计 CPU 时间（utime+stime，秒）；读不到返回 None。

    CPU 时间只用于诊断展示（判断「活着但在空转」还是「完全没被调度」），
    不参与「是否有进展」的判定 —— 死锁自旋的进程 CPU 时间照样增长。
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except OSError:
        return None
    idx = data.rfind(")")
    if idx < 0:
        return None
    fields = data[idx + 2:].split()
    if len(fields) < 13:
        return None
    try:
        ticks = int(fields[11]) + int(fields[12])
    except ValueError:
        return None
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError):
        hz = 100
    return ticks / hz


@dataclass
class ProcInfo:
    """一次只读进程探测的结果。"""

    pid: int = 0
    state: str = ""
    alive: bool = False
    own: bool = False
    #: 累计 CPU 时间；None = 读不到（与「确实是 0」区分开）
    cpu_seconds: float | None = None
    anomaly: str = ""
    reason: str = ""

    @property
    def state_text(self) -> str:
        if not self.state:
            return "未知"
        return f"{self.state}（{STATE_LABELS.get(self.state, '未知状态')}）"

    def describe(self) -> str:
        if not self.pid:
            return "无 pid 记录"
        cpu = fmt_duration(self.cpu_seconds) if self.cpu_seconds is not None else "-"
        return (
            f"pid={self.pid} 状态={self.state_text} 存活={'是' if self.alive else '否'} "
            f"owner={'本人' if self.own else '他人/未知'} CPU 时间={cpu}"
        )


@dataclass
class WatchEvent:
    """看门狗产生的一次事件（供 daemon 记日志/事件流，测试断言）。"""

    kind: str  # stall | anomaly | recovered
    task: str
    level: str
    title: str
    detail: str
    alert_index: int = 0
    snapshot_dir: str = ""
    notified: bool = True

    @property
    def event_name(self) -> str:
        if self.kind == "recovered":
            return "watchdog_recovered"
        return "watchdog_anomaly" if self.kind.startswith("anomaly") else "watchdog_stall"


@dataclass
class TaskWatch:
    """单个任务的监视状态（内存态；元数据派生自真实时间戳，重启不产生假阴性）。"""

    name: str
    first_seen: float = 0.0
    last_check: float = 0.0
    last_activity: float = 0.0
    activity_source: str = ACTIVITY_STARTUP
    idle_seconds: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    proc: ProcInfo = field(default_factory=ProcInfo)
    anomaly_hits: dict[str, int] = field(default_factory=dict)
    anomaly_kind: str = ""
    alert_kind: str = ""
    alert_reason: str = ""
    last_issue: str = ""
    alert_count: int = 0
    last_alert_at: float = 0.0
    last_snapshot: str = ""
    recovered_at: float = 0.0

    @property
    def in_alert(self) -> bool:
        return bool(self.alert_kind)


class Watchdog:
    """看门狗主体。所有时间参数都来自调用方传入的 ``now``，便于测试。"""

    def __init__(
        self,
        cfg: Config,
        logger: Any,
        notifier: Any | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.notifier = notifier
        self.dry_run = dry_run
        self.watches: dict[str, TaskWatch] = {}
        self.active = False
        self.last_check_at = 0.0
        self.check_count = 0
        self.last_alert_at = 0.0
        self.alert_total = 0

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #
    @property
    def wcfg(self) -> Any:
        return self.cfg.watchdog

    def reconfigure(self, cfg: Config) -> None:
        """热重载配置：只换阈值，保留每任务的告警历史（避免重复刷屏）。"""
        self.cfg = cfg
        self.logger.info("看门狗配置已更新: %s", self.wcfg.summary())

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def update(
        self,
        tasks: Iterable[Any],
        *,
        now: float | None = None,
        snapshot: Any | None = None,
        eta_provider: Callable[[Any], str] | None = None,
    ) -> list[WatchEvent]:
        """推进一轮检查（内部按 ``check_interval_seconds`` 节流）。

        返回本轮产生的事件；无任务、未到检查点或功能关闭时返回空列表。
        """
        now = now if now is not None else time.time()
        task_list = [t for t in tasks if getattr(t, "status", "") == "running"]

        if not self.wcfg.enabled:
            self._deactivate(now, reason="配置已禁用")
            return []
        if not task_list:
            self._deactivate(now, reason="无运行中任务")
            return []

        if not self.active:
            self.active = True
            self.logger.info(
                "看门狗启用: 开始监视 %d 个任务（%s）", len(task_list), self.wcfg.summary()
            )

        interval = max(10.0, float(self.wcfg.check_interval_seconds))
        if self.last_check_at and (now - self.last_check_at) < interval:
            return []

        self.last_check_at = now
        self.check_count += 1

        live = {t.name for t in task_list}
        for name in list(self.watches):
            if name not in live:
                self.watches.pop(name, None)

        events: list[WatchEvent] = []
        for task in task_list:
            watch = self.watches.get(task.name)
            if watch is None:
                watch = TaskWatch(name=task.name, first_seen=now)
                self.watches[task.name] = watch
            events.extend(
                self._check_task(
                    task, watch, now=now, snapshot=snapshot, eta_provider=eta_provider
                )
            )
        return events

    def _deactivate(self, now: float, *, reason: str) -> None:
        """关闭看门狗（无任务 / 配置禁用）：清空每任务状态，有新任务时重新开始。"""
        if self.active:
            self.logger.info(
                "看门狗关闭（%s）: 任务结束即停止检查，有新任务时自动恢复", reason
            )
        self.active = False
        self.last_check_at = 0.0
        if self.watches:
            self.watches.clear()

    # ------------------------------------------------------------------ #
    # 单任务检查
    # ------------------------------------------------------------------ #
    def _check_task(
        self,
        task: Any,
        watch: TaskWatch,
        *,
        now: float,
        snapshot: Any | None,
        eta_provider: Callable[[Any], str] | None,
    ) -> list[WatchEvent]:
        watch.last_check = now
        watch.proc = self._proc_info(task)
        signals = self._signals(task, now=now)
        watch.signals = signals
        activity, source = self._activity(signals, task, now)
        if activity > watch.last_activity:
            watch.last_activity = activity
            watch.activity_source = source
        watch.idle_seconds = max(0.0, now - watch.last_activity)
        return self._judge(
            task, watch, now=now, snapshot=snapshot, eta_provider=eta_provider
        )

    def _judge(
        self,
        task: Any,
        watch: TaskWatch,
        *,
        now: float,
        snapshot: Any | None,
        eta_provider: Callable[[Any], str] | None,
    ) -> list[WatchEvent]:
        issue = self._detect_issue(task, watch, now)
        if issue is not None:
            kind, reason, level = issue
            if watch.alert_kind != kind:
                watch.alert_kind = kind
                watch.alert_count = 0
                watch.last_alert_at = 0.0
            watch.alert_reason = reason
            repeat = float(self.wcfg.repeat_alert_interval_seconds or 0)
            due = watch.alert_count == 0 or (repeat > 0 and (now - watch.last_alert_at) >= repeat)
            if not due:
                return []
            watch.alert_count += 1
            watch.last_alert_at = now
            self.last_alert_at = now
            self.alert_total += 1
            return [
                self._emit_alert(
                    task,
                    watch,
                    kind=kind,
                    reason=reason,
                    level=level,
                    now=now,
                    snapshot=snapshot,
                    eta_provider=eta_provider,
                )
            ]
        if watch.alert_kind:
            return [self._emit_recovery(task, watch, now=now)]
        return []

    def _detect_issue(self, task: Any, watch: TaskWatch, now: float) -> tuple[str, str, str] | None:
        """返回 ``(kind, reason, level)`` 或 None（无问题）。"""
        proc = watch.proc
        if self.wcfg.instant_process_anomaly and proc.anomaly:
            hits = watch.anomaly_hits.get(proc.anomaly, 0) + 1
            watch.anomaly_hits = {proc.anomaly: hits}
            need = max(1, int(self.wcfg.anomaly_confirm_checks))
            if hits >= need:
                watch.anomaly_kind = proc.anomaly
                label, level = ANOMALY_LABELS.get(proc.anomaly, (proc.anomaly, "warning"))
                reason = f"{label}：{proc.reason}" if proc.reason else label
                return f"anomaly:{proc.anomaly}", reason, level
        else:
            if watch.anomaly_hits:
                watch.anomaly_hits.clear()
            watch.anomaly_kind = ""

        if self._grace_left(task, now) > 0:
            return None
        threshold = float(self.wcfg.stall_threshold_seconds)
        if watch.idle_seconds >= threshold:
            reason = (
                f"已 {fmt_duration(watch.idle_seconds)} 无任何进展"
                f"（最后活动: {watch.activity_source}，{fmt_ts(watch.last_activity)}）"
            )
            return "stall", reason, "warning"
        return None

    def _grace_left(self, task: Any, now: float) -> float:
        grace = float(self.wcfg.startup_grace_seconds or 0)
        if grace <= 0:
            return 0.0
        started = getattr(task, "started_at", None) or getattr(task, "created_at", None) or now
        return max(0.0, (started + grace) - now)

    # ------------------------------------------------------------------ #
    # 信号采集（全部只读）
    # ------------------------------------------------------------------ #
    def _signals(self, task: Any, *, now: float) -> dict[str, float]:
        """返回 ``{来源标签: 最后活动时间戳}``；时间戳一律夹到 ``now`` 以内。"""
        out: dict[str, float] = {}
        started = getattr(task, "started_at", None) or getattr(task, "created_at", None)
        if started:
            out[ACTIVITY_STARTUP] = min(float(started), now)
        progress_at = float(getattr(task, "progress_updated_at", 0.0) or 0.0)
        if progress_at:
            out[ACTIVITY_PROGRESS] = min(progress_at, now)
        for label, path in self._log_paths(task):
            mtime = _stat_mtime(path)
            if mtime:
                out[label] = min(mtime, now)
        return out

    def _activity(self, signals: dict[str, float], task: Any, now: float) -> tuple[float, str]:
        if not signals:
            return now, "无可用信号"
        source = max(signals, key=lambda key: signals[key])
        return signals[source], source

    def _log_paths(self, task: Any) -> list[tuple[str, Path]]:
        """需要观察 mtime 的日志/进度文件（去重，标签可读）。"""
        out: list[tuple[str, Path]] = []
        seen: set[str] = set()
        cwd = Path(getattr(task, "cwd", "") or ".")

        def add(label: str, raw: str) -> None:
            raw = str(raw or "").strip()
            if not raw:
                return
            path = Path(os.path.expanduser(raw))
            if not path.is_absolute():
                path = cwd / path
            key = str(path)
            if key in seen:
                return
            seen.add(key)
            out.append((truncate(label, 60), path))

        add("捕获日志", getattr(task, "log_file", ""))
        for raw in getattr(task, "eta_logs", []) or []:
            name = Path(str(raw)).name or str(raw)
            add(f"日志 {name}", raw)
        add("日志 进度来源", getattr(task, "progress_source", ""))
        return out

    def _proc_info(self, task: Any) -> ProcInfo:
        pid = int(getattr(task, "pid", 0) or 0)
        if not pid:
            return ProcInfo(pid=0, anomaly="gone", reason="任务记录里没有 pid")
        state = process_state(pid) or ""
        alive = process_alive(pid)  # 僵尸不算存活
        own = is_own_process(pid)
        cpu = read_proc_cpu_seconds(pid)
        info = ProcInfo(pid=pid, state=state, alive=alive, own=own, cpu_seconds=cpu)

        if state == "Z":
            info.anomaly = "zombie"
            info.reason = f"pid {pid} 状态 Z（已结束，等待父进程回收退出码）"
        elif not alive:
            info.anomaly = "gone"
            info.reason = f"pid {pid} 已不存在"
        elif not own:
            info.anomaly = "identity"
            info.reason = f"pid {pid} 的 owner 是 uid{proc_uid(pid)}，不是当前用户"
        else:
            recorded = getattr(task, "proc_start_time", None)
            if recorded is not None:
                actual = process_start_time(pid)
                if actual is None or abs(actual - float(recorded)) > 0.5:
                    info.anomaly = "identity"
                    info.reason = (
                        f"pid {pid} 的 starttime 与记录不符（疑似 PID 复用），"
                        "拒绝采信该进程的存活状态"
                    )
        if not info.anomaly:
            if state == "T":
                info.anomaly = "stopped"
                info.reason = f"pid {pid} 处于 T 状态（被 SIGSTOP/调停），不会继续训练"
            elif state == "D":
                info.anomaly = "uninterruptible"
                info.reason = f"pid {pid} 处于 D 状态（不可中断 I/O 等待），可能卡在磁盘/网络"
        return info

    # ------------------------------------------------------------------ #
    # 告警与恢复
    # ------------------------------------------------------------------ #
    def _emit_alert(
        self,
        task: Any,
        watch: TaskWatch,
        *,
        kind: str,
        reason: str,
        level: str,
        now: float,
        snapshot: Any | None,
        eta_provider: Callable[[Any], str] | None,
    ) -> WatchEvent:
        snapshot_dir = ""
        if self.wcfg.diagnostics_enabled and not self.dry_run:
            try:
                snapshot_dir = self.write_snapshot(
                    task,
                    watch=watch,
                    reason=reason,
                    kind=kind,
                    now=now,
                    snapshot=snapshot,
                    eta_text=(eta_provider(task) if eta_provider else ""),
                )
                watch.last_snapshot = snapshot_dir
            except Exception as exc:  # 诊断写入失败绝不能影响告警本身
                self.logger.warning("诊断快照写入失败（不影响告警）: %s", exc)

        if kind == "stall":
            title = f"⏸️ 任务疑似停滞: {task.name}"
        else:
            label = ANOMALY_LABELS.get(kind.split(":", 1)[-1], (kind, level))[0]
            title = f"❗ 进程异常（{label}）: {task.name}"
        if watch.alert_count > 1:
            title += f"（第 {watch.alert_count} 次提醒）"

        lines = [
            f"**任务**: {task.name}",
            f"**状态**: {getattr(task, 'status', '')}  pid={watch.proc.pid or '-'}",
            f"**问题**: {reason}",
            f"**运行时长**: {fmt_duration(_runtime(task, now))}",
        ]
        signals_text = self._signals_text(watch, now)
        if signals_text:
            lines.append(f"**信号**: {signals_text}")
        lines.append(f"**进程**: {watch.proc.describe()}")
        gpus = getattr(task, "gpu_ids", []) or []
        if gpus:
            lines.append("**GPU**: " + ", ".join(str(g) for g in gpus))
        log_text = self._primary_log_text(task)
        if log_text:
            lines.append(f"**日志**: {log_text}")
        if snapshot_dir:
            lines.append(f"**诊断**: {snapshot_dir}")
        lines.append("")
        lines.append(f"查看: `gpu-manager watchdog diagnostics {task.name}`")
        body = "\n".join(lines)

        notified = self._send(title, body, level)
        self.logger.warning("看门狗告警[%s] %s | %s", kind, task.name, truncate(reason, 300))
        return WatchEvent(
            kind=kind,
            task=task.name,
            level=level,
            title=title,
            detail=body,
            alert_index=watch.alert_count,
            snapshot_dir=snapshot_dir,
            notified=notified,
        )

    def _emit_recovery(self, task: Any, watch: TaskWatch, *, now: float) -> WatchEvent:
        kind, reason = watch.alert_kind, watch.alert_reason
        watch.alert_kind = ""
        watch.alert_reason = ""
        watch.last_issue = reason or kind
        watch.alert_count = 0
        watch.recovered_at = now
        watch.anomaly_hits.clear()
        title = f"✅ 任务已恢复进展: {task.name}"
        body = "\n".join(
            [
                f"**任务**: {task.name}",
                f"**原问题**: {reason or kind}",
                f"**当前**: {watch.proc.describe()}",
                f"**最后活动**: {watch.activity_source} · {fmt_ts(watch.last_activity)}",
            ]
        )
        notified = False
        if self.wcfg.notify_on_recovery:
            notified = self._send(title, body, "success")
        self.logger.info(
            "看门狗恢复: %s（原问题 %s，通知=%s）", task.name, kind, "已发" if notified else "未发"
        )
        return WatchEvent(
            kind="recovered",
            task=task.name,
            level="success",
            title=title,
            detail=body,
            notified=notified,
        )

    def _send(self, title: str, body: str, level: str) -> bool:
        if self.dry_run:
            self.logger.info("[dry-run] 看门狗通知: %s", title)
            return False
        if self.notifier is None:
            return False
        try:
            ok_sent, detail = self.notifier.send(title, body, level)
        except Exception as exc:  # 通知层问题绝不影响看门狗
            self.logger.exception("看门狗通知发送异常: %s", exc)
            return False
        if not ok_sent:
            self.logger.warning("看门狗通知发送失败: %s", detail)
        return bool(ok_sent)

    # ------------------------------------------------------------------ #
    # 诊断快照
    # ------------------------------------------------------------------ #
    def write_snapshot(
        self,
        task: Any,
        *,
        watch: TaskWatch,
        reason: str,
        kind: str,
        now: float,
        snapshot: Any | None = None,
        eta_text: str = "",
    ) -> str:
        """写入一份诊断快照，返回目录路径。全部只读采集，绝不触碰进程。"""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        root = Path(self.cfg.diagnostics_dir)
        dest = root / _safe_name(str(getattr(task, "name", "task"))) / stamp
        suffix = 1
        while dest.exists():
            suffix += 1
            dest = dest.with_name(f"{stamp}-{suffix}")
        ensure_dir(dest)

        summary = self._summary_text(
            task, watch=watch, reason=reason, kind=kind, now=now, eta_text=eta_text
        )
        atomic_write_text(dest / "summary.txt", summary)
        atomic_write_text(dest / "process.txt", self._process_text(watch.proc))
        atomic_write_text(dest / "log_tail.txt", self._log_tail_text(task))
        atomic_write_text(
            dest / "progress.json",
            json.dumps(
                {
                    "progress": getattr(task, "progress", {}) or {},
                    "progress_updated_at": getattr(task, "progress_updated_at", 0.0),
                    "progress_source": getattr(task, "progress_source", ""),
                    "stage": getattr(task, "stage", ""),
                    "total_epochs": getattr(task, "total_epochs", 0),
                    "eta_text": eta_text,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        if self.wcfg.include_gpu_snapshot:
            gpu_text = self._gpu_text(task, snapshot)
            if gpu_text:
                atomic_write_text(dest / "gpu.txt", gpu_text)
        self._prune()
        self.logger.info("看门狗诊断快照已写入: %s", dest)
        return str(dest)

    def _summary_text(
        self,
        task: Any,
        *,
        watch: TaskWatch,
        reason: str,
        kind: str,
        now: float,
        eta_text: str,
    ) -> str:
        verdict = "进程异常" if kind.startswith("anomaly") else "无进展（疑似停滞）"
        lines = [
            "GPU Manager 看门狗诊断快照",
            "=" * 60,
            f"任务      : {getattr(task, 'name', '')}",
            f"任务状态  : {getattr(task, 'status', '')}",
            f"判定      : {verdict}",
            f"原因      : {reason}",
            f"采集时间  : {fmt_ts(now)}",
            "",
            "--- 活动信号（任一有动静即算有进展）---",
        ]
        for label, ts in sorted(watch.signals.items(), key=lambda kv: -kv[1]):
            age = max(0.0, now - ts)
            lines.append(f"  {label:<28} {fmt_ts(ts)}  （{fmt_duration(age)} 前）")
        if not watch.signals:
            lines.append("  （无任何可用信号）")
        lines += [
            f"  最后活动  : {watch.activity_source} · {fmt_ts(watch.last_activity)}"
            f"（{fmt_duration(watch.idle_seconds)} 前）",
            "",
            "--- 进程 ---",
            f"  {watch.proc.describe()}",
            f"  异常      : {watch.proc.anomaly or '无'}",
            "",
            "--- 任务 ---",
            f"  进程组    : pgid={getattr(task, 'pgid', None)}",
            f"  启动时间  : {fmt_ts(getattr(task, 'started_at', None))}"
            f"（已运行 {fmt_duration(_runtime(task, now))}）",
            f"  GPU       : {getattr(task, 'gpu_ids', [])}",
            f"  工作目录  : {getattr(task, 'cwd', '')}",
            f"  命令      : {truncate(str(getattr(task, 'command', '')), 500)}",
            f"  本次重试  : {getattr(task, 'retries', 0)}",
            "",
            "--- 日志来源 ---",
        ]
        for label, path in self._log_paths(task):
            size = _file_size_text(path)
            mtime = _stat_mtime(path)
            lines.append(
                f"  {label:<28} {path}  最后修改={fmt_ts(mtime) if mtime else '-'}  大小={size}"
            )
        lines += [
            "",
            "--- ETA ---",
            f"  {eta_text or '（无估算：需要进度样本或历史耗时）'}",
            "",
            "--- 看门狗配置 ---",
            f"  {self.wcfg.summary()}",
            f"  启动宽限期: {self.wcfg.startup_grace_seconds}s"
            f"  异常确认轮数: {self.wcfg.anomaly_confirm_checks}",
            "",
            "安全说明: 本快照全部为只读采集（/proc、日志与 nvidia-smi 快照），",
            "         未对任何进程发送信号，未清理显存，未 reset GPU。",
        ]
        return "\n".join(lines) + "\n"

    def _process_text(self, proc: ProcInfo) -> str:
        if not proc.pid:
            return "无 pid 记录\n"
        chunks: list[str] = [f"# {proc.describe()}", ""]
        for name in ("status", "wchan", "io", "limits"):
            text = _read_proc_file(proc.pid, name)
            if text:
                chunks.append(f"# /proc/{proc.pid}/{name}")
                chunks.append(text.rstrip())
                chunks.append("")
        threads = _read_proc_file(proc.pid, "task")
        if threads:
            chunks.append(f"# /proc/{proc.pid}/task（线程数 {len(threads.splitlines())}）")
            chunks.append(truncate(threads.rstrip(), 4000))
            chunks.append("")
        stat = _read_proc_file(proc.pid, "stat")
        if stat:
            chunks.append(f"# /proc/{proc.pid}/stat")
            chunks.append(stat.rstrip())
            chunks.append("")
        for name, label in (("cmdline", "命令行"), ("cwd", "工作目录"), ("exe", "可执行文件")):
            path = f"/proc/{proc.pid}/{name}"
            try:
                if name == "cmdline":
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    value = " ".join(p.decode("utf-8", "replace") for p in raw.split(b"\0") if p)
                else:
                    value = os.readlink(path)
            except OSError as exc:
                value = f"<读取失败: {exc}>"
            chunks.append(f"# {label} ({path})")
            chunks.append(value or "<空>")
            chunks.append("")
        return "\n".join(chunks) + "\n"

    def _log_tail_text(self, task: Any) -> str:
        limit = max(1, int(self.wcfg.log_tail_lines))
        chunks: list[str] = []
        for label, path in self._log_paths(task):
            chunks.append(f"===== {label} · {path} =====")
            lines = _tail_lines(path, limit)
            if lines:
                chunks.extend(lines)
            else:
                chunks.append("<无法读取或无内容>")
            chunks.append("")
        return "\n".join(chunks) + "\n"

    def _gpu_text(self, task: Any, snapshot: Any | None) -> str:
        if snapshot is None or not getattr(snapshot, "ok", False):
            return ""
        wanted = {int(g) for g in (getattr(task, "gpu_ids", []) or [])}
        chunks: list[str] = []
        for gpu in getattr(snapshot, "gpus", []) or []:
            if wanted and int(getattr(gpu, "index", -1)) not in wanted:
                continue
            chunks.append(
                f"GPU {gpu.index}: {gpu.name} 利用率={gpu.utilization}% "
                f"显存={gpu.memory_used_mb}/{gpu.memory_total_mb}MB"
            )
            for proc in list(getattr(gpu, "own_processes", []) or []):
                chunks.append(
                    f"  [本人] pid={proc.pid} {proc.command} 显存={proc.memory_mb}MB"
                )
            for proc in list(getattr(gpu, "foreign_processes", []) or []):
                chunks.append(
                    f"  [他人 uid{proc.uid}] pid={proc.pid} {proc.command} 显存={proc.memory_mb}MB"
                )
        if not chunks:
            chunks.append("（快照中没有该任务的 GPU 信息）")
        return "\n".join(chunks) + "\n"

    def _prune(self) -> None:
        try:
            report = retention.prune_watchdog_diagnostics(self.cfg)
        except Exception as exc:  # 清理失败不影响主流程
            self.logger.warning("看门狗诊断清理失败: %s", exc)
            return
        if report.changed:
            self.logger.info(
                "看门狗诊断清理: 删除 %d 份快照 / %s，目录现占 %s（%s）",
                len(report.removed_files),
                retention.human_size(report.removed_bytes),
                retention.human_size(report.remaining_bytes),
                self.cfg.diagnostics_dir,
            )

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #
    def _signals_text(self, watch: TaskWatch, now: float) -> str:
        if not watch.signals:
            return ""
        ordered = sorted(watch.signals.items(), key=lambda kv: -kv[1])[:4]
        return "；".join(f"{label} {fmt_duration(max(0.0, now - ts))}前" for label, ts in ordered)

    def _primary_log_text(self, task: Any) -> str:
        paths = self._log_paths(task)
        if not paths:
            return ""
        label, path = paths[0]
        mtime = _stat_mtime(path)
        size = _file_size_text(path)
        return f"{path}（{label}，最后修改 {fmt_ts(mtime) if mtime else '-'}，大小 {size}）"

    def to_dict(self) -> dict[str, Any]:
        """写入 state.json 的看门狗段（供 ``watchdog status`` / 飞书查询展示）。"""
        w = self.wcfg
        return {
            "enabled": bool(w.enabled),
            "active": bool(self.active and w.enabled),
            "summary": w.summary(),
            "check_interval_seconds": w.check_interval_seconds,
            "stall_threshold_seconds": w.stall_threshold_seconds,
            "startup_grace_seconds": w.startup_grace_seconds,
            "repeat_alert_interval_seconds": w.repeat_alert_interval_seconds,
            "anomaly_confirm_checks": w.anomaly_confirm_checks,
            "diagnostics_enabled": bool(w.diagnostics_enabled),
            "diagnostics_dir": str(self.cfg.diagnostics_dir),
            "last_check_at": self.last_check_at,
            "check_count": self.check_count,
            "alert_total": self.alert_total,
            "last_alert_at": self.last_alert_at,
            "watched": len(self.watches),
            "tasks": {
                name: {
                    "pid": watch.proc.pid,
                    "proc_state": watch.proc.state,
                    "proc_state_text": watch.proc.state_text if watch.proc.pid else "",
                    "proc_anomaly": watch.proc.anomaly,
                    "last_activity": watch.last_activity,
                    "last_activity_source": watch.activity_source,
                    "idle_seconds": round(watch.idle_seconds, 1),
                    "cpu_seconds": (
                        round(watch.proc.cpu_seconds, 1)
                        if watch.proc.cpu_seconds is not None
                        else None
                    ),
                    "alert_kind": watch.alert_kind,
                    "alert_reason": watch.alert_reason,
                    "last_issue": watch.last_issue,
                    "alert_count": watch.alert_count,
                    "last_alert_at": watch.last_alert_at,
                    "last_snapshot": watch.last_snapshot,
                    "signals": {k: round(v, 1) for k, v in watch.signals.items()},
                }
                for name, watch in sorted(self.watches.items())
            },
        }


# --------------------------------------------------------------------------- #
# 小工具（只读文件访问，失败一律返回空而不是抛错）
# --------------------------------------------------------------------------- #
def _stat_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _file_size_text(path: Path) -> str:
    try:
        size = float(path.stat().st_size)
    except OSError:
        return "-"
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f}{unit}"
    return f"{int(size)}B"


def _read_proc_file(pid: int, name: str) -> str:
    try:
        with open(f"/proc/{pid}/{name}", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _tail_lines(path: Path, limit: int) -> list[str]:
    """只读文件尾部若干行（按块读，避免把大日志整个读进内存）。"""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    block = 64 * 1024
    data = b""
    try:
        with open(path, "rb") as fh:
            pos = size
            while pos > 0 and data.count(b"\n") <= limit:
                read = min(block, pos)
                pos -= read
                fh.seek(pos)
                data = fh.read(read) + data
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    return lines[-limit:]


def _runtime(task: Any, now: float) -> float:
    started = getattr(task, "started_at", None)
    if not started:
        return 0.0
    finished = getattr(task, "finished_at", None)
    return max(0.0, (finished or now) - started)


def format_status(data: dict[str, Any], *, now: float | None = None) -> list[str]:
    """把 ``Watchdog.to_dict()`` 的结果渲染成给人看的文本行（CLI / 飞书共用）。"""
    now = now if now is not None else time.time()
    lines: list[str] = []
    if not data:
        return ["看门狗: 无状态信息（daemon 未运行或版本较旧）"]
    state = "运行中" if data.get("active") else ("已启用（当前无任务，自动关闭）" if data.get("enabled") else "已禁用")
    lines.append(f"看门狗: {state}")
    lines.append(f"策略  : {data.get('summary', '-')}")
    if data.get("enabled"):
        lines.append(
            f"启动宽限: {data.get('startup_grace_seconds', 0)}s    "
            f"异常确认: {data.get('anomaly_confirm_checks', 0)} 轮    "
            f"诊断目录: {data.get('diagnostics_dir', '-')}"
        )
        last = data.get("last_check_at") or 0
        if last:
            lines.append(
                f"上次检查: {fmt_ts(last)}（{fmt_duration(max(0.0, now - last))} 前）"
                f"    累计检查 {data.get('check_count', 0)} 轮 / 告警 {data.get('alert_total', 0)} 次"
            )
        else:
            lines.append("上次检查: 尚未检查")
    tasks = data.get("tasks") or {}
    if not tasks:
        lines.append("监视任务: 无（没有运行中的任务）")
        return lines
    lines.append("")
    lines.append(f"监视任务 ({len(tasks)}):")
    for name, info in tasks.items():
        idle = fmt_duration(float(info.get("idle_seconds", 0)))
        source = info.get("last_activity_source", "-") or "-"
        head = f"  {name}"
        if info.get("alert_kind"):
            head += f"  ⚠️ {info.get('alert_reason', '')}"
        lines.append(head)
        lines.append(
            f"    最后进展: {fmt_ts(info.get('last_activity'))}（{idle} 前 · {source}）"
            f"    pid={info.get('pid') or '-'}"
            + (f"  {info.get('proc_state_text')}" if info.get("proc_state_text") else "")
        )
        if info.get("alert_count"):
            lines.append(
                f"    告警: 第 {info.get('alert_count')} 次 · 最近 {fmt_ts(info.get('last_alert_at'))}"
            )
        if info.get("last_snapshot"):
            lines.append(f"    最近诊断: {info.get('last_snapshot')}")
    return lines


__all__ = [
    "ANOMALY_LABELS",
    "ProcInfo",
    "TaskWatch",
    "WatchEvent",
    "Watchdog",
    "format_status",
    "read_proc_cpu_seconds",
]
