"""daemon 侧逻辑测试（不启动真实进程、不开 socket）。

覆盖两类容易被忽略的无人值守行为:
  · 日志清理的**触发**逻辑：节流、可关闭、异常不影响主循环
  · 运行超时：daemon 重启后接管的任务同样要受 timeout_seconds 约束
"""

from __future__ import annotations

import importlib.util
import os
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_manager import daemon as daemon_mod
from gpu_manager.bot import AllowList, CommandRouter, RateLimiter
from gpu_manager.daemon import MODE_IDLE, MODE_QUEUE, MODE_RUNNING, Daemon
from gpu_manager.eta import (
    ProgressSample,
    latest_samples,
    parse_progress,
    record_sample,
)
from gpu_manager.gpu_monitor import GpuInfo, Snapshot
from gpu_manager.notifier import NullNotifier
from gpu_manager.retention import PruneReport
from gpu_manager.state_store import StateStore
from gpu_manager.task_queue import CANCELLED, COMPLETED, FAILED, RUNNING
from gpu_manager.utils import process_start_time


def sample_for(line: str, at: float = 0.0) -> ProgressSample:
    """解析一行日志并断言成功（测试里到处要用）。"""
    sample = parse_progress(line, now=at)
    assert sample is not None, f"未能解析出进度: {line!r}"
    return sample


def dummy_credential() -> str:
    """运行时生成的占位凭证（测试里不写任何看似真实密钥的字面量）。"""
    return f"unit-test-{secrets.token_hex(4)}"


def make_spec(name: str, **kw) -> dict:
    spec = {"name": name, "cwd": "/path/to/project", "command": "python main.py", "gpu_num": 1}
    spec.update(kw)
    return spec


# --------------------------------------------------------------------------- #
# 模式判定
# --------------------------------------------------------------------------- #
def test_compute_mode_semantics(cfg):
    d = Daemon(cfg)
    assert d._compute_mode() == MODE_IDLE

    task = d.queue.add(make_spec("t1"))
    assert d._compute_mode() == MODE_QUEUE, "有 pending 但无 running → QUEUE"

    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    assert d._compute_mode() == MODE_RUNNING, "RUNNING 优先级最高"

    d.queue.mark_finished(task, status=COMPLETED, exit_code=0)
    assert d._compute_mode() == MODE_IDLE, "队列清空后回到 IDLE"


# --------------------------------------------------------------------------- #
# 日志清理的触发
# --------------------------------------------------------------------------- #
def test_maybe_prune_is_throttled(cfg, monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(
        daemon_mod, "prune_all", lambda *a, **k: (calls.append(1), PruneReport())[1]
    )
    cfg.storage.prune_interval_seconds = 300
    d = Daemon(cfg)

    d._maybe_prune(1000.0)  # 首次（_last_prune=0）→ 执行
    d._maybe_prune(1100.0)  # 未到间隔 → 跳过
    d._maybe_prune(1250.0)  # 仍未到 → 跳过
    d._maybe_prune(1400.0)  # 超过 300s → 执行

    assert len(calls) == 2


def test_maybe_prune_disabled_when_zero(cfg, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        daemon_mod, "prune_all", lambda *a, **k: (calls.append(1), PruneReport())[1]
    )
    cfg.storage.prune_interval_seconds = 0
    d = Daemon(cfg)

    d._maybe_prune(10_000.0)
    d._maybe_prune(99_999.0)

    assert calls == [], "prune_interval_seconds=0 表示不在 tick 中自动清理"


def test_maybe_prune_passes_active_and_failed_tasks(cfg, monkeypatch):
    captured: dict = {}

    def fake_prune(_cfg, _logger, **kwargs):
        captured.update(kwargs)
        return PruneReport()

    monkeypatch.setattr(daemon_mod, "prune_all", fake_prune)
    cfg.storage.prune_interval_seconds = 1
    d = Daemon(cfg)

    running = d.queue.add(make_spec("run1"))
    d.queue.mark_running(
        running, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/logs/run1/a.log"
    )
    failed = d.queue.add(make_spec("bad1"))
    d.queue.mark_finished(failed, status=FAILED, exit_code=1, error="x")

    d._maybe_prune(10_000.0)

    assert captured["active_log_paths"] == ["/logs/run1/a.log"], "必须告知哪些日志正被使用"
    assert captured["failed_task_names"] == ["bad1"], "失败任务日志要优先保留"


def test_maybe_prune_never_breaks_the_tick(cfg, monkeypatch):
    """清理失败（例如磁盘只读）绝不能影响调度主循环。"""

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(daemon_mod, "prune_all", boom)
    cfg.storage.prune_interval_seconds = 1
    d = Daemon(cfg)

    d._maybe_prune(10_000.0)  # 不应抛出


# --------------------------------------------------------------------------- #
# 终态通知
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [COMPLETED, FAILED, CANCELLED])
def test_finish_task_emits_notification(cfg, status):
    """回归: 终态通知里的 level 映射曾把键写成大写字面量（"FAILED"），
    与 status 的小写值（"failed"）不匹配 → 每次任务结束都抛 KeyError，
    导致「任务完成/失败」通知**静默丢失**（tick 被 try 兜住，所以 daemon 不崩，
    问题非常隐蔽）。现在改用常量映射表 + .get() 默认值。
    """
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    notifier = d.notifier
    assert isinstance(notifier, NullNotifier), "测试配置里飞书关闭 → 降级为只记录，便于断言"

    d._finish_task(
        task,
        status=status,
        exit_code=0 if status == COMPLETED else 1,
        error="" if status == COMPLETED else "boom",
    )

    assert task.status == status
    assert len(notifier.sent) == 1, f"{status} 必须发出一条通知"
    level, title, _body = notifier.sent[0]
    assert title.startswith(("✅", "❌", "🛑"))
    assert level in ("success", "error", "warning")
    assert d.stats[status] == 1


def test_finish_task_respects_notify_switches(cfg):
    cfg.notify.on_task_failed = False
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    notifier = d.notifier
    assert isinstance(notifier, NullNotifier)

    d._finish_task(task, status=FAILED, exit_code=1, error="x")

    assert task.status == FAILED, "状态照常更新"
    assert notifier.sent == [], "开关关掉就不该发"


def test_finish_task_can_skip_notification(cfg):
    """重启接管等场景会传 notify=False（不做事后通知历史事件）。"""
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    notifier = d.notifier
    assert isinstance(notifier, NullNotifier)

    d._finish_task(task, status=COMPLETED, exit_code=0, notify=False)

    assert task.status == COMPLETED
    assert notifier.sent == []


# --------------------------------------------------------------------------- #
# 运行超时（timeout_seconds）
# --------------------------------------------------------------------------- #
def test_timeout_triggers_stop_for_live_task(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1", timeout_seconds=10))
    d.queue.mark_running(
        task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l"
    )
    task.started_at = time.time() - 100
    task.pid = os.getpid()  # 用一个确实存活的 pid（自己）
    d.handles.clear()

    d._reap_tasks(time.time())

    assert task.stop_requested is True
    assert task.stop_reason == "timeout"
    assert task.status == RUNNING, "只是请求停止，等进程真正退出才改状态"


def test_timeout_works_after_daemon_restart(cfg):
    """回归: daemon 重启后没有 RunHandle，timeout 也必须继续生效。

    早期实现在「无 handle 且进程存活」时直接 continue，把超时检查整个跳过了。
    """
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1", timeout_seconds=5))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.started_at = time.time() - 60
    task.pid = os.getpid()

    assert d.handles == {}, "前置条件: 模拟重启后接管（没有 handle）"

    d._reap_tasks(time.time())

    assert task.stop_requested is True, "重启后接管的任务同样要受 timeout 约束"
    assert task.stop_reason == "timeout"


def test_timeout_not_triggered_before_deadline(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1", timeout_seconds=3600))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.started_at = time.time() - 10
    task.pid = os.getpid()

    d._reap_tasks(time.time())

    assert task.stop_requested is False


def test_no_timeout_configured_means_no_stop(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))  # timeout_seconds 默认 0 = 不限
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.started_at = time.time() - 10_000
    task.pid = os.getpid()

    d._reap_tasks(time.time())

    assert task.stop_requested is False


def test_check_timeout_is_idempotent(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1", timeout_seconds=1))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.started_at = time.time() - 100

    assert d._check_timeout(task, time.time()) is True
    assert d._check_timeout(task, time.time() + 1) is False, "已请求停止就不再重复触发"


# --------------------------------------------------------------------------- #
# 进度采样（供 ETA）
# --------------------------------------------------------------------------- #
def write_task_log(cfg, task_name: str, content: str) -> str:
    """在合法位置（task_log_dir 内）写一份日志，返回路径。"""
    log_dir = cfg.task_log_dir / task_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    # pi-lens-ignore: python-path-traversal -- 路径位于 cfg.task_log_dir 内
    log_file.write_text(content, encoding="utf-8")
    return str(log_file)


def test_sample_progress_parses_running_task_log(cfg):
    cfg.feishu.interactive_enabled = True
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    log_file = write_task_log(cfg, task.safe_name, "Epoch 12/500 [00:30<20:00]\n")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file=log_file)

    d._sample_progress(1000.0)

    assert task.progress, "应写入进度样本"
    assert task.progress_updated_at == 1000.0
    assert d._last_progress_sample == 1000.0


def test_sample_progress_is_throttled(cfg):
    cfg.feishu.interactive_enabled = True
    cfg.feishu.eta_sample_interval_seconds = 60
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    log_file = write_task_log(cfg, task.safe_name, "Epoch 3/100\n")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file=log_file)

    d._sample_progress(1000.0)
    first = dict(task.progress)
    d._sample_progress(1030.0)  # 未到间隔

    assert task.progress == first
    assert d._last_progress_sample == 1000.0


def test_sample_progress_skipped_when_interactive_disabled(cfg):
    """关掉的功能应当零开销：不读日志、不写状态。"""
    d = Daemon(cfg)
    assert cfg.feishu.interactive_enabled is False
    task = d.queue.add(make_spec("t1"))
    log_file = write_task_log(cfg, task.safe_name, "Epoch 12/500\n")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file=log_file)

    d._sample_progress(1000.0)

    assert task.progress == {}
    assert d._last_progress_sample == 0.0


def test_sample_progress_disabled_when_eta_off(cfg):
    cfg.feishu.interactive_enabled = True
    cfg.feishu.eta_enabled = False
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    log_file = write_task_log(cfg, task.safe_name, "Epoch 12/500\n")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file=log_file)

    d._sample_progress(1000.0)

    assert task.progress == {}


def test_sample_progress_skips_non_running_and_missing_logs(cfg):
    cfg.feishu.interactive_enabled = True
    d = Daemon(cfg)
    pending = d.queue.add(make_spec("p1"))
    running = d.queue.add(make_spec("r1"))
    d.queue.mark_running(running, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")

    d._sample_progress(1000.0)  # 不应抛异常

    assert pending.progress == {}
    assert running.progress == {}


# --------------------------------------------------------------------------- #
# 飞书交互查询: 状态快照与生命周期
# --------------------------------------------------------------------------- #
def test_bot_context_contains_gpus_tasks_and_eta(cfg):
    d = Daemon(cfg)
    d.snapshot = Snapshot(
        gpus=[
            GpuInfo(index=0, uuid="GPU-0", memory_total_mb=49140, memory_used_mb=100, utilization=0),
            GpuInfo(index=7, uuid="GPU-7", memory_total_mb=49140, memory_used_mb=5, utilization=0),
        ]
    )
    d.mode = MODE_IDLE
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.started_at = 1000.0
    task.progress = record_sample(None, sample_for("Epoch 10/100", 1000.0))
    task.progress = record_sample(task.progress, sample_for("Epoch 40/100", 1600.0))

    ctx = d._bot_context()

    assert ctx.server == "testhost" and ctx.mode == MODE_IDLE
    assert [g.index for g in ctx.gpus] == [0, 7]
    assert ctx.gpus[0].task_name == "t1", "占用该卡的自己的任务要标出名字"
    assert ctx.gpus[1].task_name == ""
    view = next(t for t in ctx.tasks if t.name == "t1")
    assert view.status == "running"
    assert "Epoch 40/100" in view.progress_text
    assert "预计" in view.eta_text, "两个采样点应能外推出 ETA"
    assert view.eta_confidence in ("high", "medium")
    assert ctx.log_reader is not None


def test_bot_context_log_reader_only_knows_our_tasks(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    log_file = write_task_log(cfg, task.safe_name, "line-1\nline-2\n")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file=log_file)

    ctx = d._bot_context()
    assert ctx.log_reader is not None
    assert ctx.log_reader("t1", 5) == ["line-1", "line-2"]
    assert ctx.log_reader("不存在的任务", 5) is None


def test_start_bot_skipped_when_disabled(cfg):
    d = Daemon(cfg)
    cfg.feishu.interactive_enabled = False

    d._start_bot()

    assert d.bot is None


def test_start_bot_skipped_without_credentials(cfg):
    d = Daemon(cfg)
    cfg.feishu.interactive_enabled = True
    cfg.feishu.app_id = ""
    cfg.feishu.app_secret = ""

    d._start_bot()

    assert d.bot is None, "凭证不全时不应启动（也不会影响调度）"


def test_start_bot_degrades_gracefully_when_sdk_missing(cfg):
    """未安装 lark-oapi 时只记日志，不抛异常、不影响调度。"""
    if importlib.util.find_spec("lark_oapi") is not None:
        pytest.skip("环境已安装 lark-oapi")
    d = Daemon(cfg)
    cfg.feishu.interactive_enabled = True
    cfg.feishu.app_id = "cli_unit_test"
    cfg.feishu.app_secret = dummy_credential()
    cfg.feishu.allowed_users = ["ou_me"]

    d._start_bot()

    assert d.bot is None


def test_start_bot_never_raises_when_sdk_load_fails(cfg, monkeypatch):
    """回归: SDK “装了但加载失败”（非 ImportError）曾把 daemon 启动路径拖崩。

    真实案例: setuptools 过旧 + Python 3.12 → pkg_resources 抛 AttributeError。
    """
    import gpu_manager.bot as bot_mod

    def boom(name: str):
        raise AttributeError("module 'pkgutil' has no attribute 'ImpImporter'")

    monkeypatch.setattr(bot_mod.importlib, "import_module", boom)
    d = Daemon(cfg)
    cfg.feishu.interactive_enabled = True
    cfg.feishu.app_id = "cli_unit_test"
    cfg.feishu.app_secret = dummy_credential()
    cfg.feishu.allowed_users = ["ou_me"]

    d._start_bot()  # 不应抛出任何异常

    assert d.bot is None, "加载失败时只降级，daemon 照常跑"


def test_start_bot_skipped_in_dry_run(cfg):
    d = Daemon(cfg, dry_run=True)
    cfg.feishu.interactive_enabled = True
    cfg.feishu.app_id = "cli_unit_test"
    cfg.feishu.app_secret = dummy_credential()

    d._start_bot()

    assert d.bot is None, "--dry-run 不应建立对外长连接"


def test_stop_bot_stops_and_clears(cfg):
    d = Daemon(cfg)
    stopped = {"count": 0}

    class FakeBot:
        def stop(self, timeout: float = 3.0) -> None:
            stopped["count"] += 1

    d.bot = FakeBot()  # type: ignore[assignment]

    d._stop_bot()

    assert stopped["count"] == 1 and d.bot is None


def test_reload_restarts_bot(cfg, monkeypatch):
    """连接参数（凭证）变了才重建长连接。"""
    d = Daemon(cfg)
    calls: list[str] = []
    monkeypatch.setattr(d, "_stop_bot", lambda: calls.append("stop"))
    monkeypatch.setattr(d, "_start_bot", lambda: calls.append("start"))
    raw = yaml.safe_load(cfg.source_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("feishu", {}).update(
        {"interactive_enabled": True, "app_id": "cli_changed", "app_secret": dummy_credential()}
    )
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    cfg.source_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    d._reload_config()

    assert calls == ["stop", "start"]


def test_reload_updates_allowlist_in_place_without_restart(cfg, monkeypatch):
    """回归: 改白名单不应重建长连接。

    重建会留下无法关闭的旧连接，它继续用旧白名单应答 → 用户改完白名单仍然被拒
    （实测踩到过）。因此白名单/限流属于“处理消息时才读”的设置，原地更新即可。
    """
    d = Daemon(cfg)
    stub = SimpleNamespace(
        allowlist=AllowList(["ou_old"]),
        rate_limiter=RateLimiter(per_minute=10),
        router=CommandRouter(),
    )
    d.bot = stub  # type: ignore[assignment]
    restarts: list[str] = []
    monkeypatch.setattr(d, "_stop_bot", lambda: restarts.append("stop"))
    monkeypatch.setattr(d, "_start_bot", lambda: restarts.append("start"))
    raw = yaml.safe_load(cfg.source_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("feishu", {})["allowed_users"] = ["ou_new"]
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    cfg.source_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    d._reload_config()

    assert restarts == [], "白名单变更不应重建长连接"
    assert stub.allowlist.is_allowed("ou_new") and not stub.allowlist.is_allowed("ou_old")


def test_reload_command_does_not_schedule_second_reload(cfg, monkeypatch):
    """回归: 一条 reload 命令只能重载一次。

    早期实现既同步重载、又置 _reload_requested，导致下一个 tick 再重载一次
    （于是长连接被启动了两次，而旧连接无法关闭）。
    """
    d = Daemon(cfg)
    calls: list[int] = []
    monkeypatch.setattr(d, "_reload_config", lambda: calls.append(1))

    d.handle_command("reload", {})

    assert calls == [1]
    assert d._reload_requested is False, "不得留下标志让下一个 tick 再重载一次"


def test_status_payload_reports_interactive(cfg):
    d = Daemon(cfg)
    payload = d.status_payload()
    assert payload["interactive"] == {
        "enabled": False,
        "running": False,
        "allowed_users": 0,
        "rate_limit_per_minute": 10,
        "eta_enabled": True,
    }


# --------------------------------------------------------------------------- #
# 重启后接管运行中任务
# --------------------------------------------------------------------------- #
def test_recover_marks_dead_task_failed_without_exit_code(cfg):
    """进程已消失且没有退出码记录 → 保守判为 failed（不当作成功）。"""
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=4194303, pgid=4194303, proc_start_time=None, log_file="/l")
    d.state = {}

    d._recover_after_restart()

    assert task.status == FAILED
    assert task.exit_code is None
    assert "退出码" in task.error


def test_recover_keeps_live_task_running(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    from gpu_manager.utils import process_start_time

    pid = os.getpid()
    d.queue.mark_running(
        task, gpu_ids=[0], pid=pid, pgid=pid,
        proc_start_time=process_start_time(pid), log_file="/l",
    )

    d._recover_after_restart()

    assert task.status == RUNNING, "身份校验通过的任务必须继续被跟踪"
    assert task.pid == pid


def test_recover_refuses_to_track_reused_pid(cfg):
    """pid 被复用（starttime 不符）→ 解除跟踪，但绝不发信号。"""
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(
        task, gpu_ids=[0], pid=os.getpid(), pgid=os.getpid(),
        proc_start_time=1.0, log_file="/l",  # 明显错误的 starttime
    )

    d._recover_after_restart()

    assert task.status == FAILED
    assert "身份校验失败" in task.error
    assert os.getpid() > 0  # 我们还活着（说明没有对自己发信号）


@pytest.mark.parametrize("status", [COMPLETED, FAILED])
def test_recover_uses_persisted_exit_code(cfg, status, tmp_path):
    """有 .exit 文件时应据此判定成败（daemon 重启也不丢退出码）。"""
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    log_dir = cfg.task_log_dir / task.safe_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    log_file.write_text("done\n", encoding="utf-8")
    expected = 0 if status == COMPLETED else 3
    log_file.with_suffix(".exit").write_text(f"{expected}\n", encoding="utf-8")

    d.queue.mark_running(
        task, gpu_ids=[0], pid=4194303, pgid=4194303, proc_start_time=None, log_file=str(log_file)
    )
    d._recover_after_restart()

    assert task.status == status
    assert task.exit_code == expected


# --------------------------------------------------------------------------- #
# 进度来源 / 阶段 / 总轮数（真实训练脚本的用法）
# --------------------------------------------------------------------------- #
def test_sample_progress_reads_user_declared_eta_logs(cfg, workdir):
    """训练脚本把输出重定向到自己的 console.log —— 采样必须能读那里。"""
    cfg.feishu.interactive_enabled = True
    d = Daemon(cfg)
    task = d.queue.add(
        make_spec("t1", cwd=str(workdir), eta_logs=["console.log"], total_epochs=500)
    )
    console = workdir / "console.log"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    console.write_text("Epoch: [141]  [10/63]  eta: 0:00:53  lr: 0.000040\n", encoding="utf-8")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")

    d._sample_progress(1000.0)

    assert task.progress, "应从 eta_logs 声明的文件取到进度"
    sample = latest_samples(task.progress)[1][-1]
    assert (sample.done, sample.total) == (141 * 63 + 10, 500 * 63)


def test_sample_progress_ignores_unchanged_progress(cfg, workdir):
    """日志没增长时不得伪造新样本（否则会在 eval 阶段用训练速度外推）。"""
    cfg.feishu.interactive_enabled = True
    d = Daemon(cfg)
    task = d.queue.add(
        make_spec("t1", cwd=str(workdir), eta_logs=["console.log"], total_epochs=500)
    )
    console = workdir / "console.log"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    console.write_text("Epoch: [141]  [10/63]\n", encoding="utf-8")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")

    d._sample_progress(1000.0)
    first = dict(task.progress)
    d._last_progress_sample = 0.0
    d._sample_progress(2000.0)  # 日志内容没变

    assert task.progress == first, "进度未推进 → 不应新增样本"


def test_sample_progress_resets_on_stage_change(cfg, workdir):
    """阶段从 training 切到 evaluating 后，旧进度必须失效（不能跨阶段外推）。"""
    cfg.feishu.interactive_enabled = True
    d = Daemon(cfg)
    task = d.queue.add(
        make_spec(
            "t1",
            cwd=str(workdir),
            eta_logs=["console.log"],
            total_epochs=500,
            stage_file="stage.status",
        )
    )
    console = workdir / "console.log"
    stage = workdir / "stage.status"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    console.write_text("Epoch: [141]  [10/63]\n", encoding="utf-8")
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    stage.write_text("training\n", encoding="utf-8")
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")

    d._sample_progress(1000.0)
    assert task.stage == "training" and task.progress

    # 进入 eval：训练日志不再增长
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    stage.write_text("evaluating\n", encoding="utf-8")
    d._last_progress_sample = 0.0
    d._sample_progress(2000.0)

    assert task.stage == "evaluating"
    assert task.progress == {}, "阶段变化 + 日志停滞 → 不得留下可外推的旧进度"


def test_bot_context_exposes_stage(cfg):
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    task.stage = "evaluating"

    ctx = d._bot_context()

    assert next(t for t in ctx.tasks if t.name == "t1").stage == "evaluating"


# --------------------------------------------------------------------------- #
# 看门狗接入（tick / 持久化 / 事件）
# --------------------------------------------------------------------------- #
def make_running_task(
    d, cfg, workdir, name: str = "t1", *, log_age: float = 0.0, uptime: float = 20_000.0, **spec_kw
):
    """造一个 running 任务 + 真实日志文件（日志 mtime 与已运行时长都可控）。"""
    log = Path(cfg.task_log_dir) / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    log.write_text("Epoch: [1] [1/10] eta: 0:01:00\n", encoding="utf-8")
    if log_age:
        now = time.time()
        os.utime(log, (now - log_age, now - log_age))
    task = d.queue.add(make_spec(name, cwd=str(workdir), **spec_kw))
    d.queue.mark_running(
        task,
        gpu_ids=[0],
        pid=os.getpid(),
        pgid=os.getpgid(0),
        proc_start_time=process_start_time(os.getpid()),
        log_file=str(log),
    )
    # mark_running 会把 started_at 设为「现在」；看门狗把任务启动时间也当作一个活动
    # 信号，所以要显式造出「已经跑了很久」的样子，否则永远看不到停滞。
    task.started_at = time.time() - uptime
    return task


def test_watchdog_reports_inactive_without_running_tasks(cfg, workdir):
    """没有训练任务时看门狗必须关闭（需求：无任务即关掉看门狗）。"""
    d = Daemon(cfg)
    d._run_watchdog(time.time())
    d._persist(time.time())
    assert d.state["watchdog"]["active"] is False
    assert d.state["watchdog"]["watched"] == 0


def test_watchdog_alerts_writes_diagnostics_and_event(cfg, workdir):
    """停滞 → 告警 + 诊断快照 + events.jsonl + state.json 可视状态。"""
    cfg.watchdog.check_interval_seconds = 10
    cfg.watchdog.stall_threshold_seconds = 60
    cfg.watchdog.startup_grace_seconds = 0
    d = Daemon(cfg)
    make_running_task(d, cfg, workdir, log_age=10_000.0)
    now = time.time()

    d._run_watchdog(now)
    d._persist(now)

    state = d.state["watchdog"]
    assert state["active"] is True and state["watched"] == 1
    info = state["tasks"]["t1"]
    assert info["alert_kind"] == "stall" and info["alert_count"] == 1
    snapshot = Path(info["last_snapshot"])
    assert (snapshot / "summary.txt").exists()

    events = StateStore.read_events(cfg.event_log, 50)
    assert any(e.get("event") == "watchdog_stall" and e.get("task") == "t1" for e in events)


def test_watchdog_alert_survives_diagnostics_failure(cfg, workdir, monkeypatch):
    """诊断/ETA 出问题只能丢掉快照，**告警本身必须发出去**（否则等于没看门狗）。"""
    cfg.watchdog.check_interval_seconds = 10
    cfg.watchdog.stall_threshold_seconds = 60
    cfg.watchdog.startup_grace_seconds = 0
    d = Daemon(cfg)
    make_running_task(d, cfg, workdir, log_age=10_000.0)

    def boom(_task, *_a, **_k):
        raise RuntimeError("ETA 计算炸了")

    monkeypatch.setattr(d, "_watchdog_eta_text", boom)
    now = time.time()
    d._run_watchdog(now)

    events = StateStore.read_events(cfg.event_log, 50)
    assert any(e.get("event") == "watchdog_stall" for e in events)
    assert d.watchdog.watches["t1"].alert_kind == "stall"
    assert not list(Path(cfg.diagnostics_dir).glob("*/*")), "快照失败了就不该留目录"


def test_watchdog_recovers_when_task_finishes(cfg, workdir):
    """任务结束后看门狗清空状态并关闭（不残留陈旧监视）。"""
    cfg.watchdog.check_interval_seconds = 10
    d = Daemon(cfg)
    task = make_running_task(d, cfg, workdir, log_age=10_000.0)
    now = time.time()
    d._run_watchdog(now)
    assert d.watchdog.watches

    d.queue.mark_finished(task, status=COMPLETED, exit_code=0)
    d._run_watchdog(now + 20)

    assert d.watchdog.watches == {}
    assert d.watchdog.active is False


# --------------------------------------------------------------------------- #
# 终态任务记录清理（queue.finished_retention_hours）
# --------------------------------------------------------------------------- #
def test_purge_finished_records_disabled_when_retention_zero(cfg):
    cfg.queue.finished_retention_hours = 0
    d = Daemon(cfg)
    task = d.queue.add(make_spec("t1"))
    d.queue.mark_finished(task, status=COMPLETED, exit_code=0)
    task.finished_at = time.time() - 400 * 3600

    d._maybe_purge_finished(time.time())

    assert d.queue.get("t1") is not None, "保留时长为 0（永久保留）时不得删除"


def test_purge_finished_records_removes_expired_and_emits_event(cfg):
    cfg.queue.finished_retention_hours = 72
    d = Daemon(cfg)
    task = d.queue.add(make_spec("gone"))
    d.queue.mark_finished(task, status=COMPLETED, exit_code=0)
    task.finished_at = time.time() - 100 * 3600
    fresh = d.queue.add(make_spec("keep"))
    d.queue.mark_finished(fresh, status=FAILED, exit_code=1)

    d._maybe_purge_finished(time.time())

    assert d.queue.get("gone") is None, "超过窗口的终态记录应被真正删除"
    assert d.queue.get("keep") is not None
    events = StateStore.read_events(cfg.event_log, 20)
    assert any(
        e.get("event") == "purge_finished" and "gone" in (e.get("removed") or []) for e in events
    ), "删除要留下事件流记录（可追溯）"


def test_purge_finished_records_is_throttled(cfg):
    cfg.queue.finished_retention_hours = 72
    d = Daemon(cfg)
    now = time.time()
    d._maybe_purge_finished(now)  # 首次跑一轮（队列为空）
    task = d.queue.add(make_spec("later"))
    d.queue.mark_finished(task, status=FAILED, exit_code=1)
    task.finished_at = now - 100 * 3600

    d._maybe_purge_finished(now + 1)  # 未到周期 → 不清理
    assert d.queue.get("later") is not None

    d._maybe_purge_finished(now + daemon_mod.FINISHED_PURGE_INTERVAL + 1)
    assert d.queue.get("later") is None
