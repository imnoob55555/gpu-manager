"""看门狗测试。

锁定几条关键性质（都是需求里明确写下的行为）:
  · **没有运行中任务时整体关闭**：不检查、不留状态，有新任务时自动恢复
  · 三类信号（进度记录 / 日志 mtime / 进程状态）任一有动静就算有进展
  · 连续无进展超阈值才告警；启动宽限期内不判停滞
  · 进程异常（消失/僵尸/停止/不可中断/PID 复用）连续 N 轮确认后**立即**告警
  · 同一问题只告警一次，之后按间隔重复；有进展则发恢复通知
  · 告警时写诊断快照（含进程详情/日志尾部/进度），且**绝不对进程发信号**
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

import gpu_manager.watchdog as watchdog_module
from gpu_manager.task_queue import Task
from gpu_manager.utils import process_start_time
from gpu_manager.watchdog import Watchdog, format_status

# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


class FakeNotifier:
    """记录发出的通知（不联网）。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def send(self, title: str, body: str, level: str = "info", mention_all: bool = False):
        self.sent.append({"title": title, "body": body, "level": level})
        return True, "ok"


def make_watchdog(cfg, *, notifier=None, dry_run: bool = False, **overrides) -> Watchdog:
    for key, value in overrides.items():
        setattr(cfg.watchdog, key, value)
    return Watchdog(cfg, logging.getLogger("test.watchdog"), notifier, dry_run=dry_run)


def make_task(cfg, workdir, *, name: str = "t1", log_age: float = 0.0, **overrides) -> Task:
    """造一个 running 任务，日志文件真实存在（mtime 可控）。"""
    log = Path(cfg.task_log_dir) / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Epoch: [1] [1/10] eta: 0:01:00\n", encoding="utf-8")
    now = time.time()
    if log_age:
        os.utime(log, (now - log_age, now - log_age))
    pid = overrides.pop("pid", os.getpid())
    params = {
        "name": name,
        "cwd": str(workdir),
        "command": "python train.py",
        "status": "running",
        "pid": pid,
        "proc_start_time": process_start_time(pid),
        "log_file": str(log),
        "started_at": overrides.pop("started_at", now - overrides.pop("uptime", 10_000.0)),
        "gpu_ids": [0],
    }
    params.update(overrides)
    return Task(**params)


NOW = lambda: time.time()  # noqa: E731 - 测试里读起来更直白


# --------------------------------------------------------------------------- #
# 开关语义
# --------------------------------------------------------------------------- #
def test_disabled_watchdog_is_completely_passive(cfg, workdir):
    w = make_watchdog(cfg, enabled=False)
    task = make_task(cfg, workdir, log_age=10_000)
    assert w.update([task], now=NOW()) == []
    assert w.active is False
    assert w.watches == {}
    assert w.to_dict()["enabled"] is False


def test_no_running_tasks_switches_off_and_clears_state(cfg, workdir):
    w = make_watchdog(cfg, check_interval_seconds=10, stall_threshold_seconds=60)
    task = make_task(cfg, workdir)
    now = NOW()
    w.update([task], now=now)
    assert w.active is True and "t1" in w.watches

    w.update([], now=now + 1)
    assert w.active is False
    assert w.watches == {}
    assert w.last_check_at == 0.0

    # 再来任务 → 自动恢复
    w.update([task], now=now + 2)
    assert w.active is True and "t1" in w.watches


def test_finished_task_is_dropped(cfg, workdir):
    w = make_watchdog(cfg, check_interval_seconds=10, stall_threshold_seconds=60)
    task = make_task(cfg, workdir)
    now = NOW()
    w.update([task], now=now)
    task.status = "completed"
    w.update([task], now=now + 1)
    assert "t1" not in w.watches
    assert w.active is False


def test_check_interval_throttles(cfg, workdir):
    w = make_watchdog(cfg, check_interval_seconds=100, stall_threshold_seconds=1000)
    task = make_task(cfg, workdir)
    now = NOW()
    w.update([task], now=now)
    assert w.check_count == 1
    w.update([task], now=now + 50)
    assert w.check_count == 1, "未到检查周期不应检查"
    w.update([task], now=now + 101)
    assert w.check_count == 2


# --------------------------------------------------------------------------- #
# 三类信号
# --------------------------------------------------------------------------- #
def test_fresh_log_suppresses_stall_then_stalls_when_it_freezes(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    notifier = FakeNotifier()
    w.notifier = notifier
    task = make_task(cfg, workdir, uptime=20_000, log_age=0)

    now = NOW()
    assert w.update([task], now=now) == []
    # 日志停住不动 70 秒 → 停滞
    events = w.update([task], now=now + 70)
    assert [e.kind for e in events] == ["stall"]
    assert "无任何进展" in events[0].detail
    assert notifier.sent and notifier.sent[-1]["level"] == "warning"


def test_progress_record_counts_as_activity(cfg, workdir):
    """ETA 采样写入的新进度样本也算「有进展」，即使日志没动。"""
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    task.progress_updated_at = now - 5
    assert w.update([task], now=now) == []
    assert w.watches["t1"].activity_source == "进度记录"


def test_missing_log_file_falls_back_to_started_at(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    task = make_task(cfg, workdir, uptime=10_000)
    Path(task.log_file).unlink()
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["stall"]
    assert w.watches["t1"].activity_source == "任务启动"


def test_future_log_mtime_is_clamped_to_now(cfg, workdir):
    """时钟偏移不该让「最后活动」显示成未来时间。"""
    w = make_watchdog(cfg, check_interval_seconds=10, stall_threshold_seconds=60)
    task = make_task(cfg, workdir, uptime=10_000)
    future = NOW() + 99_999
    os.utime(task.log_file, (future, future))
    now = NOW()
    w.update([task], now=now)
    assert w.watches["t1"].last_activity <= now


def test_startup_grace_suppresses_early_alert(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=600
    )
    task = make_task(cfg, workdir, uptime=5, log_age=0)
    now = NOW()
    assert w.update([task], now=now + 61) == [], "宽限期内不判停滞"

    strict = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    strict.cfg.watchdog.stall_threshold_seconds = 60
    events = strict.update([task], now=now + 61)
    assert [e.kind for e in events] == ["stall"]


# --------------------------------------------------------------------------- #
# 告警去重 / 重复 / 恢复
# --------------------------------------------------------------------------- #
def test_stall_alerts_repeats_then_recovers(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        repeat_alert_interval_seconds=120,
    )
    notifier = FakeNotifier()
    w.notifier = notifier
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)

    now = NOW()
    first = w.update([task], now=now)
    assert [e.kind for e in first] == ["stall"]
    assert first[0].alert_index == 1

    assert w.update([task], now=now + 60) == [], "未到重复间隔不应重复提醒"

    second = w.update([task], now=now + 130)
    assert [e.kind for e in second] == ["stall"]
    assert second[0].alert_index == 2
    assert "第 2 次提醒" in second[0].title

    # 日志恢复增长 → 恢复通知
    os.utime(task.log_file, (now + 140, now + 140))
    third = w.update([task], now=now + 141)
    assert [e.kind for e in third] == ["recovered"]
    assert third[0].level == "success"
    assert notifier.sent[-1]["level"] == "success"
    watch = w.watches["t1"]
    assert watch.alert_kind == ""
    assert watch.alert_reason == ""
    assert watch.last_issue, "恢复后应留下「上次问题」供展示"

    # 恢复后未再停够阈值 → 不重复告警
    assert w.update([task], now=now + 180) == []
    # 再次停够阈值 → 当作新问题重新告警（计数从 1 重新开始）
    again = w.update([task], now=now + 205)
    assert [e.kind for e in again] == ["stall"]
    assert again[0].alert_index == 1


def test_repeat_zero_alerts_only_once(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        repeat_alert_interval_seconds=0,
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    assert [e.kind for e in w.update([task], now=now)] == ["stall"]
    assert w.update([task], now=now + 100) == []
    assert w.update([task], now=now + 200) == []


def test_recovery_notification_can_be_disabled(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        repeat_alert_interval_seconds=0,
        notify_on_recovery=False,
    )
    notifier = FakeNotifier()
    w.notifier = notifier
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    w.update([task], now=now)
    os.utime(task.log_file, (now + 61, now + 61))
    events = w.update([task], now=now + 62)
    assert [e.kind for e in events] == ["recovered"]
    assert events[0].notified is False, "配置关闭时只记事件、不推送"
    assert len(notifier.sent) == 1, "只应有那一条停滞告警"


# --------------------------------------------------------------------------- #
# 进程异常
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("state", "alive", "expected"),
    [
        ("Z", False, "anomaly:zombie"),
        ("", False, "anomaly:gone"),
        ("T", True, "anomaly:stopped"),
        ("D", True, "anomaly:uninterruptible"),
    ],
)
def test_process_anomaly_needs_confirmation(
    cfg, workdir, monkeypatch, state, alive, expected
):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=999_999,
        startup_grace_seconds=0,
        anomaly_confirm_checks=2,
    )
    notifier = FakeNotifier()
    w.notifier = notifier
    task = make_task(cfg, workdir, log_age=0)
    monkeypatch.setattr(watchdog_module, "process_alive", lambda pid: alive)
    monkeypatch.setattr(watchdog_module, "process_state", lambda pid: state)

    now = NOW()
    assert w.update([task], now=now) == [], "第 1 轮只计数，不告警"
    events = w.update([task], now=now + 11)
    assert [e.kind for e in events] == [expected]
    assert events[0].level in ("warning", "error")
    assert events[0].snapshot_dir, "异常告警也应留下诊断"


def test_anomaly_confirm_one_alerts_immediately(cfg, workdir, monkeypatch):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=999_999,
        anomaly_confirm_checks=1,
    )
    task = make_task(cfg, workdir, log_age=0)
    monkeypatch.setattr(watchdog_module, "process_alive", lambda pid: False)
    monkeypatch.setattr(watchdog_module, "process_state", lambda pid: "Z")
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["anomaly:zombie"]
    assert events[0].level == "error"


def test_pid_reuse_detected_as_identity_anomaly(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=999_999,
        anomaly_confirm_checks=1,
    )
    task = make_task(cfg, workdir, log_age=0, proc_start_time=1.0)  # 与真实 starttime 不符
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["anomaly:identity"]
    assert "PID 复用" in events[0].detail


def test_anomaly_can_be_disabled_falling_back_to_stall(cfg, workdir, monkeypatch):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        instant_process_anomaly=False,
        anomaly_confirm_checks=1,
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    monkeypatch.setattr(watchdog_module, "process_alive", lambda pid: False)
    monkeypatch.setattr(watchdog_module, "process_state", lambda pid: "Z")
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["stall"], "关闭立即告警后应走停滞阈值"


# --------------------------------------------------------------------------- #
# 诊断快照
# --------------------------------------------------------------------------- #
def test_diagnostics_snapshot_contents(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000, total_epochs=500)
    events = w.update([task], now=NOW())
    snap = Path(events[0].snapshot_dir)

    assert snap.parent.name == "t1"
    assert {"summary.txt", "process.txt", "log_tail.txt", "progress.json"} <= {
        p.name for p in snap.iterdir()
    }
    summary = (snap / "summary.txt").read_text(encoding="utf-8")
    assert "GPU Manager 看门狗诊断快照" in summary
    assert "t1" in summary
    assert str(task.log_file) in summary
    assert "未对任何进程发送信号" in summary

    process = (snap / "process.txt").read_text(encoding="utf-8")
    assert f"/proc/{os.getpid()}/status" in process

    tail = (snap / "log_tail.txt").read_text(encoding="utf-8")
    assert "eta: 0:01:00" in tail

    assert w.watches["t1"].last_snapshot == str(snap)


def test_diagnostics_disabled_writes_nothing(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        diagnostics_enabled=False,
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["stall"]
    assert events[0].snapshot_dir == ""
    assert not Path(cfg.diagnostics_dir).exists()


def test_dry_run_does_not_notify_or_write(cfg, workdir):
    notifier = FakeNotifier()
    w = make_watchdog(
        cfg,
        notifier=notifier,
        dry_run=True,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    events = w.update([task], now=NOW())
    assert [e.kind for e in events] == ["stall"]
    assert events[0].notified is False
    assert notifier.sent == []
    assert events[0].snapshot_dir == ""
    assert not Path(cfg.diagnostics_dir).exists()


def test_eta_provider_only_called_when_alerting(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    calls: list[str] = []

    def provider(task):
        calls.append(task.name)
        return "ETA: 约 2h（依据: 进度样本）"

    task = make_task(cfg, workdir, log_age=0)
    now = NOW()
    w.update([task], now=now, eta_provider=provider)
    assert calls == [], "没告警就不该算 ETA（不做无谓工作）"

    events = w.update([task], now=now + 70, eta_provider=provider)
    assert calls == ["t1"]
    summary = Path(events[0].snapshot_dir, "summary.txt").read_text(encoding="utf-8")
    assert "ETA: 约 2h" in summary


def test_snapshot_cleanup_keeps_limit(cfg, workdir):
    w = make_watchdog(
        cfg,
        check_interval_seconds=1,
        stall_threshold_seconds=1,
        startup_grace_seconds=0,
        repeat_alert_interval_seconds=1,
        diagnostics_keep_per_task=2,
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    # 注意: 检查周期有 10s 下限（防止配置写得太激进），所以步长要 >10s
    for step in range(5):
        w.update([task], now=now + step * 20)
    snapshots = sorted(p for p in (Path(cfg.diagnostics_dir) / "t1").iterdir())
    assert len(snapshots) == 2, f"每任务只应保留 2 份，实际 {snapshots}"


# --------------------------------------------------------------------------- #
# 安全边界
# --------------------------------------------------------------------------- #
def test_watchdog_only_uses_signal_zero(cfg, workdir, monkeypatch):
    """看门狗必须只读：只允许 os.kill(pid, 0) 这类存在性检查，绝不发真信号。"""
    real_kill = os.kill

    def guarded(pid, sig):
        if sig != 0:
            raise AssertionError(f"看门狗不得对进程发信号: pid={pid} sig={sig}")
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", guarded)
    w = make_watchdog(
        cfg,
        check_interval_seconds=10,
        stall_threshold_seconds=60,
        startup_grace_seconds=0,
        anomaly_confirm_checks=1,
    )
    task = make_task(cfg, workdir, log_age=0)
    now = NOW()
    w.update([task], now=now)  # 正常路径
    w.update([task], now=now + 70)  # 停滞告警 + 写诊断
    monkeypatch.setattr(watchdog_module, "process_alive", lambda pid: False)
    monkeypatch.setattr(watchdog_module, "process_state", lambda pid: "")
    events = w.update([task], now=now + 80)  # 异常路径
    assert [e.kind for e in events] == ["anomaly:gone"]


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def test_to_dict_and_format_status(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    w.update([task], now=now)
    data = w.to_dict()
    assert data["enabled"] is True and data["active"] is True
    assert data["watched"] == 1 and data["check_count"] == 1
    info = data["tasks"]["t1"]
    assert info["alert_kind"] == "stall" and info["alert_count"] == 1
    assert info["idle_seconds"] >= 0

    lines = format_status(data, now=now)
    joined = "\n".join(lines)
    assert "看门狗: 运行中" in joined
    assert "t1" in joined and "最后进展" in joined


def test_format_status_handles_disabled_and_empty():
    assert "已禁用" in "\n".join(format_status({"enabled": False, "active": False}))
    assert "无（没有运行中的任务）" in "\n".join(
        format_status({"enabled": True, "active": False, "tasks": {}})
    )
    assert "无状态信息" in "\n".join(format_status({}))


def test_reconfigure_keeps_alert_history(cfg, workdir):
    w = make_watchdog(
        cfg, check_interval_seconds=10, stall_threshold_seconds=60, startup_grace_seconds=0
    )
    task = make_task(cfg, workdir, uptime=20_000, log_age=20_000)
    now = NOW()
    w.update([task], now=now)
    assert w.watches["t1"].alert_count == 1

    cfg.watchdog.stall_threshold_seconds = 7200
    w.reconfigure(cfg)
    assert w.watches["t1"].alert_count == 1, "换阈值不应清掉告警历史（否则会重复刷屏）"
    assert w.wcfg.stall_threshold_seconds == 7200
