"""日志保留与磁盘占用控制测试。

锁定几条关键性质:
  · 以「一次运行的产物」为单位删除（.log/.log.1/.run.sh/.exit 整组一起走）
  · 每个任务只留最近 N 次；logs/ 总量不超上限（超出按最旧优先）
  · 超限时优先删**成功**任务的日志，失败任务的留久一点
  · **正在运行任务**的日志绝不删
  · events.jsonl / daemon stdout 按大小裁剪，只留尾部完整行
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gpu_manager.retention import (
    dir_usage_bytes,
    human_size,
    limits_text,
    prune_all,
    prune_aux_files,
    prune_task_logs,
    prune_watchdog_diagnostics,
    trim_tail,
    usage_overview,
)

MB = 1024 * 1024


def make_run(cfg, task: str, key: str, size: int = 100_000, mtime: float | None = None) -> Path:
    """造一次运行的产物: <key>.log + <key>.run.sh + <key>.exit。"""
    task_dir = cfg.task_log_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log = task_dir / f"{key}.log"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    log.write_text("x" * size, encoding="utf-8")
    (task_dir / f"{key}.run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (task_dir / f"{key}.exit").write_text("0\n", encoding="utf-8")
    if mtime is not None:
        for suffix in (".log", ".run.sh", ".exit"):
            os.utime(task_dir / f"{key}{suffix}", (mtime, mtime))
    return log


def keys_left(cfg, task: str) -> set[str]:
    """某任务目录下还留有哪些运行组（目录被清空后会被删掉，返回空集）。"""
    task_dir = cfg.task_log_dir / task
    if not task_dir.is_dir():
        return set()
    return {p.name.split(".", 1)[0] for p in task_dir.iterdir() if p.is_file()}


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def test_human_size():
    assert human_size(0) == "0B"
    assert human_size(512) == "512B"
    assert human_size(2048) == "2.0KB"
    assert human_size(5 * MB) == "5.0MB"


def test_dir_usage_ignores_symlinks(cfg):
    make_run(cfg, "t", "run1", size=1000)
    (cfg.task_log_dir / "t" / "latest.log").symlink_to("run1.log")
    usage = dir_usage_bytes(cfg.task_log_dir)
    assert usage >= 1000
    assert usage < 2000, "软链不应被重复计入"


def test_noop_when_under_limits(cfg):
    make_run(cfg, "t", "run1", size=1000)
    report = prune_task_logs(cfg)
    assert report.removed_files == []
    assert not report.changed


# --------------------------------------------------------------------------- #
# 每个任务保留最近 N 次
# --------------------------------------------------------------------------- #
def test_keeps_newest_n_runs_per_task(cfg):
    cfg.storage.task_log_keep_files_per_task = 2
    cfg.storage.task_log_dir_max_mb = 0  # 只看数量
    base = 1_700_000_000.0
    for i in range(4):
        make_run(cfg, "taskA", f"run{i}", size=1000, mtime=base + i * 100)

    report = prune_task_logs(cfg)

    assert keys_left(cfg, "taskA") == {"run3", "run2"}, "只保留最近 2 次"
    assert len(report.removed_files) == 6, "每次运行 3 个文件要一起删（不留孤立 run.sh）"


def test_count_limit_is_per_task(cfg):
    cfg.storage.task_log_keep_files_per_task = 1
    cfg.storage.task_log_dir_max_mb = 0
    base = 1_700_000_000.0
    for i in range(3):
        make_run(cfg, "A", f"a{i}", size=10, mtime=base + i)
        make_run(cfg, "B", f"b{i}", size=10, mtime=base + i)

    prune_task_logs(cfg)

    assert keys_left(cfg, "A") == {"a2"}
    assert keys_left(cfg, "B") == {"b2"}


# --------------------------------------------------------------------------- #
# 总量上限
# --------------------------------------------------------------------------- #
def test_total_size_cap_deletes_oldest_first(cfg):
    """总量超限时按最旧优先删（用严格递增的 mtime 保证顺序确定）。"""
    cfg.storage.task_log_keep_files_per_task = 0  # 只看总量，不看次数
    cfg.storage.task_log_dir_max_mb = 1
    base = 1_700_000_000.0
    # 旧任务（小）与新任务（大）合计约 2MB，超出 1MB 上限
    for i in range(5):
        make_run(cfg, "old_task", f"run{i}", size=100_000, mtime=base + i)
    for i in range(5):
        make_run(cfg, "new_task", f"run{i}", size=300_000, mtime=base + 100 + i)
    assert dir_usage_bytes(cfg.task_log_dir) > 1 * MB, "前置条件: 要真的超限"

    report = prune_task_logs(cfg)

    assert report.removed_files, "超过上限时必须清理"
    assert dir_usage_bytes(cfg.task_log_dir) <= 1 * MB
    assert not keys_left(cfg, "old_task"), "更旧的任务应优先被删"
    assert keys_left(cfg, "new_task"), "新任务的日志应保留（降到上限内即可）"


def test_prefers_keeping_failed_task_logs(cfg):
    cfg.storage.task_log_keep_files_per_task = 0
    cfg.storage.task_log_dir_max_mb = 1
    cfg.storage.prefer_keep_failed_logs = True
    base = 1_700_000_000.0
    # 失败任务的日志更旧，但仍应被优先保留
    make_run(cfg, "failed_task", "old1", size=300_000, mtime=base)
    make_run(cfg, "ok_task", "new1", size=300_000, mtime=base + 10)
    make_run(cfg, "ok_task", "new2", size=300_000, mtime=base + 20)
    make_run(cfg, "ok_task", "new3", size=300_000, mtime=base + 30)
    assert dir_usage_bytes(cfg.task_log_dir) > 1 * MB

    prune_task_logs(cfg, failed_task_names=["failed_task"])

    assert keys_left(cfg, "failed_task") == {"old1"}, "失败任务的日志应优先保留"
    assert dir_usage_bytes(cfg.task_log_dir) <= 1 * MB


def test_preference_can_be_disabled(cfg):
    cfg.storage.task_log_keep_files_per_task = 0
    cfg.storage.task_log_dir_max_mb = 1
    cfg.storage.prefer_keep_failed_logs = False
    base = 1_700_000_000.0
    make_run(cfg, "failed_task", "old1", size=300_000, mtime=base)
    for i in range(3):
        make_run(cfg, "ok_task", f"new{i}", size=300_000, mtime=base + 10 + i)

    prune_task_logs(cfg, failed_task_names=["failed_task"])

    # 关掉偏好后就是纯按时间删 → 最旧的（失败任务）先走
    assert keys_left(cfg, "failed_task") == set()


# --------------------------------------------------------------------------- #
# 安全性质
# --------------------------------------------------------------------------- #
def test_never_prunes_active_running_task_logs(cfg):
    cfg.storage.task_log_keep_files_per_task = 1
    cfg.storage.task_log_dir_max_mb = 0
    base = 1_700_000_000.0
    active_log = make_run(cfg, "running", "old_but_active", size=5000, mtime=base)
    make_run(cfg, "running", "newer", size=5000, mtime=base + 100)

    report = prune_task_logs(cfg, active_log_paths=[str(active_log)])

    assert report.skipped_active == 1
    assert active_log.exists(), "正在运行任务的日志绝不能被删"


def test_dry_run_reports_but_does_not_delete(cfg):
    cfg.storage.task_log_keep_files_per_task = 1
    cfg.storage.task_log_dir_max_mb = 0
    base = 1_700_000_000.0
    old = make_run(cfg, "t", "old", size=1000, mtime=base)
    make_run(cfg, "t", "new", size=1000, mtime=base + 100)

    report = prune_task_logs(cfg, dry_run=True)

    assert report.dry_run and report.removed_files
    assert old.exists(), "--dry-run 不得删除任何文件"
    assert "dry-run" in report.summary()


def test_removes_stale_latest_symlink_and_empty_dir(cfg):
    cfg.storage.task_log_keep_files_per_task = 0
    cfg.storage.task_log_dir_max_mb = 1
    base = 1_700_000_000.0
    log = make_run(cfg, "t", "old", size=1_500_000, mtime=base)
    link = cfg.task_log_dir / "t" / "latest.log"
    link.symlink_to(log.name)

    prune_task_logs(cfg)

    assert not log.exists()
    assert not link.exists(), "指向已删目标的 latest 软链要清掉"
    assert not (cfg.task_log_dir / "t").exists(), "空目录也应清掉"


# --------------------------------------------------------------------------- #
# 裁剪 events / stdout
# --------------------------------------------------------------------------- #
def test_trim_tail_keeps_tail_and_complete_lines(tmp_path):
    path = tmp_path / "big.log"
    lines = [f"line-{i:06d}\n" for i in range(20000)]
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    path.write_text("".join(lines), encoding="utf-8")
    before = path.stat().st_size
    assert before > 100_000, "前置条件: 文件要足够大才谈得上裁剪"

    freed = trim_tail(path, 100_000)

    after_size = path.stat().st_size
    assert freed == before - after_size, "返回值必须等于实际释放的字节数"
    assert freed > 0
    after = path.read_text(encoding="utf-8")
    assert len(after.encode()) <= 100_000
    assert after.splitlines()[-1] == "line-019999", "必须保留最新的尾部"
    assert not after.startswith("ine-"), "不能留下被截断的半行"


def test_trim_tail_noop_when_small(tmp_path):
    path = tmp_path / "small.log"
    path.write_text("hi\n", encoding="utf-8")
    assert trim_tail(path, 1024) == 0
    assert path.read_text(encoding="utf-8") == "hi\n"


def test_prune_aux_files_trims_events_and_out_log(cfg):
    cfg.storage.events_max_mb = 1
    cfg.storage.daemon_out_max_mb = 1
    cfg.event_log.parent.mkdir(parents=True, exist_ok=True)
    out_log = cfg.log_file.with_suffix(".out.log")
    events = "".join(json.dumps({"event": "x", "i": i}) + "\n" for i in range(20000))
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    cfg.event_log.write_text(events, encoding="utf-8")
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    out_log.write_text("t" * (2 * MB), encoding="utf-8")

    report = prune_aux_files(cfg)

    assert report.trimmed_bytes > 0
    assert cfg.event_log.stat().st_size <= 1 * MB
    assert out_log.stat().st_size <= 1 * MB
    tail = cfg.event_log.read_text(encoding="utf-8").splitlines()
    assert json.loads(tail[-1])["i"] == 19999, "保留的应是最新事件"


def test_prune_all_combines_both(cfg):
    cfg.storage.task_log_keep_files_per_task = 1
    cfg.storage.events_max_mb = 1
    base = 1_700_000_000.0
    for i in range(3):
        make_run(cfg, "t", f"run{i}", size=1000, mtime=base + i)
    cfg.event_log.parent.mkdir(parents=True, exist_ok=True)
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    cfg.event_log.write_text("y" * (2 * MB), encoding="utf-8")

    report = prune_all(cfg)

    assert keys_left(cfg, "t") == {"run2"}
    assert report.trimmed_bytes > 0
    assert report.changed


# --------------------------------------------------------------------------- #
# 展示辅助
# --------------------------------------------------------------------------- #
def test_usage_overview_and_limits_text(cfg):
    make_run(cfg, "t", "run1", size=2048)
    rows = usage_overview(cfg)
    assert rows and any(size >= 2048 for _name, size in rows)

    text = limits_text(cfg)
    assert "logs/" in text and "每任务" in text

    cfg.storage.task_log_dir_max_mb = 0
    cfg.storage.task_log_keep_files_per_task = 0
    cfg.storage.events_max_mb = 0
    cfg.storage.daemon_log_max_mb = 0
    cfg.storage.daemon_out_max_mb = 0
    # 诊断快照上限也算「上限」；全部为 0 时才该出现「未设置任何上限」
    cfg.watchdog.diagnostics_keep_per_task = 0
    cfg.watchdog.diagnostics_max_mb = 0
    assert "未设置任何上限" in limits_text(cfg)


def test_zero_limits_disable_pruning(cfg):
    """0 表示不限制：即使有很多旧日志也不删。"""
    cfg.storage.task_log_keep_files_per_task = 0
    cfg.storage.task_log_dir_max_mb = 0
    base = 1_700_000_000.0
    for i in range(6):
        make_run(cfg, "t", f"run{i}", size=500_000, mtime=base + i)

    report = prune_task_logs(cfg)

    assert report.removed_files == []
    assert len(keys_left(cfg, "t")) == 6


# --------------------------------------------------------------------------- #
# 看门狗诊断快照
# --------------------------------------------------------------------------- #
def make_snapshot(cfg, task: str, stamp: str, size: int = 100_000, mtime: float = 0.0) -> Path:
    """造一份诊断快照（目录名就是时间戳）。"""
    snap = cfg.diagnostics_dir / task / stamp
    snap.mkdir(parents=True, exist_ok=True)
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    (snap / "summary.txt").write_text("x" * size, encoding="utf-8")
    stamp_time = time.mktime(time.strptime(stamp[:15], "%Y%m%d-%H%M%S"))
    os.utime(snap, (stamp_time, stamp_time))
    return snap


def test_prune_diagnostics_keeps_per_task_limit(cfg):
    cfg.watchdog.diagnostics_keep_per_task = 2
    cfg.watchdog.diagnostics_max_mb = 0
    for i in range(5):
        make_snapshot(cfg, "taskA", f"2026010{i + 1}-120000")
    for i in range(2):
        make_snapshot(cfg, "taskB", f"2026010{i + 1}-120000")

    report = prune_watchdog_diagnostics(cfg)

    assert sorted(p.name for p in (cfg.diagnostics_dir / "taskA").iterdir()) == [
        "20260104-120000",
        "20260105-120000",
    ], "应保留最新的两份"
    assert len(list((cfg.diagnostics_dir / "taskB").iterdir())) == 2
    assert len(report.removed_files) == 3
    assert report.removed_bytes > 0


def test_prune_diagnostics_respects_total_cap_and_protects_newest(cfg):
    """总量超限时从最旧删，但每个任务最新那份永不删。"""
    cfg.watchdog.diagnostics_keep_per_task = 0
    cfg.watchdog.diagnostics_max_mb = 1
    for i in range(10):
        make_snapshot(cfg, "taskA", f"202601{i + 1:02d}-120000", size=200_000)
    make_snapshot(cfg, "taskB", "20260201-120000", size=200_000)

    prune_watchdog_diagnostics(cfg)

    assert dir_usage_bytes(cfg.diagnostics_dir) <= 1 * MB
    # 每个任务仍有内容，且留下的是最新那份
    assert (cfg.diagnostics_dir / "taskB" / "20260201-120000").is_dir()
    left = sorted(p.name for p in (cfg.diagnostics_dir / "taskA").iterdir())
    assert left == ["20260110-120000"], left


def test_prune_diagnostics_dry_run_changes_nothing(cfg):
    cfg.watchdog.diagnostics_keep_per_task = 1
    for i in range(4):
        make_snapshot(cfg, "taskA", f"2026010{i + 1}-120000")

    report = prune_watchdog_diagnostics(cfg, dry_run=True)

    assert len(report.removed_files) == 3
    assert len(list((cfg.diagnostics_dir / "taskA").iterdir())) == 4, "dry-run 不真删"


def test_prune_diagnostics_disabled_or_empty_is_noop(cfg):
    cfg.watchdog.diagnostics_enabled = False
    make_snapshot(cfg, "taskA", "20260101-120000")
    assert prune_watchdog_diagnostics(cfg).removed_files == []
    assert (cfg.diagnostics_dir / "taskA" / "20260101-120000").is_dir()

    cfg.watchdog.diagnostics_enabled = True
    assert prune_watchdog_diagnostics(cfg).removed_files == []  # 空目录也不报错


def test_prune_all_includes_diagnostics(cfg):
    make_run(cfg, "t", "run1", size=2048)
    cfg.watchdog.diagnostics_keep_per_task = 1
    for i in range(3):
        make_snapshot(cfg, "taskA", f"2026010{i + 1}-120000")

    report = prune_all(cfg)

    assert any("diagnostics" in name for name in report.removed_files)


def test_usage_overview_and_limits_include_diagnostics(cfg):
    make_snapshot(cfg, "taskA", "20260101-120000", size=4096)
    rows = usage_overview(cfg)
    assert any("诊断" in name and size >= 4096 for name, size in rows)
    assert "诊断快照" in limits_text(cfg)

    cfg.watchdog.diagnostics_enabled = False
    assert "诊断快照" not in limits_text(cfg)
