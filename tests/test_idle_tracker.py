"""连续空闲确认测试。"""

from __future__ import annotations

import time

from gpu_manager.idle_tracker import IdleTracker


def test_requires_consecutive_checks():
    tracker = IdleTracker(required_checks=3)
    assert tracker.update([0], [0, 1], now=100.0) == []
    assert tracker.update([0], [0, 1], now=130.0) == []
    assert tracker.update([0], [0, 1], now=160.0) == [0], "连续第 3 次才确认"
    assert tracker.counts[0] == 3


def test_busy_interval_resets_counter():
    """训练在 checkpoint/DataLoader 等待后 util 回到高位 → 计数必须清零。"""
    tracker = IdleTracker(required_checks=3)
    tracker.update([0], [0], now=100.0)
    tracker.update([0], [0], now=130.0)
    assert tracker.counts[0] == 2
    tracker.update([], [0], now=160.0)  # 变忙
    assert tracker.counts[0] == 0
    tracker.update([0], [0], now=190.0)
    assert tracker.counts[0] == 1
    assert tracker.confirmed() == []


def test_multiple_gpus_progress_independently():
    tracker = IdleTracker(required_checks=2)
    tracker.update([0, 1], [0, 1, 2], now=100.0)
    confirmed = tracker.update([0], [0, 1, 2], now=130.0)
    assert confirmed == [0]
    assert tracker.counts[1] == 0
    assert tracker.counts[2] == 0


def test_stale_state_is_reset_after_restart():
    """daemon 停了很久后重启，陈旧的连续计数不可信 → 必须清零重新确认。"""
    data = {"counts": {"0": 3}, "since": {"0": 1000.0}, "updated_at": 1000.0}
    tracker = IdleTracker.from_dict(data, required_checks=3, max_age_seconds=60.0, now=2000.0)
    assert tracker.counts == {}, "陈旧计数必须清零（否则重启瞬间就会误占 GPU）"


def test_fresh_state_is_restored():
    now = time.time()
    data = {"counts": {"0": 2}, "since": {"0": now - 10}, "updated_at": now - 10}
    tracker = IdleTracker.from_dict(data, required_checks=3, max_age_seconds=300.0, now=now)
    assert tracker.counts == {0: 2}


def test_gpu_out_of_monitor_scope_is_dropped():
    tracker = IdleTracker(required_checks=1)
    tracker.update([0, 5], [0, 5], now=100.0)
    assert set(tracker.counts) == {0, 5}
    tracker.update([0], [0], now=130.0)  # 5 号卡不再被监控
    assert set(tracker.counts) == {0}


def test_reset_clears_selected_gpus():
    tracker = IdleTracker(required_checks=1)
    tracker.update([0, 1], [0, 1], now=100.0)
    tracker.reset([0])
    assert tracker.counts[0] == 0
    assert tracker.counts[1] == 1


def test_progress_and_streak_text():
    tracker = IdleTracker(required_checks=3)
    tracker.update([0], [0], now=100.0)
    progress = tracker.progress([0, 1])
    assert progress[0] == (1, False)
    assert progress[1] == (0, False)
    assert tracker.streak_text(0).startswith("1 ")
    assert tracker.streak_text(1) == "0"


def test_confirmed_respects_monitor_filter():
    tracker = IdleTracker(required_checks=1)
    tracker.update([0, 1], [0, 1], now=100.0)
    assert tracker.confirmed([1]) == [1]
    assert tracker.confirmed([5]) == []


def test_from_dict_tolerates_garbage():
    tracker = IdleTracker.from_dict(
        {"counts": {"x": "y"}, "since": None, "updated_at": time.time()},
        required_checks=2,
        max_age_seconds=999.0,
    )
    assert tracker.counts == {}
