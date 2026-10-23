"""ETA（进度解析与结束时间估算）测试。

锁定这些性质:
  · 能认出 tqdm / Epoch / step / iter / 裸分数等真实日志形状
  · 取**最后**出现的进度（日志里有大量历史进度行）
  · 拒绝假进度（done > total、配置打印里的无关数字）
  · 进度回退（换 epoch / total 变化 / 新阶段）→ 重置，绝不用旧数据外推
  · tqdm 自带剩余时间优先；其次两次采样算速率；再次历史中位耗时；最后明确说"无法估算"
  · 超长行/大量行不拖垮采样（有界量词 + 只扫尾部）
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from gpu_manager.eta import (
    DEFAULT_PATTERNS,
    MAX_SCAN_LINES,
    EtaEstimate,
    ProgressSample,
    estimate,
    format_eta,
    format_progress,
    latest_samples,
    median_history,
    parse_duration,
    parse_progress,
    record_sample,
)

TQDM_FULL = "45%|███████████          | 450/1000 [01:23<02:05, 3.60it/s]"
TQDM_DONE = "100%|██████████| 1000/1000 [10:00<00:00, 1.6it/s]"


def sample_for(line: str, at: float = 0.0) -> ProgressSample:
    """解析一行日志并断言成功（测试里到处要用，避免重复写 assert is not None）。"""
    sample = parse_progress(line, now=at)
    assert sample is not None, f"未能解析出进度: {line!r}"
    return sample


def add_sample(phases: dict[str, list[dict[str, Any]]] | None, line: str, at: float = 0.0) -> dict:
    """记录一条进度样本（解析失败即测试失败）。"""
    return record_sample(phases, sample_for(line, at))


# --------------------------------------------------------------------------- #
# 时间解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("02:05", 125.0),
        ("1:02:05", 3725.0),
        ("05", 5.0),
        ("0:00", 0.0),
        ("?", None),
        ("", None),
        (None, None),
        ("abc", None),
        ("1:2:3:4", None),
        ("-1:00", None),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


# --------------------------------------------------------------------------- #
# 进度解析
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "line,label,done,total",
    [
        (TQDM_FULL, "", 450, 1000),
        ("| 450/1000 [01:23<02:05, 3.60it/s]", "", 450, 1000),
        ("Epoch 12/500 [00:30<20:00] loss=0.42", "epoch", 12, 500),
        ("epoch: 12/500", "epoch", 12, 500),
        ("[2026-10-02] step 1500/5000 lr=1e-4", "step", 1500, 5000),
        ("iter 7/10", "iter", 7, 10),
        ("iteration 3/8", "iteration", 3, 8),
        ("  [  12/500]  ", "", 12, 500),
        ("(12/500)", "", 12, 500),
    ],
)
def test_parse_progress_shapes(line, label, done, total):
    sample = parse_progress(line, now=1000.0)
    assert sample is not None, f"未识别: {line}"
    assert (sample.done, sample.total) == (done, total)
    if label:
        assert sample.label.lower() == label


def test_parse_progress_extracts_tqdm_remaining_and_percent():
    sample = sample_for(TQDM_FULL)
    assert sample.remaining_seconds == 125.0
    assert sample.percent == 45.0
    assert sample.fraction == pytest.approx(0.45)


@pytest.mark.parametrize(
    "line",
    [
        "Checkpoint saved to exps/baseline/ckpt.pth",
        "Loading dataset...",
        "500/100?",
        "",
        "   ",
        "config: batch_size=7 num_queries=100",
    ],
)
def test_parse_progress_rejects_non_progress(line):
    assert parse_progress(line, now=0.0) is None, f"不该把非进度行当成进度: {line!r}"


def test_parse_progress_accepts_fully_completed_progress():
    sample = sample_for("Epoch 500/500")
    assert sample.done == sample.total == 500


def test_parse_progress_rejects_done_greater_than_total():
    assert parse_progress("Epoch 900/500", now=0.0) is None


def test_parse_progress_takes_the_last_progress_line():
    log = "\n".join(["Epoch 1/100", "Epoch 2/100", "Epoch 3/100", "loss=0.31"])
    assert sample_for(log).done == 3


def test_parse_progress_prefers_last_line_with_progress_in_mixed_log():
    log = "\n".join(["Epoch 1/100", TQDM_FULL])
    assert sample_for(log).done == 450, "应取最后一次出现的进度"


def test_parse_progress_accepts_custom_patterns():
    line = "PROGRESS 42 of 100 batches"
    assert parse_progress(line, now=0.0) is None, "默认模式不该认识它"
    custom = [r"(?P<done>\d{1,10}) of (?P<total>\d{1,10}) batches"]
    sample = parse_progress(line, patterns=custom, now=0.0)
    assert sample is not None and sample.done == 42 and sample.total == 100


def test_parse_progress_skips_broken_custom_pattern():
    """用户写错的正则不能让采样崩掉或卡死。"""
    custom = ["(unclosed", *DEFAULT_PATTERNS]
    sample = parse_progress(TQDM_FULL, patterns=custom, now=0.0)
    assert sample is not None and sample.done == 450


def test_parse_progress_handles_long_lines_and_noise():
    """真实 tqdm 长进度条: 进度数字在行尾，前面可能有几万字符。"""
    line = "noise " * 30_000 + "| 450/1000 [01:23<02:05, 3.6it/s]"
    started = time.monotonic()
    sample = parse_progress(line, now=0.0)
    assert sample is not None and sample.done == 450
    assert time.monotonic() - started < 1.0, "超长行必须仍有界"


def test_parse_progress_only_scans_tail_lines():
    """只回扫尾部若干行: 很久以前的进度不算当前状态。"""
    head = ["Epoch 99/100"]
    noise = ["step done, no progress here"] * (MAX_SCAN_LINES + 10)
    assert parse_progress("\n".join(head + noise), now=0.0) is None


# --------------------------------------------------------------------------- #
# 样本与阶段
# --------------------------------------------------------------------------- #
def test_progress_sample_dict_roundtrip():
    sample = sample_for(TQDM_FULL, at=123.0)
    restored = ProgressSample.from_dict(sample.to_dict())
    assert restored.done == sample.done
    assert restored.total == sample.total
    assert restored.remaining_seconds == sample.remaining_seconds
    assert restored.observed_at == sample.observed_at


def test_progress_text_shapes():
    assert sample_for(TQDM_FULL).progress_text() == "进度 450/1000 (45.0%)"
    # label 保留日志里的原始大小写，展示更自然
    assert sample_for("Epoch 12/500").progress_text() == "Epoch 12/500 (2.4%)"


def test_phase_key_separates_epoch_and_tqdm():
    epoch = sample_for("Epoch 12/500")
    tqdm = sample_for(TQDM_FULL)
    assert epoch.phase_key != tqdm.phase_key, "不同来源的进度不能互相污染"


def test_record_sample_appends_and_caps():
    phases: dict = {}
    for i in range(1, 9):
        phases = add_sample(phases, f"Epoch {i}/100", at=float(i))
    _key, samples = latest_samples(phases)
    assert len(samples) == 5, "样本数有上限"
    assert samples[-1].done == 8


def test_record_sample_resets_on_regression():
    phases = add_sample(None, "Epoch 90/100", at=0.0)
    phases = add_sample(phases, "Epoch 95/100", at=10.0)
    assert len(latest_samples(phases)[1]) == 2

    phases = add_sample(phases, "Epoch 5/100", at=20.0)  # 新一轮
    samples = latest_samples(phases)[1]
    assert len(samples) == 1, "进度回退必须重置，不能用旧数据外推"
    assert samples[0].done == 5


def test_record_sample_keeps_phases_apart():
    phases = add_sample(None, "Epoch 12/500", at=0.0)
    phases = add_sample(phases, TQDM_FULL, at=10.0)
    assert len(phases) == 2
    key, samples = latest_samples(phases)
    assert samples[-1].done == 450, "取最近更新过的阶段"
    assert "1000" in key


# --------------------------------------------------------------------------- #
# 估算
# --------------------------------------------------------------------------- #
def test_estimate_uses_tqdm_remaining_with_high_confidence():
    phases = add_sample(None, TQDM_FULL, at=1000.0)
    result = estimate(phases, now=1010.0)  # 10 秒后才看
    assert result.confidence == "high"
    assert result.remaining_seconds == pytest.approx(115.0, abs=1.0), "扣掉已过去的时间"
    assert "tqdm" in result.basis


def test_estimate_tqdm_note_when_progress_is_stale():
    phases = add_sample(None, TQDM_FULL, at=1000.0)
    result = estimate(phases, now=1000.0 + 4000, stale_after=300)
    assert result.stale_seconds and result.stale_seconds > 300
    assert result.confidence == "low"
    assert "未更新" in result.basis


def test_estimate_extrapolates_from_two_samples():
    phases = add_sample(None, "Epoch 10/100", at=0.0)
    phases = add_sample(phases, "Epoch 40/100", at=600.0)
    result = estimate(phases, now=600.0)
    assert result.confidence == "high"
    # 600 秒走了 30 epoch → 还剩 60 epoch → 1200 秒
    assert result.remaining_seconds == pytest.approx(1200.0, rel=0.02)


def test_estimate_single_sample_uses_average_rate():
    phases = add_sample(None, "Epoch 50/100", at=500.0)
    result = estimate(phases, now=500.0, started_at=0.0)  # 500 秒走了 50 epoch
    assert result.confidence == "medium"
    assert result.remaining_seconds == pytest.approx(500.0, rel=0.05)


def test_estimate_single_sample_without_start_time_is_none():
    phases = add_sample(None, "Epoch 50/100", at=500.0)
    result = estimate(phases, now=500.0)
    assert result.confidence == "none"
    assert not result.known
    assert "样本不足" in result.basis


def test_estimate_falls_back_to_history():
    result = estimate(None, now=1000.0, history_seconds=4500.0, history_runs=3)
    assert result.confidence == "low"
    assert result.remaining_seconds == 4500.0
    assert "历史" in result.basis and "3 次" in result.basis


def test_estimate_without_any_basis_says_so():
    result = estimate(None, now=0.0)
    assert not result.known
    assert result.confidence == "none"
    assert "无法估算" in format_eta(result, now=0.0)


def test_estimate_prefers_progress_over_history():
    phases = add_sample(None, TQDM_FULL, at=1000.0)
    result = estimate(phases, now=1000.0, history_seconds=99_999.0, history_runs=5)
    assert result.remaining_seconds is not None and result.remaining_seconds < 1000
    assert "tqdm" in result.basis


def test_estimate_handles_completed_progress():
    """进度到 100% 且 tqdm 剩余为 0 → 认为马上结束。"""
    phases = add_sample(None, TQDM_DONE, at=600.0)
    result = estimate(phases, now=600.0)
    assert result.known
    assert result.remaining_seconds == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------------- #
# 历史耗时
# --------------------------------------------------------------------------- #
def test_median_history_uses_completed_runs_only():
    events = [
        {"event": "task_finished", "task": "exp1", "status": "completed", "duration_seconds": 3600},
        {"event": "task_finished", "task": "exp1", "status": "failed", "duration_seconds": 30},
        {"event": "task_finished", "task": "exp1", "status": "completed", "duration_seconds": 5400},
        {"event": "task_finished", "task": "other", "status": "completed", "duration_seconds": 99999},
        {"event": "task_started", "task": "exp1"},
        "junk",
    ]
    median, count = median_history(events, "exp1")
    assert count == 2
    assert median == pytest.approx(4500.0)


def test_median_history_returns_none_without_history():
    assert median_history([], "exp1") == (None, 0)
    assert median_history([{"event": "task_finished", "task": "x"}], "exp1") == (None, 0)


def test_median_history_respects_limit():
    events = [
        {"event": "task_finished", "task": "e", "status": "completed", "duration_seconds": float(i)}
        for i in range(1, 51)
    ]
    _median, count = median_history(events, "e", limit=5)
    assert count == 5, "只统计最近 limit 次"


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def test_format_eta_and_progress():
    phases = add_sample(None, TQDM_FULL, at=1000.0)
    text = format_eta(estimate(phases, now=1000.0), now=1000.0)
    assert "预计" in text and "结束" in text and "约" in text

    assert format_progress(phases) == "进度 450/1000 (45.0%)"
    assert format_progress(None) == "-"


def test_estimate_dataclass_known_flag():
    assert EtaEstimate(finish_at=1.0).known
    assert not EtaEstimate().known


# --------------------------------------------------------------------------- #
# DETR 系日志（真实训练脚本的格式）
# --------------------------------------------------------------------------- #
DETR_LINE = "Epoch: [141]  [10/63]  eta: 0:00:53  lr: 0.000040  class_error: 3.88"


def test_parse_detr_epoch_line_without_total_epochs():
    sample = sample_for(DETR_LINE)
    assert sample.label == "epoch" and sample.done == 141
    assert sample.total is None, "不知道总轮数就不能换算整轮进度"
    assert sample.fraction is None
    assert "缺总量" in sample.progress_text()


def test_parse_detr_epoch_line_with_total_epochs():
    sample = parse_progress(DETR_LINE, total_epochs=500)
    assert sample is not None
    assert (sample.done, sample.total) == (141 * 63 + 10, 500 * 63)
    assert sample.fraction == pytest.approx((141 * 63 + 10) / (500 * 63))


def test_detr_inline_eta_is_not_whole_run_remaining():
    """行内的 `eta: 0:00:53` 是**本轮**剩余时间，绝不能当成整轮 ETA。"""
    sample = sample_for(DETR_LINE)
    assert sample.remaining_seconds is None
    result = estimate(record_sample(None, sample), now=0.0, history_seconds=8137, history_runs=1)
    assert result.remaining_seconds == 8137, "应退到历史耗时，而不是谎报 53 秒"
    assert "历史" in result.basis


def test_estimate_with_epoch_units_extrapolates_run_level_eta():
    # 141 轮 + 10/63 → 8883 单位；143 轮 + 40/63 → 9049 单位；相隔 174 秒
    first = parse_progress("Epoch: [141]  [10/63]", total_epochs=500, now=1000.0)
    second = parse_progress("Epoch: [143]  [40/63]", total_epochs=500, now=1174.0)
    assert first is not None and second is not None
    phases = record_sample(None, first)
    phases = record_sample(phases, second)

    result = estimate(phases, now=1174.0)

    assert result.confidence == "high"
    # 166 单位 / 174 秒 ≈ 0.95 单位/秒；剩 (31500-9049) 单位 → 约 6.5 小时
    assert 5.5 * 3600 < (result.remaining_seconds or 0) < 7.5 * 3600


def test_median_history_can_ignore_task_name():
    """任务名带日期后缀（每次都不同）时，用全体已完成任务的耗时兑底。"""
    events = [
        {"event": "task_finished", "task": "exp_a", "status": "completed", "duration_seconds": 8100},
        {"event": "task_finished", "task": "exp_b", "status": "completed", "duration_seconds": 9000},
        {"event": "task_finished", "task": "exp_c", "status": "failed", "duration_seconds": 30},
    ]
    assert median_history(events, "exp_unknown") == (None, 0)
    median, count = median_history(events, None)
    assert count == 2 and median == pytest.approx(8550.0)


def test_estimate_history_label_is_reported_verbatim():
    result = estimate(
        None, now=0.0, history_seconds=3600, history_runs=3, history_label="历史全部任务"
    )
    assert "历史全部任务" in result.basis
