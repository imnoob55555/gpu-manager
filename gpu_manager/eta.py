"""任务进度解析与结束时间（ETA）估算。

分层退化，**永远标注依据**，不假装精确:

   1. 日志进度解析（最准）
      · tqdm 自带剩余时间:  ``45%|███| 450/1000 [01:23<02:05, 3.6it/s]`` → 直接用 <02:05>
      · ``Epoch 12/500`` / ``step 1000/5000`` → 两次采样算速率后外推
   2. 历史同类任务中位耗时（日志没有进度输出时兜底）
   3. 都没有 → 明确回答「无法估算」，只报已运行时长

只接受**单调递增**的进度样本；进度回退（换 epoch、total 变化、进入新阶段）
会重置该阶段样本，避免用旧数据外推。不同来源（Epoch 行与 tqdm 行交替出现）
按「阶段键 = label|total」分开保存，取最近更新过的那个阶段，互不污染。

本模块是纯逻辑（不读文件、不发网络），因此可完整单测。
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

#: 默认进度模式，按顺序尝试（第一个匹配成功的生效）。
#: 命名组统一用 done/total/remain/elapsed/percent/label，缺哪个就取不到哪个。
#: 专用模式（如 DETR 的 ``Epoch: [141] [10/63]``）必须排在通用模式之前，
#: 否则会被「裸分数」模式误匹配成「本轮进度」，得出「还剩 13 秒」这种误导结论。
#:
#: 所有量词都写**上界**（如 ``\d{1,10}``、``[^|]{0,200}``）而不是 ``+``/``*``：
#: 日志行可能有几万字符（进度条刷屏），无界量词在匹配失败时会回退爆炸。
DEFAULT_PATTERNS: tuple[str, ...] = (
    # DETR/OW-DETR 风格: Epoch: [141]  [10/63]  eta: 0:00:53 ...
    # 注意行内的 eta 是「本 epoch 剩余」，整轮进度需要 total_epochs 参与换算
    r"(?i)\bepoch\s{0,4}:\s{0,4}\[(?P<done>\d{1,6})\]\s{0,4}"
    r"\[\s{0,4}(?P<sub_done>\d{1,6})\s{0,4}/\s{0,4}(?P<sub_total>\d{1,6})",
    # tqdm 完整形状（带百分比与剩余时间）
    r"^\s{0,8}(?P<percent>\d{1,3})%\|[^|]{0,200}\|\s{0,8}(?P<done>\d{1,10})/(?P<total>\d{1,10})"
    r"\s{0,8}\[(?P<elapsed>[\d:]{1,12})<(?P<remain>[\d:?]{1,12})",
    # tqdm 无百分比
    r"\|\s{0,8}(?P<done>\d{1,10})/(?P<total>\d{1,10})\s{0,8}"
    r"\[(?P<elapsed>[\d:]{1,12})<(?P<remain>[\d:?]{1,12})",
    # Epoch 12/500 或 epoch: 12/500
    r"(?i)\b(?P<label>epoch)\s{0,4}[:#]?\s{0,4}(?P<done>\d{1,10})\s{0,4}[/|]\s{0,4}(?P<total>\d{1,10})",
    # step / iter / iteration 1000/5000
    r"(?i)\b(?P<label>step|iter|iteration)\s{0,4}[:#]?\s{0,4}"
    r"(?P<done>\d{1,10})\s{0,4}[/|]\s{0,4}(?P<total>\d{1,10})",
    # 光秃秃的 [12/500] 或 (12/500)
    r"[\[(]\s{0,4}(?P<done>\d{1,10})\s{0,4}/\s{0,4}(?P<total>\d{1,10})\s{0,4}[\])]",
)

#: 样本最多保留几个（用来算速率；太多没必要）
MAX_SAMPLES = 5

#: 每次采样最多回扫多少行、每行最多看多少字符（限制单次匹配的工作量）
MAX_SCAN_LINES = 500
MAX_LINE_CHARS = 2000


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ProgressSample:
    """某一时刻从日志里读到的一个进度点。"""

    done: int | None = None
    total: int | None = None
    percent: float | None = None
    remaining_seconds: float | None = None
    label: str = ""
    source: str = ""
    observed_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #
    @property
    def phase_key(self) -> str:
        """阶段键: 同一「标签+总量」才算同一段进度，避免 Epoch 与 tqdm 互相污染。"""
        label = (self.label or "progress").strip().lower()
        return f"{label}|{self.total if self.total is not None else '?'}"

    @property
    def fraction(self) -> float | None:
        if self.total and self.total > 0 and self.done is not None:
            return min(1.0, max(0.0, self.done / self.total))
        if self.percent is not None:
            return min(1.0, max(0.0, self.percent / 100.0))
        return None

    def progress_text(self) -> str:
        if self.done is not None and self.total:
            head = f"{self.label or '进度'} {self.done}/{self.total}"
            fraction = self.fraction
            return f"{head} ({fraction * 100:.1f}%)" if fraction is not None else head
        if self.done is not None:
            return f"{self.label or '进度'} {self.done}（缺总量，无法换算）"
        if self.percent is not None:
            return f"进度 {self.percent:.0f}%"
        return "有进度输出"

    def to_dict(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "total": self.total,
            "percent": self.percent,
            "remaining_seconds": self.remaining_seconds,
            "label": self.label,
            "source": self.source,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressSample:
        return cls(
            done=data.get("done"),
            total=data.get("total"),
            percent=data.get("percent"),
            remaining_seconds=data.get("remaining_seconds"),
            label=str(data.get("label") or ""),
            source=str(data.get("source") or ""),
            observed_at=float(data.get("observed_at") or 0.0),
        )


@dataclass
class EtaEstimate:
    """估算结果。``confidence``: high / medium / low / none。"""

    finish_at: float | None = None
    remaining_seconds: float | None = None
    confidence: str = "none"
    basis: str = ""
    progress_text: str = ""
    stale_seconds: float | None = None

    @property
    def known(self) -> bool:
        return self.finish_at is not None


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_duration(text: str | None) -> float | None:
    """``02:05`` → 125；``1:02:05`` → 3725；``05`` → 5；``?`` → None。"""
    if not text:
        return None
    parts = str(text).strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    values = [int(p) for p in parts]
    if len(values) == 3:
        return float(values[0] * 3600 + values[1] * 60 + values[2])
    if len(values) == 2:
        return float(values[0] * 60 + values[1])
    return float(values[0])


def _sample_from_match(
    match: re.Match[str],
    patterns_source: str,
    now: float,
    total_epochs: int | None = None,
) -> ProgressSample | None:
    groups = match.groupdict()

    def as_int(name: str) -> int | None:
        raw = groups.get(name)
        if raw is None or not str(raw).isdigit():
            return None
        return int(raw)

    def as_float(name: str) -> float | None:
        raw = groups.get(name)
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    done = as_int("done")
    total = as_int("total")
    percent = as_float("percent")
    label = str(groups.get("label") or "")
    sub_done = as_int("sub_done")
    sub_total = as_int("sub_total")

    # 「Epoch: [141]  [10/63]」型日志：把「第几轮 + 轮内进度」换算成**整轮**进度。
    #
    # 关键: 行内的 `eta: 0:00:53` 只是**本 epoch 的剩余时间**，绝不能当成整轮 ETA
    # （否则会得出「还剩 13 秒」这种误导结论）。没有 total_epochs 时因此只报轮次、
    # 不给 fraction，让上层退到历史耗时。
    if sub_done is not None and sub_total and sub_total > 0:
        label = label or "epoch"
        if total_epochs and total_epochs > 0 and done is not None:
            total = total_epochs * sub_total
            done = done * sub_total + sub_done
        else:
            total = None
            percent = None

    if percent is None and done is not None and total:
        percent = 100.0 * done / total

    # 合理性检查: 进度必须落在 [0, total] 内，否则这一行不是真进度（例如日志里打印配置）
    if (
        done is not None
        and total is not None
        and (total <= 0 or done < 0 or done > total)
    ):
        return None

    if done is None and percent is None:
        return None

    return ProgressSample(
        done=done,
        total=total,
        percent=percent,
        remaining_seconds=parse_duration(groups.get("remain")),
        label=label,
        source=patterns_source,
        observed_at=now,
    )


def _probes(line: str, limit: int = MAX_LINE_CHARS) -> list[str]:
    """超长行的匹配候选：头段与尾段。

    进度数字的位置不固定：tqdm 把 ``500/1000 [01:00<02:00]`` 放在行**尾**，
    ``Epoch 12/500`` 放在行**首**。只截头部会把 tqdm 的进度整个丢掉，
    因此同时提供尾段候选（只对超长行多一两次匹配）。
    """
    if len(line) <= limit:
        return [line]
    return [line[:limit], line[-limit:]]


def parse_progress(
    text: str | list[str],
    patterns: tuple[str, ...] | list[str] | None = None,
    *,
    now: float | None = None,
    total_epochs: int | None = None,
) -> ProgressSample | None:
    """从日志文本里找出**最近**的一次进度。

    从末尾往前扫: 最后出现的进度才代表当前状态（日志中会有大量历史进度行）。
    ``total_epochs`` 用于换算「第几轮 + 轮内进度」型日志（如 DETR 的
    ``Epoch: [141] [10/63]``）—— 不传就只能报轮次，无法给出整轮进度。
    """
    now = now if now is not None else time.time()
    lines = text.splitlines() if isinstance(text, str) else list(text)
    # 只回扫尾部若干行: 进度只可能在末尾，同时限制单次匹配的工作量
    lines = lines[-MAX_SCAN_LINES:]
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for raw in patterns or DEFAULT_PATTERNS:
        try:
            compiled.append((raw, re.compile(raw)))
        except re.error:
            continue  # 用户自定义正则有误 → 跳过，不让它拖垮采样

    for line in reversed(lines):
        if not line:
            continue
        for probe in _probes(line):
            for raw, regex in compiled:
                match = regex.search(probe)
                if match is None:
                    continue
                sample = _sample_from_match(match, raw, now, total_epochs)
                if sample is not None:
                    return sample
    return None


# --------------------------------------------------------------------------- #
# 分阶段保存样本
# --------------------------------------------------------------------------- #
def record_sample(
    phases: dict[str, list[dict[str, Any]]] | None,
    sample: ProgressSample,
    *,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, list[dict[str, Any]]]:
    """把新样本并入阶段表；进度回退或 total 变化时重置该阶段。"""
    out: dict[str, list[dict[str, Any]]] = {
        key: list(items) for key, items in (phases or {}).items() if isinstance(items, list)
    }
    key = sample.phase_key
    history = out.get(key, [])

    last = ProgressSample.from_dict(history[-1]) if history else None
    if (
        last is not None
        and sample.done is not None
        and last.done is not None
        and sample.done < last.done
    ):
        history = []  # 进度回退 → 视为新阶段开始

    history.append(sample.to_dict())
    out[key] = history[-max_samples:]
    return out


def latest_samples(
    phases: dict[str, list[dict[str, Any]]] | None,
) -> tuple[str, list[ProgressSample]]:
    """取最近更新过的阶段及其样本。"""
    best_key = ""
    best_at = -1.0
    best: list[ProgressSample] = []
    for key, items in (phases or {}).items():
        if not isinstance(items, list) or not items:
            continue
        samples = [ProgressSample.from_dict(item) for item in items if isinstance(item, dict)]
        if not samples:
            continue
        newest = max(s.observed_at for s in samples)
        if newest > best_at:
            best_key, best_at, best = key, newest, samples
    return best_key, best


# --------------------------------------------------------------------------- #
# 估算
# --------------------------------------------------------------------------- #
def estimate(
    phases: dict[str, list[dict[str, Any]]] | None,
    *,
    now: float | None = None,
    history_seconds: float | None = None,
    history_runs: int = 0,
    started_at: float | None = None,
    stale_after: float = 300.0,
    history_label: str = "历史同类任务",
) -> EtaEstimate:
    """综合进度样本与历史耗时给出 ETA。

    ``history_label`` 决定依据文案（例如同名任务无记录时调用方会传
    「历史全部任务」），保证回复里说明的估算依据真实可核查。
    """
    now = now if now is not None else time.time()
    _key, samples = latest_samples(phases)

    if samples:
        newest = max(samples, key=lambda s: s.observed_at)
        stale = max(0.0, now - newest.observed_at)
        progress_text = newest.progress_text()

        # --- 1a) tqdm 直接给了剩余时间 ---
        if newest.remaining_seconds is not None and newest.fraction != 0:
            elapsed_since = max(0.0, now - newest.observed_at)
            remaining = max(0.0, newest.remaining_seconds - elapsed_since)
            note = ""
            if stale > stale_after:
                note = f"；注意进度已 {_human_duration(stale)} 未更新"
            return EtaEstimate(
                finish_at=now + remaining,
                remaining_seconds=remaining,
                confidence="high" if stale <= stale_after else "low",
                basis=f"日志 tqdm 剩余时间 {_human_duration(newest.remaining_seconds)}{note}",
                progress_text=progress_text,
                stale_seconds=stale,
            )

        # --- 1b) 两次以上采样 → 算速率外推 ---
        fraction = newest.fraction
        if fraction is not None and newest.total:
            if len(samples) >= 2:
                oldest = min(samples, key=lambda s: s.observed_at)
                if (
                    oldest.done is not None
                    and newest.done is not None
                    and newest.done > oldest.done
                    and newest.observed_at > oldest.observed_at
                ):
                    rate = (newest.done - oldest.done) / (newest.observed_at - oldest.observed_at)
                    if rate > 0:
                        remaining = (newest.total - newest.done) / rate
                        return EtaEstimate(
                            finish_at=now + remaining,
                            remaining_seconds=remaining,
                            confidence="high" if stale <= stale_after else "medium",
                            basis=(
                                f"日志进度 {newest.done}/{newest.total}"
                                f"（{rate * 60:.1f}/分钟外推）"
                            ),
                            progress_text=progress_text,
                            stale_seconds=stale,
                        )

            # --- 1c) 只有一个采样点，但知道任务开始时间 → 用平均速率 ---
            # 注意用 `is not None` 判时间戳: 0.0 是合法时间戳，
            # 用真值判断会把它当成「未知」（这类隐患已用测试锁定）。
            if (
                newest.done
                and started_at is not None
                and newest.observed_at > started_at
            ):
                rate = newest.done / (newest.observed_at - started_at)
                if rate > 0:
                    remaining = (newest.total - newest.done) / rate
                    return EtaEstimate(
                        finish_at=now + remaining,
                        remaining_seconds=remaining,
                        confidence="medium",
                        basis=f"日志进度 {newest.done}/{newest.total}，按平均速率外推（样本较少）",
                        progress_text=progress_text,
                        stale_seconds=stale,
                    )

    # --- 2) 历史同类任务中位耗时 ---
    if history_seconds and history_seconds > 0:
        return EtaEstimate(
            finish_at=now + history_seconds,
            remaining_seconds=history_seconds,
            confidence="low",
            basis=f"按{history_label}中位耗时 {_human_duration(history_seconds)} 估算"
            + (f"（{history_runs} 次）" if history_runs else "")
            + ("；当前无可用进度输出" if not samples else ""),
            progress_text=samples[-1].progress_text() if samples else "",
        )

    # --- 3) 什么依据都没有 ---
    if samples:
        return EtaEstimate(
            confidence="none",
            basis="有进度输出但样本不足（需要至少两次采样，或 tqdm 的剩余时间）",
            progress_text=samples[-1].progress_text(),
            stale_seconds=max(0.0, now - samples[-1].observed_at),
        )
    return EtaEstimate(confidence="none", basis="无进度输出，无法估算结束时间")


def median_history(
    events: list[dict[str, Any]],
    task_name: str | None = None,
    *,
    limit: int = 20,
) -> tuple[float | None, int]:
    """从事件流里算「已完成」运行的中位耗时。

    ``task_name=None`` 表示**不限任务名** —— 用于「同名任务还没跑过」时的兑底
    （例如任务名带日期/变体后缀，每次都不同）。
    """
    durations: list[float] = []
    for event in reversed(events):  # 事件是追加写的，从新的往回找
        if not isinstance(event, dict):
            continue
        if event.get("event") != "task_finished":
            continue
        if task_name is not None and event.get("task") != task_name:
            continue
        if event.get("status") != "completed":
            continue
        value = event.get("duration_seconds")
        if isinstance(value, (int, float)) and value > 0:
            durations.append(float(value))
        if len(durations) >= limit:
            break
    if not durations:
        return None, 0
    return statistics.median(durations), len(durations)


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def _human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def format_eta(estimate_result: EtaEstimate, *, now: float | None = None) -> str:
    """给聊天回复用的一行 ETA 文本。"""
    if not estimate_result.known:
        return f"结束时间: 无法估算（{estimate_result.basis}）"
    now = now if now is not None else time.time()
    finish = time.strftime("%m-%d %H:%M", time.localtime(estimate_result.finish_at or now))
    remaining = _human_duration(estimate_result.remaining_seconds)
    return f"预计 {finish} 结束（约 {remaining} 后 · {estimate_result.basis}）"


def format_progress(phases: dict[str, list[dict[str, Any]]] | None) -> str:
    _key, samples = latest_samples(phases)
    if not samples:
        return "-"
    return samples[-1].progress_text()


__all__ = [
    "DEFAULT_PATTERNS",
    "MAX_LINE_CHARS",
    "MAX_SAMPLES",
    "MAX_SCAN_LINES",
    "EtaEstimate",
    "ProgressSample",
    "estimate",
    "format_eta",
    "format_progress",
    "latest_samples",
    "median_history",
    "parse_duration",
    "parse_progress",
    "record_sample",
]
