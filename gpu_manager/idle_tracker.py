"""连续空闲确认。

单次 0% 利用率不足以判定空闲（checkpoint 保存、DataLoader 等待、评估切换都可能让
util 瞬时掉到 0），必须连续 N 次检查都满足空闲条件才确认。

状态会持久化，但带「新鲜度」判断: 若上次更新距现在超过 2 个检查周期（例如 daemon
停了很久），计数一律清零重新确认 —— 宁可多等一个周期，也不因陈旧计数误占 GPU。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdleTracker:
    required_checks: int = 3
    counts: dict[int, int] = field(default_factory=dict)
    since: dict[int, float] = field(default_factory=dict)
    updated_at: float = 0.0

    # ------------------------------------------------------------------ #
    def update(
        self,
        idle_ids: Iterable[int],
        monitored_ids: Iterable[int],
        *,
        now: float | None = None,
    ) -> list[int]:
        """推进一轮检查，返回已确认可调度的 GPU（连续空闲次数达标）。"""
        now = now if now is not None else time.time()
        idle = {int(g) for g in idle_ids}
        monitored = {int(g) for g in monitored_ids}

        for gpu_id in list(self.counts):
            if gpu_id not in monitored:
                self.counts.pop(gpu_id, None)
                self.since.pop(gpu_id, None)

        for gpu_id in monitored:
            if gpu_id in idle:
                self.counts[gpu_id] = self.counts.get(gpu_id, 0) + 1
                self.since.setdefault(gpu_id, now)
            else:
                self.counts[gpu_id] = 0
                self.since.pop(gpu_id, None)

        self.updated_at = now
        return sorted(g for g in monitored if self.counts.get(g, 0) >= self.required_checks)

    # ------------------------------------------------------------------ #
    def confirmed(self, monitored_ids: Iterable[int] | None = None) -> list[int]:
        monitored = {int(g) for g in monitored_ids} if monitored_ids is not None else None
        return sorted(
            g
            for g, count in self.counts.items()
            if count >= self.required_checks and (monitored is None or g in monitored)
        )

    def streak_text(self, gpu_id: int) -> str:
        count = self.counts.get(gpu_id, 0)
        if count <= 0:
            return "0"
        started = self.since.get(gpu_id)
        if started is None:
            return f"{count}"
        return f"{count} 次 / 约 {int(time.time() - started)} 秒"

    def progress(self, monitored_ids: Iterable[int]) -> dict[int, tuple[int, bool]]:
        """每张受监控 GPU 的 (连续空闲次数, 是否已确认)，用于 status 展示。"""
        return {
            int(g): (self.counts.get(int(g), 0), self.counts.get(int(g), 0) >= self.required_checks)
            for g in monitored_ids
        }

    def reset(self, gpu_ids: Iterable[int] | None = None) -> None:
        if gpu_ids is None:
            self.counts.clear()
            self.since.clear()
            return
        for g in gpu_ids:
            self.counts[int(g)] = 0
            self.since.pop(int(g), None)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {str(k): int(v) for k, v in self.counts.items()},
            "since": {str(k): float(v) for k, v in self.since.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        required_checks: int,
        *,
        max_age_seconds: float,
        now: float | None = None,
    ) -> IdleTracker:
        now = now if now is not None else time.time()
        tracker = cls(required_checks=required_checks)
        if not isinstance(data, dict):
            return tracker
        updated_at = float(data.get("updated_at") or 0.0)
        if updated_at <= 0 or (now - updated_at) > max_age_seconds:
            # 陈旧计数不可信 → 全部清零，重新确认
            tracker.updated_at = now
            return tracker
        counts_raw = data.get("counts") or {}
        since_raw = data.get("since") or {}
        if isinstance(counts_raw, dict):
            for key, value in counts_raw.items():
                try:
                    tracker.counts[int(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        if isinstance(since_raw, dict):
            for key, value in since_raw.items():
                try:
                    tracker.since[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        tracker.updated_at = updated_at
        return tracker
