"""常驻面板：卡片渲染 + 刷新策略（变化即刷 / 心跳 / 14 天换卡 / 失败退避）。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from gpu_manager.bot import BotContext, GpuView, TaskView
from gpu_manager.panel import (
    LIGHT_BUSY,
    LIGHT_COOLING,
    LIGHT_FREE,
    LIGHT_MINE,
    MAX_CARD_BYTES,
    PanelRefresher,
    build_card,
    card_signature,
    light_for,
)


# --------------------------------------------------------------------------- #
# 构造工具
# --------------------------------------------------------------------------- #
def make_gpu(index: int, **kw: Any) -> GpuView:
    base: dict[str, Any] = {
        "index": index,
        "utilization": 0,
        "memory_used_mb": 1024,
        "memory_total_mb": 49140,
        "schedulable": False,
        "idle_streak": 0,
        "idle_required": 3,
    }
    base.update(kw)
    return GpuView(**base)


LOGGER = logging.getLogger("gpu_manager.test_panel")
LOGGER.addHandler(logging.NullHandler())


def make_ctx(**kw: Any) -> BotContext:
    base: dict[str, Any] = {
        "server": "labhost",
        "mode": "QUEUE",
        "counts": {"pending": 2, "running": 0},
        "gpus": [make_gpu(0, schedulable=True), make_gpu(1)],
    }
    base.update(kw)
    return BotContext(**base)


class RecordingTransport:
    """记录卡片发送/更新的假传输层。"""

    def __init__(
        self,
        *,
        send_ok: bool = True,
        update_ok: bool = True,
        update_detail: str = "ok",
    ) -> None:
        self.sent: list[tuple[str, str]] = []
        self.cards: list[tuple[str, str, dict]] = []
        self.updates: list[tuple[str, dict]] = []
        self.send_ok = send_ok
        self.update_ok = update_ok
        self.update_detail = update_detail

    def send_text(self, chat_id: str, text: str, *, reply_to: str | None = None):
        self.sent.append((chat_id, text))
        return True, "ok"

    def send_card(self, receive_id: str, card: dict, *, receive_id_type: str = "open_id"):
        self.cards.append((receive_id, receive_id_type, card))
        if not self.send_ok:
            return False, "send failed"
        return True, f"om_{len(self.cards)}"

    def update_card(self, message_id: str, card: dict):
        self.updates.append((message_id, card))
        return (True, "ok") if self.update_ok else (False, self.update_detail)


def make_refresher(
    transport,
    provider: Callable[[], BotContext] | None = None,
    **kw: Any,
) -> tuple[PanelRefresher, list[tuple[str, float]]]:
    """构造一个面板；返回 (refresher, 持久化记录)。"""
    persisted: list[tuple[str, float]] = []
    refresher = PanelRefresher(
        provider=provider or (lambda: make_ctx()),
        transport=transport,
        receive_id="ou_me",
        receive_id_type="open_id",
        logger=LOGGER,
        persist=lambda mid, ts: persisted.append((mid, ts)),
        **kw,
    )
    return refresher, persisted


# --------------------------------------------------------------------------- #
# 红绿灯映射
# --------------------------------------------------------------------------- #
def test_light_mine_when_task_runs():
    light = light_for(make_gpu(0, task_name="exp1"))
    assert light.emoji == LIGHT_MINE
    assert "exp1" in light.label


def test_light_mine_for_own_process():
    light = light_for(make_gpu(0, own_pids=[123]))
    assert light.emoji == LIGHT_MINE


def test_light_busy_for_foreign_process():
    light = light_for(make_gpu(0, foreign_pids=[9], foreign_uids=[1010]))
    assert light.emoji == LIGHT_BUSY
    assert "uid1010" in light.label


def test_light_free_when_schedulable():
    light = light_for(make_gpu(0, schedulable=True))
    assert light.emoji == LIGHT_FREE
    assert "可抢" in light.label


def test_light_cooling_with_progress():
    light = light_for(make_gpu(0, schedulable=False, idle_streak=2, idle_required=3))
    assert light.emoji == LIGHT_COOLING
    assert "2/3" in light.label


def test_light_task_wins_over_foreign_processes():
    """自己的任务 + 别人残留进程：仍算「我的任务」（这台是我们在用）。"""
    light = light_for(make_gpu(0, task_name="exp1", foreign_pids=[9], foreign_uids=[1010]))
    assert light.emoji == LIGHT_MINE


# --------------------------------------------------------------------------- #
# 卡片渲染
# --------------------------------------------------------------------------- #
def test_card_header_green_when_free_gpu():
    card = build_card(make_ctx(), now=1000.0)
    assert card["header"]["template"] == "green"
    assert "1/2" in card["header"]["title"]["content"]


def test_card_header_grey_when_paused():
    card = build_card(make_ctx(paused=True), now=1000.0)
    assert card["header"]["template"] == "grey"
    assert "已暂停" in json.dumps(card, ensure_ascii=False)


def test_card_header_red_without_gpu_data():
    card = build_card(make_ctx(gpus=[], gpu_error="nvidia-smi 超时"), now=1000.0)
    assert card["header"]["template"] == "red"
    assert "nvidia-smi 超时" in json.dumps(card, ensure_ascii=False)


def test_card_header_orange_when_all_busy():
    busy = [make_gpu(0, foreign_pids=[1], foreign_uids=[1010])]
    card = build_card(make_ctx(gpus=busy), now=1000.0)
    assert card["header"]["template"] == "orange"


def test_card_update_multi_is_set():
    """飞书更新卡片要求 config.update_multi=true，缺了它 PATCH 会失败。"""
    card = build_card(make_ctx(), now=1000.0)
    assert card["config"]["update_multi"] is True


def test_card_lists_every_gpu():
    card = build_card(make_ctx(), now=1000.0)
    body = json.dumps(card, ensure_ascii=False)
    assert "GPU0" in body and "GPU1" in body
    assert LIGHT_FREE in body and LIGHT_COOLING in body


def test_card_shows_running_tasks():
    ctx = make_ctx(
        tasks=[
            TaskView(
                name="exp1",
                status="running",
                gpu_ids=[0, 1],
                runtime_seconds=3600 * 3 + 120,
                eta_text="预计 1h 后结束",
            )
        ]
    )
    body = json.dumps(build_card(ctx, now=1000.0), ensure_ascii=False)
    assert "exp1" in body
    assert "GPU0,GPU1" in body
    assert "3h02m" in body
    assert "预计 1h 后结束" in body


def test_card_hides_finished_tasks():
    ctx = make_ctx(tasks=[TaskView(name="old", status="completed", gpu_ids=[])])
    body = json.dumps(build_card(ctx, now=1000.0), ensure_ascii=False)
    assert "old" not in body


def test_card_size_below_feishu_limit():
    gpus = [make_gpu(i, schedulable=i % 2 == 0) for i in range(8)]
    payload = json.dumps(build_card(make_ctx(gpus=gpus), now=1000.0), ensure_ascii=False)
    assert len(payload.encode("utf-8")) < MAX_CARD_BYTES


def test_card_pure_for_same_inputs():
    """同一时刻 + 同一状态 → 完全相同的卡片（便于签名比较）。"""
    first = build_card(make_ctx(), now=1234.0)
    second = build_card(make_ctx(), now=1234.0)
    assert first == second


# --------------------------------------------------------------------------- #
# 签名（决定「要不要立刻刷」）
# --------------------------------------------------------------------------- #
def test_signature_stable_for_same_state():
    assert card_signature(make_ctx()) == card_signature(make_ctx())


def test_signature_changes_when_light_flips():
    before = make_ctx()
    after = make_ctx(gpus=[make_gpu(0, schedulable=True), make_gpu(1, schedulable=True)])
    assert card_signature(before) != card_signature(after)


def test_signature_ignores_progress_and_eta():
    """ETA/进度变化不该触发额外刷新（否则就退化成无脑刷）。"""
    one = make_ctx(tasks=[TaskView(name="exp1", status="running", gpu_ids=[0], eta_text="1h")])
    two = make_ctx(
        tasks=[TaskView(name="exp1", status="running", gpu_ids=[0], eta_text="2h", stage="x")]
    )
    assert card_signature(one) == card_signature(two)


# --------------------------------------------------------------------------- #
# 刷新策略
# --------------------------------------------------------------------------- #
def test_first_refresh_sends_and_persists():
    transport = RecordingTransport()
    refresher, persisted = make_refresher(transport)
    ok, detail = refresher.refresh_once(now=1000.0)
    assert ok and detail == "已发送"
    assert len(transport.cards) == 1 and not transport.updates
    assert refresher.has_card
    assert persisted == [("om_1", 1000.0)]


def test_unchanged_within_heartbeat_is_skipped():
    transport = RecordingTransport()
    refresher, _ = make_refresher(transport)
    refresher.refresh_once(now=1000.0)
    ok, detail = refresher.refresh_once(now=1010.0)
    assert ok and detail == "未变化"
    assert len(transport.cards) == 1 and not transport.updates


def test_change_triggers_immediate_update():
    transport = RecordingTransport()
    state = {"ctx": make_ctx()}
    refresher, _ = make_refresher(transport, provider=lambda: state["ctx"])
    refresher.refresh_once(now=1000.0)
    state["ctx"] = make_ctx(paused=True)  # 暂停 → 红绿灯/标题变化
    ok, detail = refresher.refresh_once(now=1006.0)
    assert ok and detail == "已更新"
    assert len(transport.updates) == 1


def test_heartbeat_updates_when_nothing_changed():
    transport = RecordingTransport()
    refresher, _ = make_refresher(transport, heartbeat_seconds=120.0)
    refresher.refresh_once(now=1000.0)
    ok, detail = refresher.refresh_once(now=1121.0)
    assert ok and detail == "已更新"
    assert len(transport.updates) == 1


def test_update_is_throttled_by_min_interval():
    """5s 内的多次变化只发一次（飞书单条消息 5 QPS，没必要贴脸刷）。"""
    transport = RecordingTransport()
    state = {"ctx": make_ctx()}
    refresher, _ = make_refresher(transport, provider=lambda: state["ctx"])
    refresher.refresh_once(now=1000.0)
    state["ctx"] = make_ctx(paused=True)
    ok, detail = refresher.refresh_once(now=1002.0)
    assert ok and detail == "节流"
    assert not transport.updates


def test_reuses_existing_card_after_restart():
    """重启后读回 message_id → 更新老卡片，而不是再发一条。"""
    transport = RecordingTransport()
    refresher, _ = make_refresher(
        transport, initial_message_id="om_old", initial_sent_at=time.time()
    )
    ok, _ = refresher.refresh_once(now=time.time() + 5)
    assert ok
    assert not transport.cards
    assert transport.updates and transport.updates[0][0] == "om_old"


def test_update_failure_keeps_card_and_backs_off():
    transport = RecordingTransport(update_ok=False, update_detail="HTTP 400: 限频")
    refresher, _ = make_refresher(transport, initial_message_id="om_old", initial_sent_at=0.0)
    ok, detail = refresher.refresh_once(now=1000.0)
    assert not ok and "更新失败" in detail
    assert not transport.cards, "普通失败不该重发新卡"
    assert refresher._failures == 1
    assert refresher._next_retry_at > 1000.0


def test_expired_card_is_recreated():
    """飞书只允许更新 14 天内的卡片（230031）→ 自动换发新卡。"""
    transport = RecordingTransport(update_ok=False, update_detail='HTTP 400: {"code":230031}')
    refresher, persisted = make_refresher(
        transport, initial_message_id="om_old", initial_sent_at=0.0
    )
    ok, detail = refresher.refresh_once(now=time.time())
    assert ok and detail == "已发送"
    assert len(transport.cards) == 1
    assert persisted and persisted[-1][0] == "om_1"


def test_card_older_than_13_days_is_replaced_even_without_changes():
    transport = RecordingTransport()
    refresher, _ = make_refresher(
        transport, initial_message_id="om_old", initial_sent_at=1000.0 - 13 * 24 * 3600 - 1
    )
    ok, detail = refresher.refresh_once(now=1000.0)
    assert ok and detail == "已发送"
    assert len(transport.cards) == 1


def test_missing_receive_id_fails_cleanly():
    transport = RecordingTransport()
    refresher = PanelRefresher(
        provider=make_ctx,
        transport=transport,
        receive_id="",
        receive_id_type="",
        logger=LOGGER,
        persist=lambda mid, ts: None,
    )
    ok, detail = refresher.refresh_once(now=1000.0)
    assert not ok and "收件人" in detail
    assert not transport.cards


def test_send_failure_is_reported():
    transport = RecordingTransport(send_ok=False)
    refresher, persisted = make_refresher(transport)
    ok, detail = refresher.refresh_once(now=1000.0)
    assert not ok and "发送失败" in detail
    assert not refresher.has_card
    assert persisted == []


def test_persist_failure_does_not_break_refresh():
    transport = RecordingTransport()

    def boom(_mid, _ts):
        raise RuntimeError("disk full")

    refresher = PanelRefresher(
        provider=make_ctx,
        transport=transport,
        receive_id="ou_me",
        receive_id_type="open_id",
        logger=LOGGER,
        persist=boom,
    )
    ok, _ = refresher.refresh_once(now=1000.0)
    assert ok and refresher.has_card


def test_thread_lifecycle_starts_and_stops():
    transport = RecordingTransport()
    refresher, _ = make_refresher(transport, poll_seconds=2.0)
    refresher.start()
    try:
        assert refresher.running
    finally:
        refresher.stop()
    assert not refresher.running
