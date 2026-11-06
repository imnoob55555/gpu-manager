"""飞书常驻 GPU 面板：一张共享卡片，变化即刷 + 心跳刷。

设计要点
--------
* **纯出站**：只读 :class:`BotContext` → 生成卡片 JSON → 发送 / 原地更新。
  不新增任何入站命令，因此「只有暂停/恢复是写操作」的安全边界完全不变。
* **用卡片而不是文本**：飞书文本消息一条最多可编辑 20 次（硬限制），
  卡片消息用 ``PATCH /im/v1/messages/:message_id`` 原地更新，没有总次数限制
  （单条 5 QPS；只能在发送后 14 天内更新 —— 到期自动另发一张，用户无感）。
* **变化即刷 + 心跳**：红绿灯/计数/暂停/运行任务的签名一变立即刷；
  否则每 ``refresh_seconds`` 刷一次，让「更新时间」和 ETA 不过期。
* **重启不重发**：``message_id`` 落在 daemon 的 state 里，重启后继续更新同一条。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .bot import BotContext, BotTransport, GpuView, TaskView

# --------------------------------------------------------------------------- #
# 红绿灯（emoji 最稳：跨客户端/深色模式都能正确显示）
# --------------------------------------------------------------------------- #
LIGHT_MINE = "🔵"
LIGHT_FREE = "🟢"
LIGHT_COOLING = "🟡"
LIGHT_BUSY = "🔴"

#: 卡片消息体上限（飞书：卡片 ≤30KB）
MAX_CARD_BYTES = 30 * 1024
#: 飞书只允许更新 14 天内发送的卡片；提前一天换新卡，避免刷新突然失效
RECREATE_AFTER_SECONDS = 13 * 24 * 3600
#: 两次更新之间的最小间隔（飞书单条消息 5 QPS，这里远低于上限即可）
MIN_UPDATE_INTERVAL = 5.0
#: 失败退避上限
MAX_BACKOFF_SECONDS = 300.0


@dataclass(frozen=True)
class Light:
    """一张卡的红绿灯状态。"""

    emoji: str
    label: str


def light_for(gpu: GpuView) -> Light:
    """把单卡状态映射成红绿灯。

    优先级：我们自己的任务 > 他人占用 > 自己的其它进程 > 空闲可抢 > 冷却中。
    """
    if gpu.task_name:
        return Light(LIGHT_MINE, f"我的任务 `{gpu.task_name}`")
    if gpu.foreign_pids:
        uid = f"uid{gpu.foreign_uids[0]}" if gpu.foreign_uids else "他人"
        return Light(LIGHT_BUSY, f"{uid} 占用")
    if gpu.own_pids:
        return Light(LIGHT_MINE, "自己的进程")
    if gpu.schedulable:
        return Light(LIGHT_FREE, "空闲·可抢")
    return Light(LIGHT_COOLING, f"冷却 {gpu.idle_streak}/{gpu.idle_required}")


def _mem_text(gpu: GpuView) -> str:
    """紧凑显存显示：``4.2/48G``；读不到时给「读取失败」。"""
    used = gpu.memory_used_mb
    total = gpu.memory_total_mb
    if used is None and total is None:
        return "读取失败"
    used_gb = "?" if used is None else f"{used / 1024:.1f}"
    total_gb = "?" if total is None else f"{total / 1024:.0f}"
    return f"{used_gb}/{total_gb}G"


def _clock_text(now: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(now))


def _header_template(ctx: BotContext, lights: list[Light]) -> str:
    """标题颜色 = 整体状态：灰(暂停) / 红(读不到) / 绿(有可抢) / 橙(全占) 。"""
    if ctx.paused:
        return "grey"
    if not ctx.gpus or ctx.gpu_error:
        return "red"
    if any(light.emoji == LIGHT_FREE for light in lights):
        return "green"
    return "orange"


def _running_md(tasks: list[TaskView]) -> str:
    lines = ["▶ **我的任务**"]
    for task in tasks:
        gpus = ",".join(f"GPU{i}" for i in task.gpu_ids) or "分配中"
        parts = [f"**{task.name}**", gpus]
        if task.runtime_seconds is not None:
            parts.append(f"已 {_short_duration(task.runtime_seconds)}")
        if task.eta_text:
            parts.append(task.eta_text)
        lines.append("• " + " · ".join(parts))
    return "\n".join(lines)


def _short_duration(seconds: float) -> str:
    """``3h12m`` / ``45m`` / ``30s``，比 fmt_duration 更紧凑。"""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes = rest // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{total}s"


def build_card(
    ctx: BotContext,
    *,
    now: float | None = None,
    heartbeat_seconds: float = 120.0,
) -> dict[str, Any]:
    """把一份状态快照渲染成飞书卡片 JSON（纯函数，便于单测）。"""
    now = time.time() if now is None else now
    lights = [light_for(gpu) for gpu in ctx.gpus]
    total = len(ctx.gpus)
    free = sum(1 for light in lights if light.emoji == LIGHT_FREE)
    counts = ctx.counts or {}
    pending = int(counts.get("pending", 0))
    running = int(counts.get("running", 0))

    title = f"🖥 GPU 面板 · {free}/{total} 可抢" if total else "🖥 GPU 面板"
    meta = [f"⏱ **{_clock_text(now)}** 更新"]
    if pending or running:
        meta.append(f"队列 **{pending}** 待执行 / **{running}** 运行中")
    if not pending and not running:
        meta.append("队列空闲")
    if ctx.paused:
        meta.append("⏸ **已暂停**")

    elements: list[dict[str, Any]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": " · ".join(meta)}}
    ]

    if not ctx.gpus:
        reason = ctx.gpu_error or "暂无 GPU 数据"
        elements.append({"tag": "hr"})
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": f"⚠️ 无法读取 GPU 状态: {reason}"}}
        )
    else:
        lines = [
            f"{light.emoji} **GPU{gpu.index}** {_mem_text(gpu)} · {light.label}"
            for gpu, light in zip(ctx.gpus, lights, strict=True)
        ]
        elements.append({"tag": "hr"})
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}})

    running_tasks = [task for task in ctx.tasks if task.status == "running"]
    if running_tasks:
        elements.append({"tag": "hr"})
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": _running_md(running_tasks)}}
        )

    heartbeat = max(1, int(heartbeat_seconds))
    elements.append(
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": (
                        f"{ctx.server or 'GPU Manager'} · 状态变化即刷新 + 每 {heartbeat}s 心跳 · "
                        "长按消息可置顶"
                    ),
                }
            ],
        }
    )

    return {
        # update_multi: 让「共享卡片」可被更新（飞书更新卡片的硬性要求）
        "config": {"wide_screen_mode": True, "update_multi": True},
        "header": {"template": _header_template(ctx, lights), "title": {"tag": "plain_text", "content": title}},
        "elements": elements,
    }


def card_signature(ctx: BotContext) -> str:
    """内容签名：只包含「值得立刻刷新」的字段。

    刻意不含时间戳/ETA/进度 —— 否则每次心跳签名都变，就退化成无脑刷新。
    """
    parts = [
        f"mode={ctx.mode}",
        f"paused={ctx.paused}",
        f"error={ctx.gpu_error}",
        "counts=" + ",".join(f"{k}:{v}" for k, v in sorted((ctx.counts or {}).items())),
        "gpus=" + ",".join(f"{gpu.index}:{light_for(gpu).emoji}" for gpu in ctx.gpus),
        "running="
        + ",".join(
            f"{task.name}:{','.join(str(i) for i in task.gpu_ids)}"
            for task in ctx.tasks
            if task.status == "running"
        ),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _needs_recreate(detail: str) -> bool:
    """更新失败时判断是否该「另发一张新卡」。

    230031 = 超过 14 天；230025 = 内容超长。其余（限频、网络、权限）重试即可。
    """
    for code in ("230031", "230025"):
        if code in detail:
            return True
    lowered = detail.lower()
    return any(word in lowered for word in ("not exist", "withdrawn", "撤回", "不存在"))


class PanelRefresher:
    """后台刷新一张常驻卡片：变化即刷 + 心跳刷 + 失败退避 + 到期换新卡。"""

    def __init__(
        self,
        *,
        provider: Callable[[], BotContext],
        transport: BotTransport,
        receive_id: str,
        receive_id_type: str,
        logger: logging.Logger,
        persist: Callable[[str, float], None] | None = None,
        initial_message_id: str = "",
        initial_sent_at: float = 0.0,
        heartbeat_seconds: float = 120.0,
        poll_seconds: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.transport = transport
        self.receive_id = receive_id
        self.receive_id_type = receive_id_type
        self.logger = logger
        # 由 daemon 注入（在 daemon 的锁内落到 state）；面板线程不直接改共享 dict
        self.persist = persist if persist is not None else (lambda _mid, _ts: None)
        self._message_id = initial_message_id
        self._sent_at = initial_sent_at
        self.heartbeat_seconds = max(30.0, float(heartbeat_seconds))
        self.poll_seconds = max(2.0, float(poll_seconds))
        self.clock = clock
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_signature = ""
        self._last_sent_at = 0.0
        self._failures = 0
        self._next_retry_at = 0.0

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-manager-panel", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def has_card(self) -> bool:
        """是否已有一条可复用的卡片（重启后接着更新，而不是再发一条）。"""
        return bool(self._message_id)

    # ------------------------------------------------------------------ #
    # 刷新
    # ------------------------------------------------------------------ #
    def _remember(self, message_id: str, sent_at: float) -> None:
        """记住「哪张卡片」并交给 daemon 持久化（重启后继续更新同一条）。"""
        self._message_id = message_id
        self._sent_at = sent_at
        try:
            self.persist(message_id, sent_at)
        except Exception:  # 持久化失败不该影响面板刷新
            self.logger.warning("面板状态持久化失败（下次重启会重发一张新卡）", exc_info=True)

    def _run(self) -> None:
        # 稍等片刻，让 daemon 完成第一次 GPU 采集，避免首屏是「暂无数据」
        self._stop.wait(5.0)
        while not self._stop.is_set():
            try:
                if self.clock() >= self._next_retry_at:
                    self.refresh_once()
            except Exception:  # 面板是旁路能力，任何异常都不许拖垮 daemon
                self.logger.exception("常驻面板刷新异常（继续重试）")
            self._stop.wait(self.poll_seconds)

    def refresh_once(self, *, now: float | None = None, force: bool = False) -> tuple[bool, str]:
        """构建并（必要时）发送/更新一次卡片。返回 (是否成功, 说明)。"""
        now = self.clock() if now is None else now
        if not self.receive_id or not self.receive_id_type:
            return False, "panel 未配置收件人（feishu.panel.receive_id 或 allowed_users 为空）"
        ctx = self.provider()
        card = build_card(ctx, now=now, heartbeat_seconds=self.heartbeat_seconds)
        payload = json.dumps(card, ensure_ascii=False)
        size = len(payload.encode("utf-8"))
        if size > MAX_CARD_BYTES:
            return False, f"卡片过大 {size}B > {MAX_CARD_BYTES}B"

        message_id = self._message_id
        sent_at = self._sent_at
        signature = card_signature(ctx)

        expired = bool(message_id and sent_at and now - sent_at > RECREATE_AFTER_SECONDS)
        changed = signature != self._last_signature
        heartbeat_due = bool(message_id and now - self._last_sent_at >= self.heartbeat_seconds)
        if not (force or not message_id or changed or heartbeat_due or expired):
            return True, "未变化"
        if message_id and not expired and now - self._last_sent_at < MIN_UPDATE_INTERVAL:
            return True, "节流"

        if message_id and not expired:
            ok, detail = self.transport.update_card(message_id, card)
            if ok:
                self._on_sent(signature, now)
                if changed:
                    # 心跳更新不写日志（每 120s 一条会吵）；只有「状态变化」值得记一笔
                    self.logger.info("常驻面板已刷新（状态变化）")
                return True, "已更新"
            if not _needs_recreate(detail):
                self._on_failure(now, detail)
                return False, f"更新失败: {detail}"
            self.logger.info("常驻面板需重发新卡（%s）", detail)
            self._message_id = ""
            message_id = ""

        ok, detail = self.transport.send_card(self.receive_id, card, receive_id_type=self.receive_id_type)
        if not ok:
            self._on_failure(now, detail)
            return False, f"发送失败: {detail}"
        self._remember(detail, now)
        self._on_sent(signature, now)
        return True, "已发送"

    def _on_sent(self, signature: str, now: float) -> None:
        self._last_signature = signature
        self._last_sent_at = now
        self._failures = 0
        self._next_retry_at = 0.0

    def _on_failure(self, now: float, detail: str) -> None:
        self._failures += 1
        # 指数退避（10s → 20s → 40s … 上限 5 分钟），避免持续性故障刷屏刷日志
        delay = min(MAX_BACKOFF_SECONDS, self.poll_seconds * (2**self._failures))
        self._next_retry_at = now + delay
        if self._failures == 1:
            self.logger.warning("常驻面板刷新失败（%s），%.0fs 后重试", detail, delay)
        elif self._failures % 12 == 0:
            self.logger.warning("常驻面板连续失败 %d 次（%s），仍每 %.0f 分钟重试", self._failures, detail, delay / 60)
