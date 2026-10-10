"""飞书交互式查询：你在飞书里发消息 → 机器人回复 GPU 状态 / 任务 / ETA。

为什么必须用自建应用 + 长连接
    自定义群机器人 Webhook 是**单向**的（只能发不能收）。要接收消息只能用
    自建应用订阅 ``im.message.receive_v1`` 事件。事件投递有两条路:
      · HTTP 回调 —— 需要公网可达的 HTTPS 地址（实验室服务器在 NAT 后，不可用）
      · **长连接（WebSocket）** —— 由服务器主动外连飞书，NAT 后可用 ✅

命令集（固定，绝无 shell 透传）:
    ``状态`` / ``status`` / ``gpu``      每张卡的 util/显存/占用者/可调度性 + 模式
    ``队列`` / ``tasks`` / ``list``      任务列表（进度 + 预计结束）
    ``预计`` / ``eta``                   运行中任务的预计结束 + 每张卡预计空出时间
    ``日志 <任务名> [行数]``              该任务日志尾部（默认 20 行，最多 100）
    ``帮助`` / ``help`` / ``?``          命令表

硬性安全边界（都有测试锁定）:
  · **只读**：没有任何写操作，发错命令也不会影响训练
  · **open_id 白名单 fail-closed**：未配置 = 拒绝所有人；未授权消息不回内容，
    只记日志（并打印对方 open_id，方便你加进白名单）
  · 每用户限流（默认 10 条/分钟），并对 message_id 去重（飞书事件至少投递一次）
  · 入站指令全量审计到 events.jsonl
  · 消息长度截断，避免刷屏

本模块的所有判断逻辑（路由/白名单/限流/格式化）都是纯函数或纯对象，
不依赖网络与文件系统，因此可以完整单测；真实网络收发只在 FeishuTransport /
FeishuBot 里，且可用假实现替换。

线程模型（为什么不会与调度互相干扰）:
    · 主线程: daemon 主循环，**拥有**全部状态，写入均在 ``Daemon.lock`` 保护下
    · 长连接线程: lark-oapi 的 WebSocket 循环；消息回调在**该线程**里执行
    · 共享状态的同步方式:
        - ``RateLimiter`` / ``MessageDeduper`` 各自带锁
        - 回调通过 daemon 注入的 ``context_provider()`` 取一份**快照**，
          而该函数在 daemon 锁内构建 —— 因此 router 只会看到一致的状态
        - **网络 I/O（回复飞书）在 daemon 锁外发生**，
          飞书接口变慢也不会卡住调度主循环
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .utils import truncate

#: 回复文本上限（飞书单条文本上限远大于此，这里只为可读性）
MAX_REPLY_CHARS = 1800
#: ``日志`` 命令允许的最大行数
MAX_LOG_LINES = 100
#: 命令别名表: 归一化后的命令名 → 规范命令
COMMAND_ALIASES: dict[str, str] = {
    "状态": "status", "gpu": "status", "status": "status", "显卡": "status", "空闲": "status",
    "队列": "tasks", "任务": "tasks", "list": "tasks", "tasks": "tasks", "queue": "tasks",
    "预计": "eta", "eta": "eta", "何时": "eta", "完成时间": "eta", "结束时间": "eta",
    "日志": "log", "log": "log", "logs": "log", "输出": "log",
    "帮助": "help", "help": "help", "?": "help", "？": "help", "命令": "help",
}

_MENTION_RE = re.compile(r"@_(?:user_\d+|all|online_user_\d+)\b")


# --------------------------------------------------------------------------- #
# 上下文（由 daemon 在锁内组装，router 只读）
# --------------------------------------------------------------------------- #
@dataclass
class GpuView:
    """给聊天展示用的单卡信息。"""

    index: int
    utilization: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    schedulable: bool = False
    idle_streak: int = 0
    idle_required: int = 3
    own_pids: list[int] = field(default_factory=list)
    foreign_pids: list[int] = field(default_factory=list)
    foreign_uids: list[int] = field(default_factory=list)
    #: 正在这张卡上跑的「我们的任务名」（若有）
    task_name: str = ""

    def usage_text(self) -> str:
        if self.utilization is None and self.memory_used_mb is None:
            return "读取失败"
        util = "?" if self.utilization is None else f"{self.utilization}%"
        mem = (
            "?"
            if self.memory_used_mb is None
            else f"{self.memory_used_mb}/{self.memory_total_mb or '?'}MB"
        )
        return f"{util} {mem}"

    def owner_text(self) -> str:
        if self.task_name:
            return f"本机任务 `{self.task_name}`"
        if self.own_pids:
            return f"自己 pid {self.own_pids[0]}"
        if self.foreign_pids:
            uid = self.foreign_uids[0] if self.foreign_uids else "?"
            return f"他人 uid{uid} pid {self.foreign_pids[0]}"
        return "空闲"


@dataclass
class TaskView:
    """给聊天展示用的任务信息。"""

    name: str
    status: str
    gpu_ids: list[int] = field(default_factory=list)
    gpu_num: int = 1
    priority: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    runtime_seconds: float | None = None
    wait_seconds: float | None = None
    progress_text: str = ""
    eta_text: str = ""
    eta_confidence: str = "none"
    error: str = ""
    waiting_reason: str = ""
    #: 当前阶段（来自任务的 stage_file，如 training / evaluating）
    stage: str = ""


@dataclass
class BotContext:
    """一次查询所需的全部状态快照。"""

    server: str = ""
    mode: str = "IDLE"
    counts: dict[str, int] = field(default_factory=dict)
    gpus: list[GpuView] = field(default_factory=list)
    tasks: list[TaskView] = field(default_factory=list)
    daemon_pid: int | None = None
    uptime_seconds: float = 0.0
    gpu_error: str = ""
    #: 读任务日志尾部（由 daemon 注入，router 自己不碰文件系统）
    log_reader: Callable[[str, int], list[str] | None] | None = None


@dataclass
class BotRequest:
    """归一化后的入站消息。"""

    text: str
    user_id: str
    chat_id: str = ""
    message_id: str = ""
    chat_type: str = "p2p"


# --------------------------------------------------------------------------- #
# 安全: 白名单与限流
# --------------------------------------------------------------------------- #
class AllowList:
    """open_id 白名单，**fail-closed**：没配置就拒绝所有人。"""

    def __init__(self, allowed: Iterable[str] = ()) -> None:
        self.allowed = {str(x).strip() for x in allowed if str(x).strip()}

    @property
    def configured(self) -> bool:
        return bool(self.allowed)

    def replace(self, allowed: Iterable[str]) -> None:
        """原地更新白名单（reload 时用）。

        为什么不在 reload 时重建长连接: lark-oapi 的 ws.Client **没有公开的
        stop/close**，重建会留下旧连接继续消费事件（并被旧白名单拒绝）。
        白名单/限流这类“消息处理时才读”的设置原地更新即可生效。
        """
        self.allowed = {str(x).strip() for x in allowed if str(x).strip()}

    def is_allowed(self, user_id: str) -> bool:
        return bool(user_id) and user_id in self.allowed


class RateLimiter:
    """每用户滑动窗口限流。"""

    def __init__(self, per_minute: int = 10, window_seconds: float = 60.0) -> None:
        self.per_minute = max(1, int(per_minute))
        self.window = float(window_seconds)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, user_id: str, now: float | None = None) -> tuple[bool, float]:
        """返回 (是否放行, 需要等待的秒数)。"""
        now = now if now is not None else time.time()
        with self._lock:
            hits = self._hits.setdefault(user_id, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.per_minute:
                wait = max(0.0, self.window - (now - hits[0]))
                return False, wait
            hits.append(now)
            return True, 0.0


class MessageDeduper:
    """飞书事件是「至少投递一次」，用 message_id 去重避免重复回复。"""

    def __init__(self, capacity: int = 500) -> None:
        self.capacity = max(16, int(capacity))
        self._seen: deque[str] = deque(maxlen=self.capacity)
        self._set: set[str] = set()
        self._lock = threading.Lock()

    def first_time(self, message_id: str) -> bool:
        if not message_id:
            return True
        with self._lock:
            if message_id in self._set:
                return False
            if len(self._seen) == self._seen.maxlen and self._seen:
                oldest = self._seen[0]
                self._set.discard(oldest)
            self._seen.append(message_id)
            self._set.add(message_id)
            return True


# --------------------------------------------------------------------------- #
# 命令路由（纯逻辑）
# --------------------------------------------------------------------------- #
HELP_TEXT = """\
可用命令（只读，发错也不会影响训练）:
  状态        GPU 利用率/显存/占用者/可调度性
  队列        任务列表（进度与预计结束）
  预计        运行中任务的预计结束 + 各卡预计空出时间
  日志 <任务名> [行数]   任务日志尾部（默认 20 行）
  帮助        显示本说明"""


class CommandRouter:
    """把一条消息变成一段回复文本。纯逻辑，不读文件、不发网络。"""

    def __init__(self, *, log_lines_default: int = 20) -> None:
        self.log_lines_default = max(1, min(MAX_LOG_LINES, int(log_lines_default)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(text: str) -> tuple[str, list[str]]:
        """去掉 @提及、切出命令与参数。"""
        cleaned = _MENTION_RE.sub(" ", str(text or "")).strip()
        parts = cleaned.split()
        if not parts:
            return "", []
        command = COMMAND_ALIASES.get(parts[0].lower(), COMMAND_ALIASES.get(parts[0], ""))
        return command, parts[1:]

    def handle(self, text: str, ctx: BotContext) -> str:
        """返回回复文本；无法识别时仍给帮助（而不是沉默）。"""
        command, args = self.normalize(text)
        if command == "status":
            return self.render_status(ctx)
        if command == "tasks":
            return self.render_tasks(ctx)
        if command == "eta":
            return self.render_eta(ctx)
        if command == "log":
            return self.render_log(ctx, args)
        if command == "help":
            return HELP_TEXT
        if not command:
            return "没看懂这条消息。\n\n" + HELP_TEXT
        return f"未知命令 `{command}`。\n\n" + HELP_TEXT

    # ------------------------------------------------------------------ #
    def render_status(self, ctx: BotContext) -> str:
        lines = [
            f"GPU 状态 · {ctx.server} · {time.strftime('%m-%d %H:%M')}",
            f"模式 {ctx.mode} · 队列 pending={ctx.counts.get('pending', 0)}"
            f" running={ctx.counts.get('running', 0)}",
        ]
        if ctx.gpu_error and not ctx.gpus:
            lines.append(f"⚠️ nvidia-smi 读取失败: {truncate(ctx.gpu_error, 200)}")
            return self._finish(lines)

        if ctx.gpus:
            lines.append("")
            lines.append("卡  util 显存              可调度  空闲确认  占用")
            for gpu in sorted(ctx.gpus, key=lambda g: g.index):
                mark = "是" if gpu.schedulable else "否"
                idle = f"{gpu.idle_streak}/{gpu.idle_required}"
                lines.append(
                    f"{gpu.index:>2}  {gpu.usage_text():<18} {mark:<6}  {idle:<8}  {gpu.owner_text()}"
                )

        idle_gpus = [g.index for g in ctx.gpus if g.idle_streak >= g.idle_required]
        lines += [
            "",
            "确认空闲: " + (", ".join(f"GPU{i}" for i in idle_gpus) if idle_gpus else "暂无（需连续确认）"),
        ]
        running = [t for t in ctx.tasks if t.status == "running"]
        if running:
            lines.append("本机运行中: " + ", ".join(f"{t.name}→GPU{t.gpu_ids or '?'}" for t in running))
        return self._finish(lines)

    def render_tasks(self, ctx: BotContext) -> str:
        if not ctx.tasks:
            return self._finish([f"任务 · {ctx.server}", "", "（队列为空）"])

        order = {"running": 0, "pending": 1, "failed": 2, "cancelled": 3, "completed": 4}
        tasks = sorted(ctx.tasks, key=lambda t: (order.get(t.status, 9), -t.priority, t.name))
        lines = [f"任务 · {ctx.server} · {time.strftime('%m-%d %H:%M')}"]
        for task in tasks[:15]:
            icon = {"running": "▶", "pending": "◻", "completed": "✔", "failed": "✘", "cancelled": "⊘"}.get(
                task.status, "?"
            )
            head = f"{icon} {task.name} [{task.status}]"
            detail = []
            if task.gpu_ids:
                detail.append("GPU " + ",".join(str(g) for g in task.gpu_ids))
            elif task.status == "pending":
                detail.append(f"需要 {task.gpu_num} 卡")
            if task.status == "running":
                detail.append(f"已跑 {_human(task.runtime_seconds)}")
                if task.stage:
                    detail.append(f"阶段 {task.stage}")
                if task.progress_text:
                    detail.append(task.progress_text)
            elif task.status == "pending":
                detail.append(f"等待 {_human(task.wait_seconds)}")
                if task.waiting_reason:
                    detail.append(task.waiting_reason)
            elif task.status in ("failed", "cancelled") and task.error:
                detail.append(truncate(task.error, 60))
            lines.append(f"{head}  {' · '.join(detail)}")
            if task.eta_text and task.status == "running":
                lines.append(f"    {task.eta_text}")
        if len(tasks) > 15:
            lines.append(f"...另有 {len(tasks) - 15} 个任务（用 `gpu-manager list` 看全部）")
        return self._finish(lines)

    def render_eta(self, ctx: BotContext) -> str:
        running = [t for t in ctx.tasks if t.status == "running"]
        pending = [t for t in ctx.tasks if t.status == "pending"]
        lines = [f"预计结束时间 · {ctx.server} · {time.strftime('%m-%d %H:%M')}", ""]

        if not running:
            lines.append("当前没有运行中的任务。")
        for task in running:
            whose = "GPU " + ",".join(str(g) for g in task.gpu_ids) if task.gpu_ids else "GPU ?"
            stage = f" · 阶段 {task.stage}" if task.stage else ""
            lines.append(f"▶ {task.name} ({whose}){stage}")
            if task.progress_text:
                lines.append(f"    进度: {task.progress_text}")
            lines.append(f"    {task.eta_text or '结束时间: 无法估算'}")
        if pending:
            lines += ["", f"待执行 {len(pending)} 个: " + ", ".join(t.name for t in pending[:5])]

        # 每张卡的预计空出时间
        lines += ["", "各卡预计空出:"]
        for gpu in sorted(ctx.gpus, key=lambda g: g.index):
            holder = next((t for t in running if gpu.index in t.gpu_ids), None)
            if holder is not None:
                short = holder.eta_text.split("（")[0] if holder.eta_text else "无法估算"
                lines.append(f"  GPU{gpu.index}: 等 `{holder.name}` 结束 · {short}")
            elif gpu.own_pids or gpu.foreign_pids:
                owner = (
                    "自己"
                    if gpu.own_pids
                    else f"他人 uid{gpu.foreign_uids[0] if gpu.foreign_uids else '?'}"
                )
                lines.append(f"  GPU{gpu.index}: {owner} 占用 · 无法估算（不是本工具启动的任务）")
            else:
                lines.append(f"  GPU{gpu.index}: 已空闲")
        return self._finish(lines)

    def render_log(self, ctx: BotContext, args: list[str]) -> str:
        if not args:
            return self._finish(["用法: `日志 <任务名> [行数]`", "", *self._task_name_hints(ctx)])
        name = args[0]
        lines_wanted = self.log_lines_default
        if len(args) > 1 and args[1].isdigit():
            lines_wanted = max(1, min(MAX_LOG_LINES, int(args[1])))

        if ctx.log_reader is None:
            return self._finish(["日志读取不可用（daemon 未提供读取器）"])
        tail = ctx.log_reader(name, lines_wanted)
        if tail is None:
            return self._finish([f"找不到任务 `{name}` 的日志。", "", *self._task_name_hints(ctx)])
        if not tail:
            return self._finish([f"任务 `{name}` 还没有日志输出。"])
        body = "\n".join(tail)
        return self._finish(
            [f"`{name}` 最后 {len(tail)} 行:", "", truncate(body, MAX_REPLY_CHARS - 200)]
        )

    @staticmethod
    def _task_name_hints(ctx: BotContext) -> list[str]:
        names = [t.name for t in ctx.tasks][:5]
        return ["可用任务名: " + ", ".join(names)] if names else ["当前没有任务。"]

    @staticmethod
    def _finish(lines: list[str]) -> str:
        text = "\n".join(line for line in lines if line is not None).rstrip()
        if len(text) > MAX_REPLY_CHARS:
            text = text[: MAX_REPLY_CHARS - 20] + "\n...（已截断）"
        return text


def _human(seconds: float | None) -> str:
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


# --------------------------------------------------------------------------- #
# 传输层
# --------------------------------------------------------------------------- #
class BotTransport(Protocol):
    def send_text(
        self, chat_id: str, text: str, *, reply_to: str | None = None
    ) -> tuple[bool, str]: ...


class FeishuTransport:
    """真实发送: 优先用「回复消息」接口（话题聚合），失败则退化为普通发消息。

    复用 notifier 里已验证的 token 获取与出站 URL 白名单（防 SSRF）。
    """

    BASE = "https://open.feishu.cn/open-apis"

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        logger: logging.Logger,
        *,
        timeout: float = 10.0,
        allowed_hosts: Iterable[str] | None = None,
        allow_any_host: bool = False,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.logger = logger
        self.timeout = timeout
        self.allowed_hosts = list(allowed_hosts or ())
        self.allow_any_host = allow_any_host
        self._token = ""
        self._token_expire_at = 0.0

    def _tenant_token(self) -> tuple[bool, str]:
        from .notifier import _http_post_json  # 复用已验证的出站实现

        # 快速失败：凭证不全时**不发任何请求**（既不浪费配额，也让行为可离线验证）
        if not self.app_id or not self.app_secret:
            return False, "feishu.app_id/app_secret 未配置"
        if self._token and time.time() < self._token_expire_at - 60:
            return True, self._token
        ok, data = _http_post_json(
            f"{self.BASE}/auth/v3/tenant_access_token/internal",
            {"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=self.timeout,
            allowed_hosts=self.allowed_hosts,
            allow_any_host=self.allow_any_host,
        )
        if not ok:
            return False, str(data)
        token = str(data.get("tenant_access_token") or "")
        if not token:
            return False, f"获取 token 失败: {data}"
        self._token = token
        self._token_expire_at = time.time() + int(data.get("expire", 7200))
        return True, token

    def send_text(
        self, chat_id: str, text: str, *, reply_to: str | None = None
    ) -> tuple[bool, str]:
        from .notifier import _http_post_json

        ok, token = self._tenant_token()
        if not ok:
            return False, token
        headers = {"Authorization": f"Bearer {token}"}
        content = json.dumps({"text": text}, ensure_ascii=False)

        if reply_to:
            ok, data = _http_post_json(
                f"{self.BASE}/im/v1/messages/{reply_to}/reply",
                {"msg_type": "text", "content": content},
                timeout=self.timeout,
                headers=headers,
                allowed_hosts=self.allowed_hosts,
                allow_any_host=self.allow_any_host,
            )
            if ok and data.get("code") in (0, None):
                return True, "ok"
            self.logger.debug("回复接口失败，退化为普通发送: %s", data)
            if not chat_id:
                return False, f"回复失败: {data}"

        ok, data = _http_post_json(
            f"{self.BASE}/im/v1/messages?receive_id_type=chat_id",
            {"receive_id": chat_id, "msg_type": "text", "content": content},
            timeout=self.timeout,
            headers=headers,
            allowed_hosts=self.allowed_hosts,
            allow_any_host=self.allow_any_host,
        )
        if not ok:
            return False, str(data)
        if data.get("code") in (0, None):
            return True, "ok"
        return False, f"飞书返回 code={data.get('code')} msg={data.get('msg')}"


def _event_to_dict(data: Any) -> dict[str, Any] | None:
    """把长连接回调对象（或普通 dict）归一到 ``{"message": {...}, "sender": {...}}``。

    不依赖 lark-oapi 的内部结构（如 ``lark.JSON.marshal``）:
    不同版本传进来的可能是 dict，也可能是带 ``.event`` 的对象，
    这里两种都按 duck typing 兼容，避免后续升级 SDK 时静默失效。
    """
    if isinstance(data, dict):
        payload = data.get("event", data)
        return payload if isinstance(payload, dict) else None

    event = getattr(data, "event", None)
    if event is None:
        return None
    message = getattr(event, "message", None)
    if message is None:
        return None
    sender = getattr(event, "sender", None)
    sender_id = getattr(sender, "sender_id", None)
    return {
        "message": {
            "message_type": getattr(message, "message_type", None),
            "content": getattr(message, "content", None),
            "chat_id": getattr(message, "chat_id", None),
            "message_id": getattr(message, "message_id", None),
            "chat_type": getattr(message, "chat_type", None),
        },
        "sender": {"sender_id": {"open_id": getattr(sender_id, "open_id", None)}},
    }


# --------------------------------------------------------------------------- #
# 长连接客户端
# --------------------------------------------------------------------------- #
class FeishuBot:
    """在后台线程跑飞书长连接，收到消息后交给 router 回复。"""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        router: CommandRouter,
        context_provider: Callable[[], BotContext],
        transport: BotTransport,
        allowlist: AllowList,
        rate_limiter: RateLimiter | None = None,
        logger: logging.Logger,
        audit: Callable[[dict[str, Any]], None] | None = None,
        deduper: MessageDeduper | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.router = router
        self.context_provider = context_provider
        self.transport = transport
        self.allowlist = allowlist
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = logger
        self.audit = audit
        self.deduper = deduper or MessageDeduper()
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # 事件处理（可单测: 直接调 handle_text）
    # ------------------------------------------------------------------ #
    def handle_text(
        self, request: BotRequest, now: float | None = None
    ) -> tuple[bool, str]:
        """处理一条消息并回复。返回 (是否回复成功, 说明)。

        顺序很重要: **先去重/鉴权/限流，再去构建上下文**（未授权请求不读取任何状态）。
        """
        now = now if now is not None else time.time()
        if self._stop.is_set():
            # 已被替换/停止的实例绝不再处理事件（SDK 无法真正停掉旧连接，
            # 因此用这个标志保证“只有活着的实例会应答”）。
            return False, "stopped"
        if not self.deduper.first_time(request.message_id):
            return False, "duplicate"

        if not self.allowlist.is_allowed(request.user_id):
            # fail-closed: 不回内容，但把 open_id 写进日志方便加入白名单
            self.logger.warning(
                "收到未授权消息（已忽略）。user_id=%s chat=%s text=%r  如需授权请把该 open_id 填入 "
                "config.yaml 的 feishu.allowed_users",
                request.user_id or "(未知)",
                request.chat_id or "-",
                truncate(request.text, 100),
            )
            self._audit(
                {"event": "bot_denied", "user_id": request.user_id, "chat_id": request.chat_id,
                 "text": truncate(request.text, 200)}
            )
            return False, "unauthorized"

        allowed, wait = self.rate_limiter.check(request.user_id, now)
        if not allowed:
            self._audit({"event": "bot_rate_limited", "user_id": request.user_id})
            reply = f"请求太频繁，请 {int(wait) + 1} 秒后再试。"
            return self.transport.send_text(request.chat_id, reply, reply_to=request.message_id)

        try:
            context = self.context_provider()
            reply = self.router.handle(request.text, context)
        except Exception as exc:  # 任何内部错误都不能让长连接线程死掉
            self.logger.exception("处理飞书命令失败: %s", exc)
            reply = f"内部错误: {type(exc).__name__}（详见 daemon 日志）"

        ok, detail = self.transport.send_text(request.chat_id, reply, reply_to=request.message_id)
        self._audit(
            {
                "event": "bot_command",
                "user_id": request.user_id,
                "chat_id": request.chat_id,
                "text": truncate(request.text, 200),
                "reply_ok": ok,
                "reply_chars": len(reply),
            }
        )
        if not ok:
            self.logger.warning("回复飞书消息失败: %s", detail)
        return ok, detail

    def _audit(self, record: dict[str, Any]) -> None:
        if self.audit is None:
            return
        with contextlib.suppress(Exception):  # 审计失败不影响回复
            self.audit(record)

    # ------------------------------------------------------------------ #
    # 事件 → BotRequest
    # ------------------------------------------------------------------ #
    @staticmethod
    def request_from_event(event: dict[str, Any]) -> BotRequest | None:
        """把飞书 ``im.message.receive_v1`` 事件归一化为 BotRequest。"""
        try:
            message = event["message"]
            sender = event["sender"]
        except (KeyError, TypeError):
            return None
        if str(message.get("message_type") or "") != "text":
            return None
        text = ""
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            content = json.loads(message.get("content") or "{}")
            text = str(content.get("text") or "")
        if not text:
            return None
        sender_id = sender.get("sender_id") or {}
        return BotRequest(
            text=text,
            user_id=str(sender_id.get("open_id") or ""),
            chat_id=str(message.get("chat_id") or ""),
            message_id=str(message.get("message_id") or ""),
            chat_type=str(message.get("chat_type") or "p2p"),
        )

    # ------------------------------------------------------------------ #
    # 长连接生命周期
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """启动长连接线程。lark-oapi 缺失或加载失败时清晰降级（绝不影响调度）。"""
        # lark-oapi 是**可选依赖**：用 importlib 动态导入，
        # 这样顶层不会去 import 它（未安装的环境/静态分析都不受影响）。
        try:
            lark = importlib.import_module("lark_oapi")
        except ImportError:
            self.logger.error(
                "feishu.interactive_enabled=true 但未安装 lark-oapi，交互查询不可用。\n"
                "  安装: python3 -m pip install lark-oapi\n"
                "  （不影响调度与通知；装好后 `gpu-manager reload` 或重启即可）"
            )
            return False
        except Exception as exc:
            # 装了但**加载失败**: 例如环境里 setuptools/pkg_resources 与当前 Python
            # 不兼容会抛 AttributeError（而不是 ImportError）。同样只降级，绝不向上抛。
            self.logger.error(
                "lark-oapi 已安装但加载失败: %s: %s\n"
                "  常见原因: setuptools 过旧与当前 Python 不兼容"
                "（Python 3.12 需要 setuptools >= 68）\n"
                "  修复: python3 -m pip install -U setuptools\n"
                "  交互查询暂不可用；调度与通知不受影响。",
                type(exc).__name__, exc,
            )
            return False

        def on_message(data: Any) -> None:
            if self._stop.is_set():
                return  # 旧实例：静默丢弃事件
            try:
                event = _event_to_dict(data)
                if event is None:
                    return
                request = self.request_from_event(event)
                if request is None:
                    return
                self.handle_text(request)
            except Exception as exc:  # 绝不让回调异常冒出去
                self.logger.exception("处理飞书事件异常: %s", exc)

        try:
            handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(on_message)
                .build()
            )
            self._client = lark.ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=handler,
                log_level=lark.LogLevel.WARNING,
            )
        except Exception as exc:
            self.logger.error("初始化飞书长连接失败: %s", exc)
            return False

        self._thread = threading.Thread(target=self._run, name="gpu-manager-feishu-bot", daemon=True)
        self._thread.start()
        self.logger.info("飞书交互查询已启动（长连接）")
        return True

    def _run(self) -> None:
        try:
            self._client.start()  # 阻塞：内部自动重连
        except Exception as exc:
            if not self._stop.is_set():
                self.logger.error("飞书长连接异常退出: %s", exc)

    def stop(self, timeout: float = 3.0) -> None:
        """停止长连接。

        注意 lark-oapi 的 ``ws.Client`` **没有公开的 stop/close**，只有一个
        async 的 ``_disconnect()``。直接调用它只会得到一个从不 await 的协程对象
        （什么都不会发生），旧连接会继续持有并消费事件 —— 这正是「reload 后
        仍按旧白名单拒绝消息」的原因。因此这里做两件事:

          1. 置 ``_stop`` 标志: 让本实例**立即忽略后续事件**（正确性的硬保证）
          2. 尽力真正断开: 把 ``_disconnect()`` 调度到 SDK 自己的事件循环上执行

        另外: 寻常的设置变更（白名单/限流）不会走到这里 —— daemon 会原地更新，
        只有凭证变了才需要重建连接。
        """
        self._stop.set()
        client = self._client
        if client is not None:
            self._disconnect_client(client)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @staticmethod
    def _disconnect_client(client: Any) -> None:
        """尽力关闭 SDK 的 WebSocket（失败无妨：正确性靠 _stop 标志保证）。"""
        with contextlib.suppress(Exception):
            from lark_oapi.ws.client import loop as lark_loop

            # 这是 SDK 的私有异步方法，类型无法从 stub 推出，故显式标 Any
            disconnect: Any = getattr(client, "_disconnect", None)
            if callable(disconnect):
                coro: Any = disconnect()
                future = asyncio.run_coroutine_threadsafe(coro, lark_loop)
                future.result(timeout=3.0)


__all__ = [
    "COMMAND_ALIASES",
    "HELP_TEXT",
    "MAX_LOG_LINES",
    "MAX_REPLY_CHARS",
    "AllowList",
    "BotContext",
    "BotRequest",
    "BotTransport",
    "CommandRouter",
    "FeishuBot",
    "FeishuTransport",
    "GpuView",
    "MessageDeduper",
    "RateLimiter",
    "TaskView",
]
