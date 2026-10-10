"""飞书交互式查询测试：命令路由 + 安全边界 + 事件归一化。

重点锁定（这些是「聊天入口」最容易出问题的地方）:
  · **fail-closed**: 白名单为空时拒绝所有人；未授权请求**不读取任何状态**、不回内容
  · 限流与 message_id 去重（飞书事件至少投递一次）
  · 固定命令集，未知命令只回帮助，绝无 shell 透传
  · 回复长度有上限（不刷屏）
  · 事件归一化兼容 dict 与 SDK 对象两种形态
  · 出站仍受主机白名单约束（不因加了交互功能而绕过 SSRF 防护）

全部无网络、无文件系统副作用；真实收发用假 transport。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import secrets
from types import SimpleNamespace

import pytest

from gpu_manager.bot import (
    HELP_TEXT,
    MAX_LOG_LINES,
    MAX_REPLY_CHARS,
    AllowList,
    BotContext,
    BotRequest,
    CommandRouter,
    FeishuBot,
    FeishuTransport,
    GpuView,
    MessageDeduper,
    RateLimiter,
    TaskView,
    _event_to_dict,
)


def make_logger() -> logging.Logger:
    log = logging.getLogger("test.bot")
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.CRITICAL)
    return log


def dummy_credential(prefix: str = "unit-test") -> str:
    """运行时生成的占位凭证。

    测试里**不应出现任何看似真实密钥的字面量**（即使是假的也会被密钥扫描规则拦下）。
    """
    return f"{prefix}-{secrets.token_hex(4)}"


class FakeTransport:
    """记录发送内容的假传输层。"""

    def __init__(self, ok: bool = True) -> None:
        self.sent: list[tuple[str, str, str | None]] = []
        self.ok = ok

    def send_text(self, chat_id: str, text: str, *, reply_to: str | None = None):
        self.sent.append((chat_id, text, reply_to))
        return self.ok, "fake-ok" if self.ok else "fake-fail"


def make_context(**kw) -> BotContext:
    base = BotContext(
        server="labhost",
        mode="RUNNING",
        counts={"pending": 1, "running": 1, "completed": 2, "failed": 1, "cancelled": 0},
        gpus=[
            GpuView(
                index=0, utilization=95, memory_used_mb=44000, memory_total_mb=49140,
                schedulable=True, idle_streak=0, idle_required=3,
                foreign_pids=[1234], foreign_uids=[1001],
            ),
            GpuView(
                index=1, utilization=60, memory_used_mb=30000, memory_total_mb=49140,
                schedulable=True, idle_streak=0, idle_required=3,
                own_pids=[5678], task_name="exp1",
            ),
            GpuView(
                index=7, utilization=0, memory_used_mb=5, memory_total_mb=49140,
                schedulable=False, idle_streak=3, idle_required=3,
            ),
        ],
        tasks=[
            TaskView(
                name="exp1", status="running", gpu_ids=[1], gpu_num=1, priority=10,
                runtime_seconds=8100, progress_text="Epoch 12/500 (2.4%)",
                eta_text="预计 10-03 01:20 结束（约 2h5m 后 · 日志进度 12/500 外推）",
                eta_confidence="high",
            ),
            TaskView(
                name="exp2", status="pending", gpu_num=2, priority=5,
                wait_seconds=720, waiting_reason="需要 2 张确认空闲卡，当前可用 0",
            ),
            TaskView(name="old", status="failed", gpu_ids=[3], error="退出码 1"),
        ],
        daemon_pid=999,
        uptime_seconds=3600,
    )
    for key, value in kw.items():
        setattr(base, key, value)
    return base


# --------------------------------------------------------------------------- #
# 安全: 白名单 / 限流 / 去重
# --------------------------------------------------------------------------- #
def test_allowlist_is_fail_closed_when_empty():
    allow = AllowList([])
    assert not allow.configured
    assert not allow.is_allowed("ou_me"), "空白名单必须拒绝所有人"


def test_allowlist_matches_exactly():
    allow = AllowList(["ou_me", " ou_other "])
    assert allow.is_allowed("ou_me")
    assert allow.is_allowed("ou_other"), "两侧空白应被去掉"
    assert not allow.is_allowed("ou_me2")
    assert not allow.is_allowed("")


def test_allowlist_replace_updates_in_place():
    """reload 改白名单靠原地更新生效（不重建长连接）。"""
    allow = AllowList(["ou_a"])
    allow.replace(["ou_b", " ou_c "])
    assert not allow.is_allowed("ou_a")
    assert allow.is_allowed("ou_b") and allow.is_allowed("ou_c")
    allow.replace([])
    assert not allow.configured and not allow.is_allowed("ou_b")


def test_rate_limiter_allows_then_blocks_with_wait():
    limiter = RateLimiter(per_minute=3, window_seconds=60)
    now = 1000.0
    for _ in range(3):
        allowed, _ = limiter.check("u1", now)
        assert allowed
    allowed, wait = limiter.check("u1", now)
    assert not allowed and wait > 0


def test_rate_limiter_window_slides():
    limiter = RateLimiter(per_minute=2, window_seconds=60)
    limiter.check("u1", 1000.0)
    limiter.check("u1", 1010.0)
    assert not limiter.check("u1", 1020.0)[0]
    allowed, _ = limiter.check("u1", 1000.0 + 61)
    assert allowed, "窗口滑过后应重新放行"


def test_rate_limiter_is_per_user():
    limiter = RateLimiter(per_minute=1, window_seconds=60)
    assert limiter.check("u1", 1000.0)[0]
    assert not limiter.check("u1", 1000.0)[0]
    assert limiter.check("u2", 1000.0)[0], "不同用户互不影响"


def test_deduper_rejects_repeat_message_ids():
    deduper = MessageDeduper(capacity=16)
    assert deduper.first_time("m1")
    assert not deduper.first_time("m1")
    assert deduper.first_time("m2")
    assert deduper.first_time(""), "空 id 不参与去重（无法判定重复）"


def test_deduper_evicts_old_ids():
    deduper = MessageDeduper(capacity=16)
    for i in range(20):
        deduper.first_time(f"m{i}")
    assert not deduper.first_time("m19"), "最近处理过的算重复"
    assert deduper.first_time("m0"), "超出容量后旧的被淘汰，可再次处理"


# --------------------------------------------------------------------------- #
# 命令归一化
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected_cmd,expected_args",
    [
        ("状态", "status", []),
        ("gpu", "status", []),
        ("GPUs", "", []),
        ("状态 1", "status", ["1"]),
        ("@_user_1 状态", "status", []),
        ("  @_user_1  队列  ", "tasks", []),
        ("日志 exp1 50", "log", ["exp1", "50"]),
        ("预计", "eta", []),
        ("help", "help", []),
        ("？", "help", []),
        ("", "", []),
        ("   ", "", []),
        ("随便说点什么", "", []),
    ],
)
def test_normalize(text, expected_cmd, expected_args):
    assert CommandRouter.normalize(text) == (expected_cmd, expected_args)


def test_normalize_strips_mentions_even_without_command():
    text = json.dumps({"text": "@_user_1 状态"})
    assert CommandRouter.normalize(text)[0] == ""


# --------------------------------------------------------------------------- #
# 命令渲染
# --------------------------------------------------------------------------- #
def test_render_status_contains_gpu_rows_and_mode():
    text = CommandRouter().handle("状态", make_context())
    assert "labhost" in text and "RUNNING" in text
    assert "GPU" in text or "卡" in text
    assert "95%" in text and "44000/49140MB" in text
    assert "他人 uid1001" in text
    assert "exp1" in text, "自己任务占用的卡要标出任务名"
    assert "GPU7" in text, "确认空闲的卡要列出"


def test_render_status_handles_nvidia_smi_failure():
    text = CommandRouter().handle("状态", make_context(gpus=[], gpu_error="nvidia-smi timeout"))
    assert "读取失败" in text and "nvidia-smi timeout" in text


def test_render_tasks_shows_progress_eta_and_waiting_reason():
    text = CommandRouter().handle("队列", make_context())
    assert "exp1" in text and "Epoch 12/500" in text
    assert "预计" in text
    assert "exp2" in text and "当前可用 0" in text
    assert "old" in text and "退出码 1" in text


def test_render_tasks_empty_queue():
    text = CommandRouter().handle("队列", make_context(tasks=[]))
    assert "队列为空" in text


def test_render_tasks_caps_long_list():
    tasks = [TaskView(name=f"t{i}", status="pending", gpu_num=1) for i in range(20)]
    text = CommandRouter().handle("队列", make_context(tasks=tasks))
    assert "另有 5 个任务" in text


def test_render_eta_per_gpu_free_time():
    text = CommandRouter().handle("预计", make_context())
    assert "exp1" in text and "Epoch 12/500" in text
    assert "GPU1" in text and "等 `exp1` 结束" in text
    assert "GPU0" in text and "uid1001" in text and "无法估算" in text
    assert "GPU7" in text and "已空闲" in text


def test_render_eta_without_running_tasks():
    text = CommandRouter().handle("预计", make_context(tasks=[TaskView(name="p", status="pending")]))
    assert "没有运行中的任务" in text


def test_render_log_uses_injected_reader():
    ctx = make_context(log_reader=lambda name, lines: [f"{name}-line-{i}" for i in range(lines)])
    text = CommandRouter().handle("日志 exp1 3", ctx)
    assert "exp1-line-0" in text and "exp1-line-2" in text


def test_render_log_unknown_task():
    ctx = make_context(log_reader=lambda name, lines: None)
    text = CommandRouter().handle("日志 不存在", ctx)
    assert "找不到任务" in text


def test_render_log_without_reader():
    text = CommandRouter().handle("日志 exp1", make_context(log_reader=None))
    assert "不可用" in text


def test_render_log_requires_task_name():
    text = CommandRouter().handle("日志", make_context(log_reader=lambda name, lines: []))
    assert "用法" in text and "exp1" in text


def test_render_log_clamps_line_count():
    seen: dict = {}

    def reader(name: str, lines: int):
        seen["lines"] = lines
        return ["x"]

    CommandRouter().handle("日志 exp1 99999", make_context(log_reader=reader))
    assert seen["lines"] == MAX_LOG_LINES, "行数必须有上限"


def test_render_log_bounds_reply_length():
    ctx = make_context(log_reader=lambda name, lines: ["y" * 500] * 50)
    text = CommandRouter().handle("日志 exp1 100", ctx)
    assert len(text) <= MAX_REPLY_CHARS


def test_unknown_command_returns_help():
    text = CommandRouter().handle("请帮我训练一个模型", make_context())
    assert "没看懂" in text and HELP_TEXT.splitlines()[0] in text


def test_help_command():
    assert CommandRouter().handle("帮助", make_context()) == HELP_TEXT


# --------------------------------------------------------------------------- #
# 事件归一化
# --------------------------------------------------------------------------- #
def _message_event(text: str = "状态", message_type: str = "text", **kw) -> dict:
    event = {
        "message": {
            "message_type": message_type,
            "content": json.dumps({"text": text}),
            "chat_id": "oc_chat",
            "message_id": "om_1",
            "chat_type": "p2p",
        },
        "sender": {"sender_id": {"open_id": "ou_me"}},
    }
    for key, value in kw.items():
        event["message"][key] = value
    return event


def test_request_from_dict_event():
    request = FeishuBot.request_from_event(_message_event("@_user_1 状态"))
    assert request is not None
    assert request.user_id == "ou_me"
    assert request.chat_id == "oc_chat"
    assert request.message_id == "om_1"
    assert "状态" in request.text


def test_request_from_event_rejects_non_text():
    assert FeishuBot.request_from_event(_message_event(message_type="image")) is None


def test_request_from_event_tolerates_broken_payloads():
    assert FeishuBot.request_from_event({}) is None
    assert FeishuBot.request_from_event({"message": {}}) is None
    assert FeishuBot.request_from_event(
        {"message": {"message_type": "text", "content": "not-json"}, "sender": {}}
    ) is None
    assert FeishuBot.request_from_event(
        {"message": {"message_type": "text", "content": json.dumps({"text": ""})}, "sender": {}}
    ) is None


def test_request_from_object_event():
    """兼容 lark-oapi 把事件包装成对象（不依赖 SDK 内部结构）。"""
    sender_id = SimpleNamespace(open_id="ou_obj")
    sender = SimpleNamespace(sender_id=sender_id)
    message = SimpleNamespace(
        message_type="text",
        content=json.dumps({"text": "状态"}),
        chat_id="oc_obj",
        message_id="om_obj",
        chat_type="group",
    )
    event = SimpleNamespace(message=message, sender=sender)
    wrapper = SimpleNamespace(event=event)

    normalized = _event_to_dict(wrapper)
    assert normalized is not None
    request = FeishuBot.request_from_event(normalized)
    assert request is not None
    assert (request.user_id, request.chat_id, request.message_id) == ("ou_obj", "oc_obj", "om_obj")


def test_event_to_dict_tolerates_missing_attributes():
    assert _event_to_dict(object()) is None
    assert _event_to_dict(SimpleNamespace(event=SimpleNamespace())) is None


# --------------------------------------------------------------------------- #
# FeishuBot: 鉴权 → 限流 → 上下文 → 回复
# --------------------------------------------------------------------------- #
def make_bot(*, allowed=("ou_me",), transport=None, context=None, audit=None, per_minute=10):
    transport = transport or FakeTransport()
    audits: list[dict] = []
    bot = FeishuBot(
        app_id="cli_x",
        app_secret=dummy_credential(),
        router=CommandRouter(),
        context_provider=lambda: context if context is not None else make_context(),
        transport=transport,
        allowlist=AllowList(allowed),
        rate_limiter=RateLimiter(per_minute=per_minute),
        logger=make_logger(),
        audit=(audit if audit is not None else audits.append),
    )
    return bot, transport, audits


def test_authorized_user_gets_reply_and_audit():
    bot, transport, audits = make_bot()
    ok, _detail = bot.handle_text(BotRequest(text="状态", user_id="ou_me", chat_id="oc", message_id="m1"))

    assert ok
    assert len(transport.sent) == 1
    chat_id, text, reply_to = transport.sent[0]
    assert chat_id == "oc" and reply_to == "m1", "应作为该消息的回复发出"
    assert "labhost" in text
    assert audits[-1]["event"] == "bot_command" and audits[-1]["user_id"] == "ou_me"


def test_unauthorized_user_gets_nothing_and_state_is_not_read():
    read = {"count": 0}

    def provider() -> BotContext:
        read["count"] += 1
        return make_context()

    transport = FakeTransport()
    bot = FeishuBot(
        app_id="cli_x", app_secret=dummy_credential(), router=CommandRouter(), context_provider=provider,
        transport=transport, allowlist=AllowList(["ou_me"]), logger=make_logger(),
        audit=lambda record: None,
    )
    ok, detail = bot.handle_text(BotRequest(text="状态", user_id="ou_stranger", chat_id="oc"))

    assert not ok and detail == "unauthorized"
    assert transport.sent == [], "未授权请求绝不能被回复"
    assert read["count"] == 0, "未授权请求不应触发状态读取（白名单在取数之前判定）"


def test_denied_requests_are_audited_with_user_id():
    audits: list[dict] = []
    bot, _transport, _ = make_bot(allowed=("ou_me",), audit=audits.append)
    bot.handle_text(BotRequest(text="状态", user_id="ou_stranger", chat_id="oc"))
    assert audits[-1]["event"] == "bot_denied"
    assert audits[-1]["user_id"] == "ou_stranger", "审计里必须带 open_id，方便加入白名单"


def test_duplicate_message_id_is_replied_once():
    bot, transport, _audits = make_bot()
    request = BotRequest(text="状态", user_id="ou_me", chat_id="oc", message_id="same")
    bot.handle_text(request)
    ok, detail = bot.handle_text(request)
    assert not ok and detail == "duplicate"
    assert len(transport.sent) == 1


def test_rate_limited_user_is_told_to_wait():
    bot, transport, audits = make_bot(per_minute=1)
    bot.handle_text(BotRequest(text="状态", user_id="ou_me", chat_id="oc", message_id="m1"))
    ok, _ = bot.handle_text(BotRequest(text="状态", user_id="ou_me", chat_id="oc", message_id="m2"))

    assert ok, "限流时也要给个明确回复"
    assert "太频繁" in transport.sent[-1][1]
    assert audits[-1]["event"] == "bot_rate_limited"


def test_internal_error_does_not_break_the_bot():
    def boom() -> BotContext:
        raise RuntimeError("state exploded")

    transport = FakeTransport()
    bot = FeishuBot(
        app_id="cli_x", app_secret=dummy_credential(), router=CommandRouter(), context_provider=boom,
        transport=transport, allowlist=AllowList(["ou_me"]), logger=make_logger(),
        audit=lambda record: None,
    )
    ok, _detail = bot.handle_text(BotRequest(text="状态", user_id="ou_me", chat_id="oc"))

    assert ok
    assert "内部错误" in transport.sent[-1][1]


def test_reply_failure_is_reported_not_raised():
    bot, _transport, _audits = make_bot(transport=FakeTransport(ok=False))
    ok, detail = bot.handle_text(BotRequest(text="状态", user_id="ou_me", chat_id="oc"))
    assert not ok and detail == "fake-fail"


def test_stopped_bot_ignores_events():
    """回归: 被替换/停止的旧实例必须静默丢弃事件。

    lark-oapi 的 ws.Client 没有公开的 stop，旧连接可能仍在消费事件；
    若旧实例还用旧白名单应答，就会出现「改了白名单仍是未授权」（用户实际踩到）。
    """
    bot, transport, _audits = make_bot()
    bot.stop()  # 没有真实 client，仅置停止标志

    ok, detail = bot.handle_text(
        BotRequest(text="状态", user_id="ou_me", chat_id="oc", message_id="m1")
    )

    assert not ok and detail == "stopped"
    assert transport.sent == [], "已停止的实例不应再发出任何消息"


def test_bot_start_degrades_cleanly_without_sdk():
    """未安装 lark-oapi 时应返回 False 并记日志，而不是抛异常。"""
    bot, _transport, _audits = make_bot()
    # 用 find_spec 探测是否安装（不 import，避免把可选依赖变成硬依赖）
    if importlib.util.find_spec("lark_oapi") is not None:
        pytest.skip("环境里已安装 lark-oapi，跳过降级测试")

    assert bot.start() is False
    bot.stop()


def test_bot_start_degrades_when_sdk_load_raises(monkeypatch):
    """回归: SDK “装了但加载失败”时必须同样降级。

    真实案例: 环境里的 setuptools 过旧，在 Python 3.12 上 pkg_resources 抛
    ``AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`` —— 它不是
    ImportError，曾穿透降级逻辑，把 daemon 的启动路径直接拖崩。
    """
    import gpu_manager.bot as bot_mod

    def boom(name: str):
        raise AttributeError("module 'pkgutil' has no attribute 'ImpImporter'")

    monkeypatch.setattr(bot_mod.importlib, "import_module", boom)
    bot, _transport, _audits = make_bot()

    assert bot.start() is False, "只降级，不得向上抛异常"
    bot.stop()


# --------------------------------------------------------------------------- #
# 出站安全: 主机白名单同样适用于交互回复
# --------------------------------------------------------------------------- #
def test_transport_rejects_non_allowlisted_host_without_network():
    """出站白名单同样约束交互回复：不在白名单就**连请求都不发**。"""
    transport = FeishuTransport(
        "cli_x",
        dummy_credential(),
        make_logger(),
        allowed_hosts=["example.invalid"],  # 故意不含 open.feishu.cn
        allow_any_host=False,
    )
    ok, detail = transport.send_text("oc", "hi")
    assert not ok
    assert "白名单" in detail, "必须在发请求前被拦下（纯离线）"


def test_transport_fails_fast_without_credentials():
    """凭证为空时快速失败，不发任何网络请求。"""
    transport = FeishuTransport("", "", make_logger(), allowed_hosts=["open.feishu.cn"])
    ok, detail = transport.send_text("oc", "hi")
    assert not ok
    assert "未配置" in detail


# --------------------------------------------------------------------------- #
# 阶段展示（多阶段流程: 先训 8 卡、再单卡 eval）
# --------------------------------------------------------------------------- #
def test_render_tasks_and_eta_show_stage():
    ctx = make_context(
        tasks=[
            TaskView(
                name="exp1",
                status="running",
                gpu_ids=[1],
                gpu_num=1,
                runtime_seconds=600,
                stage="evaluating",
                progress_text="Epoch 141/500",
                eta_text="预计 01-01 10:00 结束",
            )
        ]
    )
    router = CommandRouter()

    assert "阶段 evaluating" in router.handle("队列", ctx)
    assert "阶段 evaluating" in router.handle("预计", ctx)


def test_render_tasks_without_stage_has_no_stage_noise():
    text = CommandRouter().handle("队列", make_context())
    assert "阶段" not in text, "未声明 stage_file 的任务不应出现阶段字样"
