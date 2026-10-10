"""通知层测试：出站 URL 白名单（防 SSRF）、重试策略、降级行为、payload 构造。

测试全部不打真实网络：非法主机/非法 scheme 会在发请求前被拦下；
发送路径用 monkeypatch 替换 `_http_post_json` 来断言真实发出的报文。

通知层现在**只有自建应用**一种方式（群机器人 webhook 已取消）：
自定义群机器人只能「发」不能「收」，而查询必须用应用，于是通知复用同一份凭证。
"""

from __future__ import annotations

import json
import logging
import secrets

import pytest

import gpu_manager.notifier as notifier_mod
from gpu_manager.config import FeishuConfig
from gpu_manager.notifier import (
    AppNotifier,
    NullNotifier,
    _build_retry_options,
    _session,
    _validate_url,
    build_notifier,
)

API_URL = "https://open.feishu.cn/open-apis/im/v1/messages"


def dummy_credential(prefix: str = "unit-test") -> str:
    """生成测试用占位凭证。

    刻意在运行时拼接：**测试里不应出现任何看似真实密钥的字面量**
    （即使能猜到是假的，把它写进代码也会被密钥扫描规则拦下）。
    """
    return f"{prefix}-{secrets.token_hex(4)}"


def logger() -> logging.Logger:
    log = logging.getLogger("test.notifier")
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.CRITICAL)
    return log


def make_cfg(**kw) -> FeishuConfig:
    """默认造一份「开箱可用」的配置：凭证齐备，目标由白名单自动推导。"""
    base: dict = {
        "enabled": True,
        "app_id": "cli_unit_test",
        "app_secret": dummy_credential(),
        "allowed_users": ["ou_unit_test"],
    }
    base.update(kw)
    return FeishuConfig(**base)


# --------------------------------------------------------------------------- #
# 出站 URL 白名单
# --------------------------------------------------------------------------- #
def test_official_hosts_allowed():
    cfg = make_cfg()
    assert _validate_url(API_URL, cfg.allowed_hosts) is None
    assert _validate_url("https://open.larksuite.com/x", cfg.allowed_hosts) is None


def test_subdomain_allowed():
    assert _validate_url("https://gw.open.feishu.cn/x", ["open.feishu.cn"]) is None


def test_unknown_host_rejected():
    error = _validate_url("https://evil.example.com/hook", ["open.feishu.cn", "open.larksuite.com"])
    assert error is not None
    assert "白名单" in error and "evil.example.com" in error


def test_non_http_scheme_rejected():
    for url in ("file:///etc/passwd", "ftp://open.feishu.cn/x", "gopher://x"):
        error = _validate_url(url, ["open.feishu.cn"])
        assert error is not None and "scheme" in error


def test_allow_any_host_opt_out():
    assert _validate_url("https://my-gateway.internal/hook", ["open.feishu.cn"], True) is None


def test_empty_allowlist_means_no_host_restriction():
    assert _validate_url("https://anything.example.com/x", []) is None
    assert _validate_url("https://anything.example.com/x", None) is None


def test_missing_hostname_rejected():
    error = _validate_url("https:///path", ["open.feishu.cn"])
    assert error is not None and "主机名" in error


# --------------------------------------------------------------------------- #
# 重试策略（用库的 Retry 替代手写 sleep 重试）
# --------------------------------------------------------------------------- #
def test_retry_options_enable_post_and_backoff():
    options = _build_retry_options()
    method_key = "allowed_methods" if "allowed_methods" in options else "method_whitelist"
    assert "POST" in options[method_key], "POST 默认不在可重试方法里，必须显式开启"
    assert options["total"] == 2
    assert options["backoff_factor"] > 0
    assert 429 in options["status_forcelist"]


def test_session_mounts_retrying_adapter():
    session = _session()
    if session is None:  # pragma: no cover - 无 requests 的环境
        pytest.skip("requests 不可用")
    retries = session.get_adapter("https://open.feishu.cn").max_retries
    assert retries.total == 2
    assert "POST" in retries.allowed_methods


# --------------------------------------------------------------------------- #
# 降级行为：通知没配好也不能影响调度
# --------------------------------------------------------------------------- #
def test_null_notifier_never_sends():
    notifier = NullNotifier(logger(), reason="test")
    assert notifier.send("t", "b") == (True, "suppressed (test)")
    assert len(notifier.sent) == 1


def test_build_notifier_degrades_without_credentials():
    notifier = build_notifier(make_cfg(app_id="", app_secret=""), logger())
    assert isinstance(notifier, NullNotifier), "缺少应用凭证时必须降级，而不是抛异常"


def test_build_notifier_degrades_without_target():
    """凭证齐备但没有任何收件人 → 也要降级（否则每次发送都白跑一趟网络）。"""
    notifier = build_notifier(make_cfg(allowed_users=[], receive_id=""), logger())
    assert isinstance(notifier, NullNotifier)


def test_build_notifier_dry_run():
    assert isinstance(build_notifier(make_cfg(), logger(), dry_run=True), NullNotifier)


def test_build_notifier_returns_app_notifier():
    assert isinstance(build_notifier(make_cfg(), logger()), AppNotifier)


def test_build_notifier_disabled():
    assert isinstance(build_notifier(make_cfg(enabled=False), logger()), NullNotifier)


def test_build_notifier_ignores_query_only_warnings():
    """交互查询没配好（白名单为空）不该连累通知降级。"""
    cfg = make_cfg(allowed_users=[], receive_id="ou_explicit", interactive_enabled=True)
    assert "fail-closed" in " ".join(cfg.credential_warnings())
    assert isinstance(build_notifier(cfg, logger()), AppNotifier)


# --------------------------------------------------------------------------- #
# 发送：报文结构 / 目标解析 / 失败路径
# --------------------------------------------------------------------------- #
class _FakePost:
    """替换 _http_post_json：分别应答 token 请求与消息发送，记录真实报文。"""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url, payload, **_kw):
        self.requests.append((url, payload))
        if "tenant_access_token" in url:
            return True, {"code": 0, "tenant_access_token": "t-unit", "expire": 7200}
        return True, {"code": 0, "msg": "ok"}

    @property
    def message(self) -> tuple[str, dict]:
        messages = [(u, p) for u, p in self.requests if "tenant_access_token" not in u]
        assert messages, "没有发出消息请求"
        return messages[-1]


def test_app_send_uses_whitelist_target_and_card(monkeypatch):
    fake = _FakePost()
    monkeypatch.setattr(notifier_mod, "_http_post_json", fake)
    cfg = make_cfg(allowed_users=["ou_me"])

    ok, detail = AppNotifier(cfg, logger()).send("标题", "正文", level="error")

    assert ok and detail == "ok"
    url, payload = fake.message
    assert "receive_id_type=open_id" in url
    assert payload["receive_id"] == "ou_me", "留空 receive_id 时应自动用白名单第一人（私聊）"
    assert payload["msg_type"] == "interactive"
    card = json.loads(payload["content"])
    assert card["header"]["template"] == "red", "error 级别应映射成红色表头"
    assert card["header"]["title"]["content"] == "标题"
    assert card["elements"][0]["text"]["content"] == "正文"


def test_app_send_explicit_receive_id_wins(monkeypatch):
    fake = _FakePost()
    monkeypatch.setattr(notifier_mod, "_http_post_json", fake)
    cfg = make_cfg(allowed_users=["ou_me"], receive_id="oc_group", receive_id_type="chat_id")

    assert AppNotifier(cfg, logger()).send("t", "b")[0] is True
    url, payload = fake.message
    assert "receive_id_type=chat_id" in url
    assert payload["receive_id"] == "oc_group", "显式配置优先于白名单推导"


def test_app_send_text_when_card_disabled(monkeypatch):
    fake = _FakePost()
    monkeypatch.setattr(notifier_mod, "_http_post_json", fake)
    cfg = make_cfg(card=False)

    assert AppNotifier(cfg, logger()).send("标题", "正文")[0] is True
    _url, payload = fake.message
    assert payload["msg_type"] == "text"
    assert "标题" in json.loads(payload["content"])["text"]


def test_app_send_mention_all(monkeypatch):
    fake = _FakePost()
    monkeypatch.setattr(notifier_mod, "_http_post_json", fake)
    cfg = make_cfg(card=False, mention_all=True)

    assert AppNotifier(cfg, logger()).send("t", "b")[0] is True
    _url, payload = fake.message
    assert "<at id=all></at>" in json.loads(payload["content"])["text"]


def test_send_without_target_fails_cleanly():
    notifier = AppNotifier(make_cfg(allowed_users=[], receive_id=""), logger())
    ok, detail = notifier.send("t", "b")
    assert not ok and "未配置通知目标" in detail, "无目标时应立即失败，不发任何请求"


def test_send_reports_feishu_error_code(monkeypatch):
    class Failing(_FakePost):
        def __call__(self, url, payload, **_kw):
            if "tenant_access_token" in url:
                return super().__call__(url, payload)
            return True, {"code": 99991663, "msg": "no permission"}

    monkeypatch.setattr(notifier_mod, "_http_post_json", Failing())
    ok, detail = AppNotifier(make_cfg(), logger()).send("t", "b")
    assert not ok
    assert "99991663" in detail and "no permission" in detail


def test_app_notifier_rejects_disallowed_base_host():
    """出站白名单同样约束应用 API（防 SSRF）：主机不在白名单时立刻失败。"""
    cfg = make_cfg(allowed_hosts=["my-gateway.internal"])
    notifier = AppNotifier(cfg, logger())
    ok, detail = notifier.send("t", "b")
    assert not ok
    assert "白名单" in detail


def test_token_is_cached_between_sends(monkeypatch):
    fake = _FakePost()
    monkeypatch.setattr(notifier_mod, "_http_post_json", fake)
    notifier = AppNotifier(make_cfg(), logger())

    assert notifier.send("t1", "b")[0] is True
    assert notifier.send("t2", "b")[0] is True

    token_calls = [u for u, _p in fake.requests if "tenant_access_token" in u]
    assert len(token_calls) == 1, "第二次发送应复用缓存 token（避免每发一条都换 token）"
