"""飞书通知 / 交互查询共用的自建应用客户端。

选型: **只保留自建应用**（tenant_access_token + im/v1/messages）。
自定义群机器人 Webhook 只能「发」不能「收」，且需要单独维护一份凭证，
已取消；现在通知与「在飞书里问答」共用同一个应用凭证。

通知层是「哑」的：不做去重/冷却决策，那属于 daemon 的职责。
所有网络失败只记录日志，绝不影响调度主循环。
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

try:  # requests 已确认可用；缺失时降级到 urllib
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    HTTPAdapter = None  # type: ignore[assignment,misc]
    Retry = None  # type: ignore[assignment,misc]

import urllib.error
import urllib.parse
import urllib.request

from .config import FeishuConfig
from .utils import truncate

LEVEL_TEMPLATE = {
    "info": "blue",
    "success": "green",
    "warning": "orange",
    "error": "red",
}


class Notifier(ABC):
    """通知后端接口。"""

    name = "base"

    @abstractmethod
    def send(self, title: str, body: str, level: str = "info", mention_all: bool = False) -> tuple[bool, str]:
        """返回 (是否成功, 说明)。实现不得抛异常。"""

    def close(self) -> None:  # pragma: no cover - 预留
        return None


class NullNotifier(Notifier):
    """禁用通知（feishu.enabled=false 或 --dry-run）：只写日志。"""

    name = "null"

    def __init__(self, logger: logging.Logger | None = None, reason: str = "disabled") -> None:
        self.logger = logger or logging.getLogger("gpu_manager.notify")
        self.reason = reason
        self.sent: list[tuple[str, str, str]] = []

    def send(self, title: str, body: str, level: str = "info", mention_all: bool = False) -> tuple[bool, str]:
        self.sent.append((level, title, body))
        self.logger.info("[notify:%s] %s\n%s", self.reason, title, body)
        return True, f"suppressed ({self.reason})"


class AppNotifier(Notifier):
    """飞书自建应用（tenant_access_token + im/v1/messages）。"""

    name = "app"

    BASE = "https://open.feishu.cn/open-apis"

    def __init__(self, cfg: FeishuConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self._token = ""
        self._token_expire_at = 0.0

    def _tenant_token(self) -> tuple[bool, str]:
        if self._token and time.time() < self._token_expire_at - 60:
            return True, self._token
        ok, data = _http_post_json(
            f"{self.BASE}/auth/v3/tenant_access_token/internal",
            {"app_id": self.cfg.app_id, "app_secret": self.cfg.app_secret},
            timeout=self.cfg.timeout_seconds,
            verify=self.cfg.verify_ssl,
            allowed_hosts=self.cfg.allowed_hosts,
            allow_any_host=self.cfg.allow_any_host,
        )
        if not ok:
            return False, str(data)
        token = data.get("tenant_access_token", "")
        if not token:
            return False, f"获取 token 失败: {data}"
        self._token = token
        self._token_expire_at = time.time() + int(data.get("expire", 7200))
        return True, token

    def send(self, title: str, body: str, level: str = "info", mention_all: bool = False) -> tuple[bool, str]:
        target = self.cfg.resolved_receive_id()
        if not target:
            return False, "未配置通知目标（feishu.receive_id 与 allowed_users 都为空）"
        ok, token = self._tenant_token()
        if not ok:
            return False, token

        text = body
        if self.cfg.mention_all or mention_all:
            text = f"{text}\n<at id=all></at>"

        if self.cfg.card:
            msg_type = "interactive"
            content: Any = {
                "config": {"wide_screen_mode": True},
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": LEVEL_TEMPLATE.get(level, "blue"),
                },
                "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": text}}],
            }
        else:
            msg_type = "text"
            content = {"text": f"{title}\n{text}"}

        url = f"{self.BASE}/im/v1/messages?receive_id_type={self.cfg.receive_id_type}"
        payload = {
            "receive_id": target,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        ok, data = _http_post_json(
            url, payload, timeout=self.cfg.timeout_seconds, verify=self.cfg.verify_ssl,
            headers={"Authorization": f"Bearer {token}"},
            allowed_hosts=self.cfg.allowed_hosts,
            allow_any_host=self.cfg.allow_any_host,
        )
        if not ok:
            return False, str(data)
        if data.get("code") in (0, None):
            return True, "ok"
        return False, f"飞书返回 code={data.get('code')} msg={data.get('msg')}"


# --------------------------------------------------------------------------- #
# HTTP 细节
# --------------------------------------------------------------------------- #
_SESSION: Any = None


def _build_retry_options() -> dict[str, Any]:
    """构造 urllib3 Retry 参数。

    两个要点:
      · POST **不在** urllib3 默认的可重试方法里，必须显式加入，否则重试不会发生；
      · 参数名跨版本不同：urllib3 >= 1.26 为 allowed_methods（新），
        更早版本为 method_whitelist。这里按**运行时实际签名**选择，
        既不绑定某个版本的 stub，也兼容旧环境。
    """
    options: dict[str, Any] = {
        "total": 2,
        "connect": 2,
        "read": 2,
        "status": 2,
        "backoff_factor": 0.5,  # 0.5s → 1s 退避
        "status_forcelist": (429, 500, 502, 503, 504),
        "raise_on_status": False,
    }
    if Retry is None:  # pragma: no cover - requests 不可用时不会创建 session
        return options
    params = inspect.signature(Retry.__init__).parameters
    method_key = "allowed_methods" if "allowed_methods" in params else "method_whitelist"
    options[method_key] = frozenset({"POST"})
    return options


def _session() -> Any:
    """带退避重试的 requests.Session（进程内单例）。

    用 urllib3 的 Retry 统一处理瞬时失败（连接错误 / 429 / 5xx）与指数退避，
    而不是手写「重试一次 + time.sleep」：重试策略交给 HTTP 库更可靠，
    也不在通知路径上做无意义的固定等待。
    """
    global _SESSION
    if _SESSION is not None or requests is None or HTTPAdapter is None or Retry is None:
        return _SESSION
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(**_build_retry_options()))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _SESSION = session
    return _SESSION


def _validate_url(
    url: str,
    allowed_hosts: Iterable[str] | None = None,
    allow_any_host: bool = False,
) -> str | None:
    """出站 URL 校验（防 SSRF）：scheme 仅允许 http/https，且主机必须在白名单内。

    返回错误说明；None 表示通过。
    """
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return f"不支持的 URL scheme: {scheme or '(空)'}（仅允许 http/https）"
    if allow_any_host:
        return None
    allowed = {str(h).strip().lower() for h in (allowed_hosts or ()) if str(h).strip()}
    if not allowed:
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return "URL 缺少主机名"
    if host in allowed or any(host.endswith("." + h) for h in allowed):
        return None
    return (
        f"主机 {host!r} 不在白名单 {sorted(allowed)} 内（防 SSRF）。"
        "若使用自建网关/代理，请把它加入 feishu.allowed_hosts 或设置 allow_any_host: true"
    )


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
    verify: bool = True,
    headers: dict[str, str] | None = None,
    allowed_hosts: Iterable[str] | None = None,
    allow_any_host: bool = False,
) -> tuple[bool, Any]:
    """返回 (是否 HTTP 成功, 解析后的 JSON 或错误字符串)。"""
    url_error = _validate_url(url, allowed_hosts, allow_any_host)
    if url_error:
        return False, url_error

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        request_headers.update(headers)

    session = _session()
    if session is not None:
        try:
            resp = session.post(url, data=body, headers=request_headers, timeout=timeout, verify=verify)
        except Exception as exc:  # requests 异常体系复杂，统一兜住
            return False, f"{type(exc).__name__}: {exc}"
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}: {truncate(resp.text, 300)}"
        try:
            return True, resp.json()
        except ValueError:
            return False, f"响应非 JSON: {truncate(resp.text, 200)}"

    req = urllib.request.Request(  # noqa: S310 - scheme 与主机已由 _validate_url 白名单校验
        url, data=body, headers=request_headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 同上
            raw = resp.read().decode("utf-8", errors="replace")
        return True, json.loads(raw)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {truncate(exc.read().decode('utf-8', 'replace'), 300)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def build_notifier(cfg: FeishuConfig, logger: logging.Logger, dry_run: bool = False) -> Notifier:
    if dry_run:
        return NullNotifier(logger, reason="dry-run")
    if not cfg.enabled:
        logger.warning("飞书通知已禁用（feishu.enabled=false），消息只会写入日志")
        return NullNotifier(logger, reason="feishu.enabled=false")
    warnings = cfg.notification_warnings()
    if warnings:
        # 刻意不抛异常: 通知没配好也必须能继续监控与调度训练
        for warning in warnings:
            logger.warning("%s", warning)
        return NullNotifier(logger, reason="通知配置不完整")
    return AppNotifier(cfg, logger)
