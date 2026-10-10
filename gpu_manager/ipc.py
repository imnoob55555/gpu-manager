"""进程内 IPC：CLI ↔ daemon（AF_UNIX + 单行 JSON）。

CLI 优先走 socket（daemon 在线时立即可靠地变更状态）；daemon 不在线时 CLI 退化为
直接读写 state.json。socket 权限 0600，仅当前用户可用。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import selectors
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .utils import abs_path

ProtocolHandler = Callable[[str, dict[str, Any]], dict[str, Any]]


def ok(data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "data": data or {}}


def fail(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


# --------------------------------------------------------------------------- #
# 客户端（CLI 用）
# --------------------------------------------------------------------------- #
def send_request(
    socket_path: str | os.PathLike[str],
    command: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """发送命令。daemon 不在线/不可用时返回 None（调用方退化为直接操作状态文件）。"""
    path = abs_path(socket_path)
    if not path.exists():
        return None
    payload = json.dumps({"command": command, "params": params or {}}, ensure_ascii=False)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(path))
        sock.sendall(payload.encode("utf-8") + b"\n")
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except (OSError, TimeoutError):
        return None
    finally:
        with contextlib.suppress(OSError):
            sock.close()
    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not raw:
        return None
    try:
        result = json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


# --------------------------------------------------------------------------- #
# 服务端（daemon 用）
# --------------------------------------------------------------------------- #
class ControlServer:
    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        handler: ProtocolHandler,
        logger: logging.Logger,
        *,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        self.path = abs_path(socket_path)
        self.handler = handler
        self.logger = logger
        self.on_wake = on_wake
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # 残留 socket 文件（上次异常退出）→ 清理
            with contextlib.suppress(OSError):
                self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        sock.listen(16)
        sock.settimeout(1.0)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="gpu-manager-ipc", daemon=True)
        self._thread.start()
        self.logger.info("控制接口已就绪: %s", self.path)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    def _serve(self) -> None:
        sock = self._sock
        if sock is None:
            self.logger.error("控制接口未初始化，服务线程退出")
            return
        # 用 selectors 等「有连接可接」，而不是 sleep 轮询：
        # 空闲时不空转，且能被停止标志及时打断（最长 0.5s）。
        selector = selectors.DefaultSelector()
        try:
            selector.register(sock, selectors.EVENT_READ)
            while not self._stop.is_set():
                try:
                    events = selector.select(timeout=0.5)
                except OSError:
                    if self._stop.is_set():
                        break
                    continue
                if not events:
                    continue
                try:
                    conn, _ = sock.accept()
                except OSError as exc:
                    if isinstance(exc, TimeoutError):
                        continue  # accept 超时：只为周期性检查停止标志
                    if self._stop.is_set():
                        break
                    self.logger.warning("接受控制连接失败: %s", exc)
                    continue
                try:
                    self._handle_conn(conn)
                except Exception as exc:  # 任何异常都不能让服务线程死掉
                    self.logger.exception("处理控制请求异常: %s", exc)
                finally:
                    with contextlib.suppress(OSError):
                        conn.close()
        finally:
            with contextlib.suppress(Exception):
                selector.unregister(sock)
            selector.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        conn.settimeout(30.0)
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if not raw:
            return
        try:
            request = json.loads(raw.splitlines()[0])
        except json.JSONDecodeError:
            response = fail("请求不是合法 JSON")
        else:
            command = str(request.get("command", ""))
            params = request.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            try:
                response = self.handler(command, params)
            except Exception as exc:
                self.logger.exception("控制命令 %s 执行失败", command)
                response = fail(f"{type(exc).__name__}: {exc}")
            if command in ("add", "cancel", "remove", "requeue", "reload", "stop_daemon") and self.on_wake:
                # 让主循环立刻醒来处理（而不是等下一个 tick）
                self.on_wake()
        payload = json.dumps(response, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
        with contextlib.suppress(OSError):
            conn.sendall(payload)


def socket_in_use(socket_path: str | os.PathLike[str]) -> bool:
    """socket 文件存在且真的有人 listen。"""
    path = Path(abs_path(socket_path))
    if not path.exists():
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(2.0)
    try:
        probe.connect(str(path))
        return True
    except (OSError, TimeoutError):
        return False
    finally:
        with contextlib.suppress(OSError):
            probe.close()
