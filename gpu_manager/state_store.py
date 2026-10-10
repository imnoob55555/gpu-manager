"""状态持久化与文件锁。

- 权威运行状态存于 data/state.json（原子写）。
- 进程间互斥使用 filelock（daemon 与 CLI 可能并发读写）。
- 事件流追加写 data/events.jsonl，便于事后追溯「谁在什么时候动了什么」。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from .utils import abs_path, atomic_write_text

STATE_VERSION = 1
_LOCK_TIMEOUT = 15.0

_LOCKS: dict[str, FileLock] = {}


def get_lock(path: str | os.PathLike[str]) -> FileLock:
    """同一路径返回同一 FileLock 实例（同进程内可重入，避免自我死锁）。"""
    key = str(abs_path(path))
    lock = _LOCKS.get(key)
    if lock is None:
        lock = FileLock(key + ".lock")
        _LOCKS[key] = lock
    return lock


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "daemon_pid": None,
        "started_at": None,
        "updated_at": None,
        "mode": None,
        "mode_since": None,
        "next_seq": 1,
        "idle": {"counts": {}, "since": {}},
        "notify": {"gpu_idle_at": 0.0, "queue_wait_at": 0.0, "last_idle_signature": "", "last_error_at": 0.0},
        "tasks": [],
        "last_snapshot": {"timestamp": 0.0, "error": "", "gpus": []},
        "stats": {"launched": 0, "completed": 0, "failed": 0, "cancelled": 0},
    }


class StateStore:
    def __init__(self, path: str | os.PathLike[str], logger: logging.Logger) -> None:
        self.path = abs_path(path)
        self.logger = logger
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock = get_lock(self.path)

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #
    def read(self) -> dict[str, Any]:
        """读取状态；文件缺失/损坏时返回默认状态（损坏文件会备份）。"""
        if not self.path.exists():
            return default_state()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            backup = self.path.with_suffix(self.path.suffix + f".corrupt.{int(time.time())}")
            self.logger.error("state.json 不可解析(%s)，已备份到 %s 并重建", exc, backup)
            with contextlib.suppress(OSError):
                shutil.copy2(self.path, backup)
            return default_state()
        if not isinstance(raw, dict):
            self.logger.error("state.json 顶层不是对象，重建状态")
            return default_state()
        state = default_state()
        state.update(raw)
        # 保证嵌套结构完整（老版本状态升级）
        for key, value in default_state().items():
            if isinstance(value, dict) and isinstance(state.get(key), dict):
                merged = dict(value)
                merged.update(state[key])
                state[key] = merged
        return state

    def write(self, state: dict[str, Any]) -> None:
        state["version"] = STATE_VERSION
        state["updated_at"] = time.time()
        text = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=False)
        atomic_write_text(self.path, text, mode=0o600)

    def update(self, mutator: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """加锁读改写。mutator 原地修改 state 并返回 state（或 None）。"""
        with self._lock:
            state = self.read()
            result = mutator(state)
            self.write(state)
            return result if isinstance(result, dict) else state

    # ------------------------------------------------------------------ #
    # 事件流
    # ------------------------------------------------------------------ #
    def append_event(self, event: dict[str, Any], event_path: str | os.PathLike[str]) -> None:
        path = abs_path(event_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **event}
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # 事件记录失败不应影响主流程
            self.logger.warning("事件写入失败 %s: %s", path, exc)

    @staticmethod
    def read_events(event_path: str | os.PathLike[str], limit: int = 50) -> list[dict[str, Any]]:
        path = abs_path(event_path)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in lines[-max(1, limit):]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------ #
    # PID 文件
    # ------------------------------------------------------------------ #
    def write_pid(self, pid_path: str | os.PathLike[str], pid: int) -> None:
        atomic_write_text(pid_path, f"{pid}\n")

    @staticmethod
    def read_pid(pid_path: str | os.PathLike[str]) -> int | None:
        path = Path(abs_path(pid_path))
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return int(text) if text.isdigit() else None

    @staticmethod
    def clear_pid(pid_path: str | os.PathLike[str]) -> None:
        path = Path(abs_path(pid_path))
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


__all__ = ["StateStore", "default_state", "get_lock", "STATE_VERSION", "Timeout"]
