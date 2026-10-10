"""通用工具: 时间、路径、进程归属校验、原子写、日志。"""

from __future__ import annotations

import contextlib
import errno
import logging
import logging.handlers
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# 时间
# ---------------------------------------------------------------------------
def now_ts() -> float:
    return time.time()


def wait_until(
    predicate: Callable[[], bool],
    timeout: float,
    *,
    interval: float = 0.1,
    stop_event: threading.Event | None = None,
) -> bool:
    """有界等待 predicate() 成立，返回是否在超时前成立。

    为什么不用固定 `time.sleep(0.5)`：
      - 条件一满足就立刻返回，取消/停止任务时响应更快，测试也不白等；
      - 传入 stop_event 时，daemon 关闭可以**立即**打断等待，
        而不再傻等完整个 grace 周期。
    这是全项目**唯一**的轮询等待原语，其他地方一律复用它。
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    timer = stop_event or threading.Event()
    while True:
        if predicate():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return predicate()
        # Event.wait 可被 set() 立即唤醒（当 stop_event 被设置时立即返回 False）
        timer.wait(min(interval, remaining))
        if stop_event is not None and stop_event.is_set() and not predicate():
            return False


def fmt_ts(ts: float | None = None) -> str:
    """人类可读时间戳，用于通知与状态展示。"""
    if ts is None:
        ts = now_ts()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h}h{m}m"
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
def abs_path(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(p))).absolute()


def ensure_dir(p: str | os.PathLike[str]) -> Path:
    path = abs_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> Path:
    """把 path 解析为绝对路径并确认它位于 root 之内，否则抛 ValueError。

    用于防路径穿越：凡是「由名字拼出来的路径」（如 logs/<任务名>/x.log）
    在只用前必须过一遍这里。符号链接会被 realpath 展开后比较，
    因此不能靠软链把读写引到目录之外。
    """
    root_resolved = Path(os.path.realpath(abs_path(root)))
    target = Path(os.path.realpath(abs_path(path)))
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"路径越界，拒绝访问 {root_resolved} 之外的 {target}")
    return target


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    mode: int | None = None,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
) -> None:
    """原子写入: 同目录临时文件 + fsync + rename，避免半截文件。

    allowed_root 非空时先做包含校验（防路径穿越）。凡是「路径里含拼接出来的
    名字」的调用方都应该传入它。临时文件由 mkstemp 在**已校验的 target 所在目录**
    内创建，因此后续 chmod / unlink / replace 都不会逃出该目录。
    """
    target = resolve_within(path, allowed_root) if allowed_root is not None else abs_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# 进程与归属（安全边界的核心依据）
# --------------------------------------------------------------------------- #
def current_uid() -> int:
    return os.getuid()


def proc_uid(pid: int) -> int | None:
    """读取进程 owner uid；进程不存在或无权读取时返回 None。"""
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        return None
    except OSError:
        return None


def is_own_process(pid: int) -> bool:
    """该 pid 是否是当前用户拥有的、且确实存活的进程。"""
    if pid <= 0:
        return False
    uid = proc_uid(pid)
    if uid is None:
        return False
    return uid == current_uid()


def process_state(pid: int) -> str | None:
    """读取 /proc/<pid>/stat 的状态字符（R/S/D/Z/T ...）；读不到返回 None。

    'Z' 表示僵尸：进程已结束，仅等父进程回收退出码。
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except OSError:
        return None
    idx = data.rfind(")")
    if idx < 0:
        return None
    fields = data[idx + 2:].split()
    return fields[0] if fields else None


def process_alive(pid: int) -> bool:
    """pid 是否仍在运行。

    注意两点：
      1. 僵尸进程（Z，已结束但父进程属未 reap）不算「存活」——
         os.kill(pid, 0) 对僵尸会成功，若不排除会得到错误的「还在跑」判断。
      2. EPERM 表示存在但不属于我们 → 仍算存活（但我们永远不会对它发信号）。
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno != errno.EPERM:
            return False
    return process_state(pid) != "Z"


def process_start_time(pid: int) -> float | None:
    """进程启动时间（秒，boot 后基于 /proc/<pid>/stat 的 starttime）。

    用于防止 PID 复用导致的误判: 保存 (pid, start_time) 后再次比对，
    若 start_time 不同则说明 pid 已被复用，绝不是我们启动的那个进程。
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as fh:
            data = fh.read()
    except OSError:
        return None
    # comm 字段可能含空格/括号，从最后一个 ')' 之后开始切分
    idx = data.rfind(")")
    if idx < 0:
        return None
    fields = data[idx + 2:].split()
    # fields[0] == state; starttime 是第 22 个字段，切分后索引 19
    if len(fields) < 20:
        return None
    try:
        starttime_ticks = int(fields[19])
    except ValueError:
        return None
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError):
        hz = 100
    try:
        with open("/proc/stat", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("btime"):
                    btime = int(line.split()[1])
                    return btime + starttime_ticks / hz
    except OSError:
        pass
    return None


def safe_terminate(
    pid: int,
    grace_seconds: float = 20.0,
    kill_after_grace: bool = True,
    *,
    poll_interval: float = 0.2,
    stop_event: threading.Event | None = None,
) -> str:
    """只对「当前用户拥有的」进程组执行优雅停止，必要时升级 SIGKILL。

    返回: "not_owned" | "gone" | "term_sent" | "killed"
    绝不向其他用户的进程发送任何 signal。

    grace 等待用 wait_until（谓词式轮询）：条件一满足就立刻返回，
    传入 stop_event 时 daemon 关闭可以立即打断等待而不必等满 grace。
    """
    if not process_alive(pid):
        return "gone"
    if not is_own_process(pid):
        return "not_owned"

    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError as exc:
        # 用 errno 显式分派：不用多个 except 子句（广的 OSError 与窄子类混在一起容易
        # 让人误以为后面子句不可达/截断），同时也兼容只带 errno 的裸 OSError。
        if isinstance(exc, ProcessLookupError) or exc.errno == errno.ESRCH:
            return "gone"
        if isinstance(exc, PermissionError) or exc.errno == errno.EPERM:
            return "not_owned"
        # 其它 OSError（例如 pid 不是进程组组长）：退化为只停主进程
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return "gone"

    if wait_until(
        lambda: not process_alive(pid),
        grace_seconds,
        interval=poll_interval,
        stop_event=stop_event,
    ):
        return "term_sent"

    if not kill_after_grace:
        return "term_sent"
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    return "killed"


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def build_logger(
    name: str,
    log_file: str | os.PathLike[str] | None = None,
    level: str = "INFO",
    rotate_mb: int = 20,
    backups: int = 5,
    console: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if log_file:
        path = abs_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=max(1, rotate_mb) * 1024 * 1024, backupCount=max(1, backups),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if console or not log_file:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #
def parse_gpu_list(raw: Any) -> list[int]:
    """解析 GPU 列表配置: "all" | [0,1] | "0,1,2" | "0-3"。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        body = raw.strip().lower()
        if body in ("", "none"):
            return []
        if body == "all":
            return []
        raw = [x for x in body.replace(";", ",").split(",") if x.strip()]
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[int] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        if "-" in text and not text.startswith("-"):
            a, _, b = text.partition("-")
            if a.strip().isdigit() and b.strip().isdigit():
                start, end = int(a), int(b)
                step = 1 if end >= start else -1
                out.extend(range(start, end + step, step))
                continue
        if text.isdigit():
            out.append(int(text))
    # 去重并升序，保证可解释
    return sorted(set(out))


def uniq_keep_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def gpu_ids_text(gpu_ids: Iterable[int]) -> str:
    ids = sorted(int(g) for g in gpu_ids)
    return ",".join(str(g) for g in ids) if ids else "-"


def truncate(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit} chars)"
