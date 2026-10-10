"""通用工具测试：进程语义、信号安全边界、配置解析辅助。

这里锁定一条容易踩坑的语义:
  「进程存活」必须排除僵尸（Z）。os.kill(pid, 0) 对僵尸仍会成功，
  若不排除，取消任务后会一直以为它「还在跑」，日志会反复刷 SIGKILL。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

import pytest

from gpu_manager.utils import (
    atomic_write_text,
    build_logger,
    fmt_duration,
    gpu_ids_text,
    is_own_process,
    parse_gpu_list,
    proc_uid,
    process_alive,
    process_start_time,
    process_state,
    resolve_within,
    safe_terminate,
    truncate,
    uniq_keep_order,
    wait_until,
)


def foreign_pid() -> int | None:
    for pid in (1, 2, 3):
        uid = proc_uid(pid)
        if uid is not None and uid != os.getuid():
            return pid
    return None


# --------------------------------------------------------------------------- #
# resolve_within: 路径包含校验（防穿越）
# --------------------------------------------------------------------------- #
def test_resolve_within_accepts_path_inside_root(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    assert resolve_within(root / "job" / "x.log", root) == (root / "job" / "x.log")
    assert resolve_within(root, root) == root, "根目录自身也应通过"


def test_resolve_within_rejects_path_outside_root(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    with pytest.raises(ValueError, match="越界"):
        resolve_within(tmp_path / "elsewhere" / "x.log", root)
    with pytest.raises(ValueError, match="越界"):
        resolve_within("/etc/passwd", root)


def test_resolve_within_rejects_symlink_escape(tmp_path):
    """软链不能把读写引到目录之外（realpath 展开后再比较）。"""
    root = tmp_path / "logs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="越界"):
        resolve_within(link / "secret.txt", root)


def test_resolve_within_rejects_dotdot(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    with pytest.raises(ValueError, match="越界"):
        resolve_within(root / ".." / "passwd", root)


# --------------------------------------------------------------------------- #
# wait_until: 全项目唯一的轮询原语
# --------------------------------------------------------------------------- #
def test_wait_until_returns_as_soon_as_condition_holds():
    calls = {"n": 0}

    def predicate() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    assert wait_until(predicate, 5.0, interval=0.01) is True
    assert calls["n"] == 3, "条件成立后不应再多轮询一次"


def test_wait_until_returns_false_on_timeout():
    start = time.monotonic()
    assert wait_until(lambda: False, 0.2, interval=0.05) is False
    assert time.monotonic() - start < 2.0, "必须在超时后很快返回"


def test_wait_until_stop_event_interrupts_immediately():
    """daemon 关闭时不应傻等满 grace 周期。"""
    stop_event = threading.Event()
    threading.Timer(0.05, stop_event.set).start()
    start = time.monotonic()
    assert wait_until(lambda: False, 30.0, interval=0.05, stop_event=stop_event) is False
    assert time.monotonic() - start < 5.0, "stop_event 应立即打断等待"


def test_wait_until_zero_timeout_still_checks_once():
    assert wait_until(lambda: True, 0.0) is True
    assert wait_until(lambda: False, 0.0) is False


# --------------------------------------------------------------------------- #
# atomic_write_text: 可选包含校验
# --------------------------------------------------------------------------- #
def test_atomic_write_text_with_allowed_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    target = root / "state.json"
    atomic_write_text(target, "ok", allowed_root=root)
    assert target.read_text(encoding="utf-8") == "ok"
    # 不留临时文件
    assert sorted(p.name for p in root.iterdir()) == ["state.json"]


def test_atomic_write_text_rejects_outside_allowed_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(ValueError, match="越界"):
        atomic_write_text(tmp_path / "evil.json", "x", allowed_root=root)
    assert not (tmp_path / "evil.json").exists(), "越界写入必须被拒绝且不留文件"


# --------------------------------------------------------------------------- #
# 进程语义
# --------------------------------------------------------------------------- #
def test_process_alive_true_for_self():
    assert process_alive(os.getpid())
    assert is_own_process(os.getpid())
    assert process_state(os.getpid()) in ("R", "S", "D")


def test_process_alive_false_for_nonexistent_pid():
    assert not process_alive(4194300)


def test_process_alive_false_for_zombie():
    """子进程退出但未被回收 = 僵尸：不应算作「还在运行」。"""
    if not hasattr(os, "waitid"):  # pragma: no cover - 非 Linux
        pytest.skip("该平台不支持 os.waitid(WNOWAIT)")
    proc = subprocess.Popen(["true"])
    # WNOWAIT: 阻塞等到它退出但**不回收**，从而确定性地处于僵尸态
    # （不需要 sleep 轮询，也不依赖调度时序）
    info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
    assert info is not None and info.si_pid == proc.pid
    assert process_state(proc.pid) == "Z", "应处于僵尸态"
    assert not process_alive(proc.pid), "僵尸必须判为不存活"
    assert proc.wait() == 0  # 真正回收
    assert not process_alive(proc.pid), "回收后 pid 应彻底消失"


def test_process_start_time_is_stable_for_self():
    first = process_start_time(os.getpid())
    second = process_start_time(os.getpid())
    assert first is not None and first == second, "同一进程的 starttime 必须稳定"
    assert first <= time.time()


def test_process_start_time_none_for_dead_pid():
    assert process_start_time(4194300) is None


def test_is_own_process_false_for_foreign():
    pid = foreign_pid()
    if pid is None:
        pytest.skip("找不到其他用户的进程")
    assert not is_own_process(pid)
    assert proc_uid(pid) != os.getuid()


# --------------------------------------------------------------------------- #
# safe_terminate: 安全边界
# --------------------------------------------------------------------------- #
def test_safe_terminate_refuses_foreign_process():
    pid = foreign_pid()
    if pid is None:
        pytest.skip("找不到其他用户的进程")
    assert safe_terminate(pid, grace_seconds=0) == "not_owned"
    assert process_alive(pid)


def test_safe_terminate_gone_for_dead_pid():
    assert safe_terminate(4194300, grace_seconds=0) == "gone"


def test_safe_terminate_terminates_own_process_group(tmp_path, wait_for):
    ready = tmp_path / "ready"
    # 子进程先落地就绪标记再 exec，保证「信号到达时它已在运行」是确定的，而非靠 sleep 猜
    proc = subprocess.Popen(
        ["bash", "-c", f"touch {shlex.quote(str(ready))}; exec sleep 60"],
        start_new_session=True,
    )
    try:
        assert wait_for(ready.exists), "子进程未就绪"
        assert safe_terminate(proc.pid, grace_seconds=3.0) in ("term_sent", "killed")
        assert proc.wait(timeout=5) is not None
        assert not process_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_safe_terminate_escalates_to_sigkill(tmp_path, wait_for):
    """忽略 SIGTERM 的进程必须在 grace 之后被 SIGKILL 收掉。"""
    ready = tmp_path / "ready"
    # trap '' TERM → SIG_IGN；忽略属性在 exec 后依然保留（POSIX），
    # 因此 exec sleep 后它仍会忽略 SIGTERM，从而必然触发升级到 SIGKILL。
    script = f"trap '' TERM; touch {shlex.quote(str(ready))}; exec sleep 60"
    proc = subprocess.Popen(["bash", "-c", script], start_new_session=True)
    try:
        assert wait_for(ready.exists), "SIG_IGN 尚未生效"
        assert safe_terminate(proc.pid, grace_seconds=0.5, kill_after_grace=True) == "killed"
        assert proc.wait(timeout=5) == -9, "应被 SIGKILL 收掉"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# --------------------------------------------------------------------------- #
# 配置辅助
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("all", []),
        (None, []),
        ("", []),
        ([0, 1, 2], [0, 1, 2]),
        ("0,1,2", [0, 1, 2]),
        ("4-7", [4, 5, 6, 7]),
        (["4", "5"], [4, 5]),
        ([3, 1, 3], [1, 3]),
        (["7"], [7]),
        ("3, 1", [1, 3]),
    ],
)
def test_parse_gpu_list(raw, expected):
    assert parse_gpu_list(raw) == expected


def test_atomic_write_text_creates_file_with_mode(tmp_path):
    target = tmp_path / "sub" / "state.json"
    atomic_write_text(target, '{"a": 1}', mode=0o600)
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert oct(target.stat().st_mode & 0o777) == "0o600"
    # 覆盖写
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    # 不残留临时文件
    leftovers = [p.name for p in target.parent.iterdir() if p.name != "state.json"]
    assert leftovers == []


@pytest.mark.parametrize(
    "seconds,expected",
    [(None, "-"), (-1, "-"), (5, "5s"), (65, "1m5s"), (3700, "1h1m40s"), (90000, "1d1h0m")],
)
def test_fmt_duration(seconds, expected):
    assert fmt_duration(seconds) == expected


def test_gpu_ids_text():
    assert gpu_ids_text([3, 6]) == "3,6"
    assert gpu_ids_text([]) == "-"


def test_truncate_and_uniq():
    assert truncate("abc") == "abc"
    assert "…" not in truncate("abc")
    short = truncate("x" * 100, 10)
    assert len(short) < 40 and "chars" in short
    assert uniq_keep_order([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_logger_creates_rotating_file(tmp_path):
    log_file = tmp_path / "logs" / "x.log"
    logger = build_logger("test.logger.rot", log_file, console=False)
    logger.warning("hello")
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")


def test_paths_with_spaces_are_handled(tmp_path):
    """路径含空格/引号时，run.sh 与日志仍必须正确（trap 用变量传路径）。"""
    weird = tmp_path / "dir with space" / "and'quote"
    weird.mkdir(parents=True)
    target = weird / "state.json"
    atomic_write_text(target, "ok")
    assert Path(target).read_text(encoding="utf-8") == "ok"
