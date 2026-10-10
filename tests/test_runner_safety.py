"""执行器与多用户安全边界测试。

这是本项目最重要的测试文件：它锁定「绝不误杀/误伤其他用户进程」这条硬边界。
守护机制:
  1. 发送 signal 前必须确认进程 owner 是当前用户
  2. 还要比对 /proc/<pid> 的 starttime，防止 PID 复用（pid 相同但不是当初那个进程）
  3. 只在自己的进程组内操作
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from gpu_manager.runner import RunnerError, TaskRunner
from gpu_manager.task_queue import Task
from gpu_manager.utils import (
    is_own_process,
    proc_uid,
    process_alive,
    process_start_time,
    wait_until,
)


def make_task(cwd: str, command: str = "echo hi", **kw) -> Task:
    """构造任务；gpu_num 默认 1，可用关键字覆盖（不要重复传入）。"""
    kw.setdefault("gpu_num", 1)
    return Task(name="t1", cwd=str(cwd), command=command, **kw)


def wait_exit(handle, timeout: float = 20.0) -> int | None:
    """等到进程退出并返回退出码（复用共享的 wait_until，不用固定 sleep）。"""
    wait_until(lambda: handle.poll() is not None, timeout, interval=0.02)
    return handle.poll()


def wait_all_dead(pids, handle=None, timeout: float = 10.0) -> bool:
    """等到指定 pid 全部消失。

    handle 非空时同时 poll() 回收子进程：僵尸态的 pid 仍然存在，
    不回收就会一直等不到（kill(pid,0) 对僵尸成功）。
    """
    wanted = list(pids)

    def done() -> bool:
        if handle is not None and handle.poll() is None:
            return False
        return all(not process_alive(pid) for pid in wanted)

    return wait_until(done, timeout, interval=0.05)


def foreign_pid() -> int | None:
    """找一个属于其他用户的可读进程（通常 pid 1 是 root）。找不到返回 None。"""
    if proc_uid(1) is not None and proc_uid(1) != os.getuid():
        return 1
    for pid in (1, 2, 3):
        uid = proc_uid(pid)
        if uid is not None and uid != os.getuid():
            return pid
    return None


# --------------------------------------------------------------------------- #
# 命令构造
# --------------------------------------------------------------------------- #
def test_build_script_plain_command(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    lines = runner.build_script(make_task(workdir, "python main.py --x 1"))
    assert lines == ["python main.py --x 1"]


def test_build_script_env_prefix_path(cfg, workdir, tmp_path):
    prefix = tmp_path / "envs" / "undetr"
    (prefix / "bin").mkdir(parents=True)
    runner = TaskRunner(cfg, pytest_logger())
    lines = runner.build_script(make_task(workdir, env=str(prefix)))
    joined = "\n".join(lines)
    assert f'export CONDA_PREFIX={prefix}' in joined or str(prefix) in joined
    assert "PATH" in joined


def test_build_script_rejects_invalid_env_prefix(cfg, workdir, tmp_path):
    runner = TaskRunner(cfg, pytest_logger())
    with pytest.raises(RunnerError, match="不是有效的"):
        runner.build_script(make_task(workdir, env=str(tmp_path / "no-bin-here")))


def test_build_script_with_conda_env_activates(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    if not runner.conda_bin():
        pytest.skip("环境里没有 conda")
    lines = runner.build_script(make_task(workdir, env="some_env"))
    joined = "\n".join(lines)
    assert "conda activate" in joined or "conda run" in joined


def test_describe_command_never_raises(cfg, workdir, tmp_path):
    runner = TaskRunner(cfg, pytest_logger())
    text = runner.describe_command(make_task(workdir, env=str(tmp_path / "bad")))
    assert "无法构造命令" in text


# --------------------------------------------------------------------------- #
# 身份校验：安全边界的核心
# --------------------------------------------------------------------------- #
def test_verify_identity_refuses_other_users_process(cfg, workdir):
    pid = foreign_pid()
    if pid is None:
        pytest.skip("找不到属于其他用户的进程（可能运行在容器内）")
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = pid
    ok, detail = runner.verify_identity(task)
    assert not ok
    assert "不是当前用户" in detail


def test_verify_identity_detects_pid_reuse(cfg, workdir, own_pid):
    """pid 还是那个 pid，但 starttime 对不上 → 说明已被复用，必须拒绝。"""
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = own_pid
    real = process_start_time(own_pid)
    assert real is not None
    task.proc_start_time = real - 9999.0  # 伪造一个不同的启动时间
    ok, detail = runner.verify_identity(task)
    assert not ok
    assert "PID 复用" in detail


def test_verify_identity_accepts_matching_own_process(cfg, workdir, own_pid):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = own_pid
    task.proc_start_time = process_start_time(own_pid)
    ok, detail = runner.verify_identity(task)
    assert ok, detail


def test_verify_identity_handles_missing_process(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = 4194303  # 几乎不可能存在的 pid
    ok, detail = runner.verify_identity(task)
    assert not ok
    assert "不存在" in detail or "owner" in detail


def test_verify_identity_without_pid(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    ok, detail = runner.verify_identity(make_task(workdir))
    assert not ok
    assert "无 pid" in detail


def test_send_stop_signal_refuses_foreign_process(cfg, workdir):
    """最重要的安全断言：绝不能向其他用户的进程发信号。"""
    pid = foreign_pid()
    if pid is None:
        pytest.skip("找不到属于其他用户的进程（可能运行在容器内）")
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = pid
    task.pgid = pid

    assert runner.send_stop_signal(task, hard=False) == "refused"
    assert runner.send_stop_signal(task, hard=True) == "refused"
    assert process_alive(pid), "拒绝之后该进程必须仍然存活"
    assert not is_own_process(pid)


def test_stop_refuses_when_pid_reused(cfg, workdir, own_pid):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.pid = own_pid
    task.pgid = own_pid
    task.proc_start_time = (process_start_time(own_pid) or 0.0) - 9999.0
    assert runner.send_stop_signal(task, hard=True) == "refused"
    assert process_alive(own_pid)
    assert runner.stop(task) == "refused", "同步 stop 也必须拒绝"


# --------------------------------------------------------------------------- #
# 真实启动 / 停止（不用 GPU 也能验证）
# --------------------------------------------------------------------------- #
def test_launch_rejects_gpu_count_mismatch(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir)
    task.gpu_num = 2
    with pytest.raises(RunnerError, match="GPU 数量不匹配"):
        runner.launch(task, [0])


def test_launch_sets_cuda_visible_devices_and_completes(cfg, workdir):
    """必须通过 CUDA_VISIBLE_DEVICES 限制可见 GPU。"""
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(
        workdir,
        command='printf "%s" "$CUDA_VISIBLE_DEVICES" > gpus.txt',
        gpu_num=2,
    )
    handle = runner.launch(task, [3, 6])
    try:
        assert wait_exit(handle) == 0
        assert handle.read_exit_code() == 0
        assert (Path(workdir) / "gpus.txt").read_text().strip() == "3,6"
    finally:
        handle.flush_close()


def test_launch_exports_manager_env_vars(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command='printf "%s|%s" "$GPU_MANAGER_TASK" "$GPU_MANAGER_GPU_IDS" > env.txt')
    handle = runner.launch(task, [5])
    try:
        assert wait_exit(handle) == 0
        assert (Path(workdir) / "env.txt").read_text().strip() == "t1|5"
    finally:
        handle.flush_close()


def test_launch_task_level_env_vars_override(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command='printf "%s" "$MY_FLAG" > flag.txt', env_vars={"MY_FLAG": "yes"})
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        assert (Path(workdir) / "flag.txt").read_text().strip() == "yes"
    finally:
        handle.flush_close()


@pytest.mark.parametrize(
    "command,expected",
    [
        ("exit 7", 7),
        ("set -e; false", 1),
        ("echo ok; exit 0", 0),
        ("python3 -c 'import sys; sys.exit(5)'", 5),
        ("false || exit 3", 3),
    ],
)
def test_exit_code_is_recorded_even_with_set_e_or_exit(cfg, workdir, command, expected):
    """回归测试: EXIT trap 保证 `set -e` / `exit N` 场景下退出码也不丢失。

    早期实现把记录语句放在脚本末尾，用户命令里一旦有 set -e 或 exit，
    脚本会提前结束，退出码文件永远不会写 → daemon 重启后无法判断成败。
    """
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command=command)
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == expected
        assert handle.read_exit_code() == expected, "退出码文件必须记录真实退出码"
    finally:
        handle.flush_close()


def test_launch_writes_banner_and_script(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="echo hello")
    handle = runner.launch(task, [1])
    try:
        assert wait_exit(handle) == 0
        log = Path(task.log_file).read_text(encoding="utf-8")
        assert "[gpu-manager] task      : t1" in log
        assert "CUDA_VISIBLE_DEVICES=1" in log
        assert "hello" in log
        script = handle.script_path.read_text(encoding="utf-8")
        assert "trap" in script and "EXIT" in script
        assert handle.script_path.stat().st_mode & 0o700 == 0o700
    finally:
        handle.flush_close()


def test_launch_creates_latest_symlink(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="true")
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        latest = handle.log_path.parent / "latest.log"
        assert latest.is_symlink()
        assert latest.resolve() == handle.log_path.resolve()
    finally:
        handle.flush_close()


def test_launch_uses_task_cwd(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="pwd > where.txt")
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        assert (Path(workdir) / "where.txt").read_text().strip() == str(workdir)
    finally:
        handle.flush_close()


def test_launch_fails_on_missing_cwd(cfg, tmp_path):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(tmp_path / "does-not-exist", command="true")
    with pytest.raises(RunnerError):
        runner.launch(task, [0])


def test_stop_terminates_own_running_task(cfg, workdir):
    """取消自己的任务: SIGTERM 应能在 grace 期内结束它。"""
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="sleep 120")
    handle = runner.launch(task, [2])
    try:
        assert handle.poll() is None, "sleep 应仍在运行"
        task.pid = handle.pid
        task.pgid = handle.pgid
        task.proc_start_time = handle.proc_start_time

        assert runner.stop(task, reason="测试") in ("term_sent", "killed")
        # 必须 poll() 回收子进程，否则它会以僵尸形式保留 pid
        assert wait_all_dead([handle.pid], handle), "自己的任务应被停止并回收"
        assert not process_alive(handle.pid)
    finally:
        handle.flush_close()


def test_stop_kills_children_in_process_group(cfg, workdir):
    """子进程（如 DataLoader worker）也必须一并停止 —— 靠独立进程组 + killpg。"""
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="sleep 120 & echo $! > child.pid; sleep 120")
    handle = runner.launch(task, [0])
    try:
        child_file = Path(workdir) / "child.pid"
        assert wait_until(child_file.exists, 5.0, interval=0.02), "子进程未写出 pid 文件"
        child_pid = int(child_file.read_text().strip())
        assert process_alive(child_pid)

        task.pid = handle.pid
        task.pgid = handle.pgid
        task.proc_start_time = handle.proc_start_time
        runner.stop(task)

        assert wait_all_dead([handle.pid, child_pid], handle), (
            "主进程与同一进程组内的子进程都应被停止"
        )
        assert not process_alive(handle.pid)
        assert not process_alive(child_pid)
    finally:
        handle.flush_close()


def test_stop_is_noop_for_already_dead_task(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="true")
    handle = runner.launch(task, [0])
    assert wait_exit(handle) == 0
    task.pid = handle.pid
    task.pgid = handle.pgid
    task.proc_start_time = handle.proc_start_time
    assert runner.stop(task) in ("gone", "term_sent", "killed")
    handle.flush_close()


def test_rotate_log_when_exceeding_limit(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="printf 'x%.0s' $(seq 1 200000)")
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        assert handle.log_size_mb() > 0.05
        assert handle.rotate_if_needed(0) is False, "0 表示不限制"
        assert handle.rotate_if_needed(0.01) is True
        assert handle.log_path.with_suffix(".log.1").exists()
        assert handle.log_path.exists()
    finally:
        handle.flush_close()


def test_tail_log(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="echo line1; echo line2")
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        tail = runner.tail_log(task, 5)
        assert "line2" in "\n".join(tail)
    finally:
        handle.flush_close()


# --------------------------------------------------------------------------- #
# tail_path: 读取用户声明的日志（ETA 用）
# --------------------------------------------------------------------------- #
def test_tail_path_reads_small_file_completely(tmp_path):
    """回归: 文件小于尾窗时，不能因为「丢弃截断首行」而把全部内容丢掉。

    小日志（刚启动的训练）与 stage.status（几十字节）都靠这条。
    """
    path = tmp_path / "console.log"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    path.write_text("Epoch: [141]  [10/63]\n", encoding="utf-8")
    assert TaskRunner.tail_path(path, 200) == ["Epoch: [141]  [10/63]"]
    assert TaskRunner.tail_path(path, 1) == ["Epoch: [141]  [10/63]"]


def test_tail_path_reads_single_line_without_newline(tmp_path):
    path = tmp_path / "stage.status"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    path.write_text("training", encoding="utf-8")  # 无换行
    assert TaskRunner.tail_path(path, 1) == ["training"]


def test_tail_path_on_large_file_keeps_complete_tail_lines(tmp_path):
    path = tmp_path / "big.log"
    lines = [f"line-{i:05d}" for i in range(5000)]
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tail = TaskRunner.tail_path(path, 10)

    assert len(tail) == 10
    assert tail[-1] == "line-04999"
    assert all(re.fullmatch(r"line-\d{5}", item) for item in tail), "不能留下被截断的半行"


def test_tail_path_handles_bad_targets(tmp_path):
    assert TaskRunner.tail_path(tmp_path, 10) == [], "目录不是普通文件"
    assert TaskRunner.tail_path(tmp_path / "missing.log", 10) == []


# --------------------------------------------------------------------------- #
# 日志读取安全边界
# --------------------------------------------------------------------------- #
def test_tail_log_refuses_path_outside_task_log_dir(cfg, workdir, tmp_path):
    """日志读取必须限定在任务日志目录内（防任意文件读取）。"""
    runner = TaskRunner(cfg, pytest_logger())
    secret = tmp_path / "secret.log"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    secret.write_text("top-secret\n", encoding="utf-8")
    task = make_task(workdir)
    task.log_file = str(secret)
    assert runner.tail_log(task, 5) == [], "不得读取日志目录之外的任何文件"


def test_tail_log_reads_file_inside_task_log_dir(cfg, workdir):
    runner = TaskRunner(cfg, pytest_logger())
    log_dir = cfg.task_log_dir / "t1"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    # pi-lens-ignore: python-path-traversal -- 路径位于 cfg.task_log_dir 内
    log_file.write_text("line1\nline2\n", encoding="utf-8")
    task = make_task(workdir)
    task.log_file = str(log_file)
    assert runner.tail_log(task, 1) == ["line2"]


def test_launch_keeps_artifacts_inside_task_log_dir(cfg, workdir):
    """真实启动后，日志/脚本/退出码文件必须都落在 cfg.task_log_dir 内。"""
    runner = TaskRunner(cfg, pytest_logger())
    task = make_task(workdir, command="echo hi")
    handle = runner.launch(task, [0])
    try:
        assert wait_exit(handle) == 0
        assert handle.log_path.parent == handle.script_path.parent
        assert handle.exit_path.parent == handle.log_path.parent
        assert handle.log_path.parent.parent == cfg.task_log_dir, "日志目录必须是 task_log_dir 的直接子目录"
    finally:
        handle.flush_close()


# --------------------------------------------------------------------------- #
def pytest_logger():
    import logging

    logger = logging.getLogger("test.runner")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger
