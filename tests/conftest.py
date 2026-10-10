"""pytest 公共 fixture。

刻意用「写一个真实 config.yaml → load_config」的路径构造配置，
这样测试同时也覆盖了配置校验逻辑本身。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu_manager.config import Config, load_config  # noqa: E402
from gpu_manager.utils import resolve_within, wait_until  # noqa: E402


@pytest.fixture
def wait_for():
    """谓词轮询等待（测试统一入口）。

    为什么不用固定 `time.sleep(0.3)`：条件一满足就返回，测试更快，
    也不会因机器慢而失败。实现直接复用生产代码的
    gpu_manager.utils.wait_until，保证测试与 daemon 用的是**同一套**等待语义。
    """

    def _wait(predicate, timeout: float = 10.0) -> bool:
        return wait_until(predicate, timeout, interval=0.02)

    return _wait


def _base_config(**gpu_overrides) -> dict:
    gpu = {
        "max_idle_utilization": 5,
        "max_idle_memory_mb": 1000,
        "require_no_compute_process": True,
        "check_interval_seconds": 30,
        "idle_confirm_checks": 3,
        "monitor_gpus": "all",
        "scheduler_allowed_gpus": "all",
        "reserved_gpus": [],
    }
    gpu.update(gpu_overrides)
    return {
        "server": {"name": "testhost", "user": "tester"},
        "gpu": gpu,
        "scheduler": {"max_concurrent_tasks": 0, "launch_second_check": True},
        "notify": {"on_gpu_idle": True, "on_task_queued": True},
        "feishu": {"enabled": False},
        "runner": {"term_grace_seconds": 1, "kill_after_grace": True},
        "paths": {
            "state_file": "data/state.json",
            "log_file": "data/manager.log",
            "event_log": "data/events.jsonl",
            "task_log_dir": "logs",
            "pid_file": "data/manager.pid",
            "socket_file": "data/manager.sock",
            "queue_file": "tasks.yaml",
        },
        "task_defaults": {"gpu_num": 1, "priority": 0},
    }


def write_config(tmp_path: Path, **gpu_overrides) -> Config:
    raw = _base_config(**gpu_overrides)
    # 显式确认写入目标在 tmp_path 之内（测试隔离，同时演示路径包含校验的用法）
    path = resolve_within(tmp_path / "config.yaml", tmp_path)
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的文件，且已 resolve_within 校验
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    resolve_within(tmp_path / "tasks.yaml", tmp_path).write_text("tasks: []\n", encoding="utf-8")
    return load_config(path)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return write_config(tmp_path)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    """写出一份临时 config.yaml 并返回其路径。

    供需要**真实启动子进程**的测试（后台 start/stop 端到端）使用；
    配置里的相对路径都落在 tmp_path 下，不会污染项目的 data/ 与 logs/。
    """
    write_config(tmp_path)
    return tmp_path / "config.yaml"


@pytest.fixture
def make_cfg(tmp_path: Path):
    def _make(**gpu_overrides) -> Config:
        return write_config(tmp_path, **gpu_overrides)

    return _make


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """一个存在的任务工作目录（任务 cwd 必须存在才会通过校验）。"""
    d = tmp_path / "project"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def own_pid() -> int:
    """当前进程 pid —— 在 /proc 里的 owner 就是当前用户，可用于测「自己的进程」。"""
    return os.getpid()
