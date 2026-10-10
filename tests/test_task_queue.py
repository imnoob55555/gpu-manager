"""任务队列：校验、幂等导入、状态流转。

关键安全/正确性属性:
  - tasks.yaml 导入幂等，重复导入不会重复入队
  - 运行中的任务绝不被文件改动打断（不偷偷杀实验）
  - 取消运行中任务只置 stop_requested，由 daemon 负责优雅停止
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import yaml

from gpu_manager.task_queue import (
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    QueueManager,
    TaskError,
    build_task,
    validate_cwd,
)


def make_qm(cfg) -> QueueManager:
    return QueueManager(cfg, logging.getLogger("test.queue"))


def spec(
    name: Any = "t1",
    command: Any = "python main.py",
    cwd: Any = "/path/to/project",
    **kw: Any,
) -> dict[str, Any]:
    """构造**未校验**的原始任务定义。

    参数故意放宽类型：很多用例就是要喂非法值（列表命令、超长名字、负数卡数）
    来验证 build_task 会拒绝它们。
    """
    base: dict[str, Any] = {"name": name, "command": command, "cwd": cwd}
    base.update(kw)
    return base


def write_tasks(cfg, tasks: list[dict]) -> None:
    cfg.queue_file.write_text(
        yaml.safe_dump({"tasks": tasks}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def test_build_task_minimal_uses_defaults(cfg):
    task = build_task(spec(), cfg.task_defaults, cfg)
    assert task.name == "t1"
    assert task.gpu_num == cfg.task_defaults.gpu_num
    assert task.priority == 0
    assert task.status == PENDING
    assert task.signature, "必须生成内容签名（用于幂等导入）"


@pytest.mark.parametrize(
    "bad_name",
    ["", "  ", "has space", "with/slash", "-leading-dash", "a" * 65, "名字"],
)
def test_invalid_names_rejected(cfg, bad_name):
    with pytest.raises(TaskError):
        build_task(spec(name=bad_name), cfg.task_defaults, cfg)


@pytest.mark.parametrize("good_name", ["dino_baseline", "exp-1", "a.b.c", "A1"])
def test_valid_names_accepted(cfg, good_name):
    assert build_task(spec(name=good_name), cfg.task_defaults, cfg).name == good_name


def test_command_and_cwd_required(cfg):
    with pytest.raises(TaskError, match="command"):
        build_task(spec(command=""), cfg.task_defaults, cfg)
    with pytest.raises(TaskError, match="cwd"):
        build_task(spec(cwd=""), cfg.task_defaults, cfg)


def test_command_as_list_rejected_with_helpful_message(cfg):
    with pytest.raises(TaskError, match="多行命令"):
        build_task(spec(command=["python", "main.py"]), cfg.task_defaults, cfg)


def test_validate_cwd(workdir, tmp_path, cfg):
    task = build_task(spec(cwd=str(workdir)), cfg.task_defaults, cfg)
    assert validate_cwd(task) is None

    missing = build_task(spec(cwd=str(tmp_path / "nope")), cfg.task_defaults, cfg)
    assert "不存在" in (validate_cwd(missing) or "")

    as_file = tmp_path / "afile"
    as_file.write_text("x", encoding="utf-8")
    not_dir = build_task(spec(cwd=str(as_file)), cfg.task_defaults, cfg)
    assert "不是目录" in (validate_cwd(not_dir) or "")


def test_gpu_num_zero_requires_explicit_opt_in(cfg):
    with pytest.raises(TaskError, match="allow_cpu_tasks"):
        build_task(spec(gpu_num=0), cfg.task_defaults, cfg)
    cfg.scheduler.allow_cpu_tasks = True
    assert build_task(spec(gpu_num=0), cfg.task_defaults, cfg).gpu_num == 0


def test_negative_gpu_num_rejected(cfg):
    with pytest.raises(TaskError, match="gpu_num"):
        build_task(spec(gpu_num=-1), cfg.task_defaults, cfg)


def test_env_as_string(cfg):
    task = build_task(spec(env="undetr"), cfg.task_defaults, cfg)
    assert task.env == "undetr"
    assert task.env_vars == {}


def test_env_as_dict_with_name_and_vars(cfg):
    task = build_task(spec(env={"name": "undetr", "vars": {"A": 1}}), cfg.task_defaults, cfg)
    assert task.env == "undetr"
    assert task.env_vars == {"A": "1"}


def test_env_as_plain_vars_dict(cfg):
    task = build_task(spec(env={"WANDB_MODE": "offline"}), cfg.task_defaults, cfg)
    assert task.env == ""
    assert task.env_vars == {"WANDB_MODE": "offline"}


def test_env_vars_field_merged(cfg):
    task = build_task(spec(env_vars={"X": "1", "Y": 2}), cfg.task_defaults, cfg)
    assert task.env_vars == {"X": "1", "Y": "2"}


def test_defaults_env_applied(cfg):
    cfg.task_defaults.env = "undetr"
    assert build_task(spec(), cfg.task_defaults, cfg).env == "undetr"


# --------------------------------------------------------------------------- #
# 入队 / 变更
# --------------------------------------------------------------------------- #
def test_add_assigns_increasing_seq(cfg):
    qm = make_qm(cfg)
    first = qm.add(spec(name="a"), source="cli")
    second = qm.add(spec(name="b"), source="cli")
    assert second.seq > first.seq


def test_add_duplicate_pending_rejected(cfg):
    qm = make_qm(cfg)
    qm.add(spec(name="a"))
    with pytest.raises(TaskError, match="已存在"):
        qm.add(spec(name="a"))


def test_add_duplicate_running_rejected_with_hint(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.mark_running(task, gpu_ids=[1], pid=1234, pgid=1234, proc_start_time=None, log_file="/x.log")
    with pytest.raises(TaskError, match="正在运行"):
        qm.add(spec(name="a"), replace=True)


def test_replace_terminal_task_allowed(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a", command="python v1.py"))
    qm.mark_finished(task, status=COMPLETED, exit_code=0)
    new = qm.add(spec(name="a", command="python v2.py"), replace=True)
    assert new.command == "python v2.py"
    assert new.status == PENDING
    assert new.signature != task.signature


def test_signature_ignores_status_and_timestamps(cfg):
    """签名只反映「提交内容」，不因运行状态变化而改变（否则会反复重新入队）。"""
    a = build_task(spec(), cfg.task_defaults, cfg)
    b = build_task(spec(), cfg.task_defaults, cfg)
    a.status = RUNNING
    a.pid = 999
    assert a.signature == b.signature


def test_signature_changes_when_content_changes(cfg):
    a = build_task(spec(), cfg.task_defaults, cfg)
    for field, value in (("command", "python other.py"), ("gpu_num", 4), ("priority", 9)):
        b = build_task(spec(**{field: value}), cfg.task_defaults, cfg)
        assert a.signature != b.signature, field


def test_pending_order_priority_desc_then_fifo(cfg):
    qm = make_qm(cfg)
    qm.add(spec(name="low", priority=1))
    qm.add(spec(name="high_first", priority=10))
    qm.add(spec(name="high_second", priority=10))
    assert [t.name for t in qm.pending_tasks()] == ["high_first", "high_second", "low"]


def test_busy_gpu_ids_and_counts(cfg):
    qm = make_qm(cfg)
    a = qm.add(spec(name="a", gpu_num=2))
    b = qm.add(spec(name="b"))
    qm.mark_running(b, gpu_ids=[5], pid=1, pgid=1, proc_start_time=None, log_file="/l")
    assert qm.busy_gpu_ids() == {5}
    counts = qm.counts()
    assert counts[PENDING] == 1 and counts[RUNNING] == 1 and counts["total"] == 2
    assert a.is_pending


# --------------------------------------------------------------------------- #
# 导入 tasks.yaml
# --------------------------------------------------------------------------- #
def test_import_adds_tasks(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir)), spec(name="b", cwd=str(workdir))])
    result = qm.import_queue_file()
    assert sorted(result.added) == ["a", "b"]
    assert len(qm.tasks) == 2


def test_import_is_idempotent(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir))])
    first = qm.import_queue_file()
    assert first.added == ["a"]
    second = qm.import_queue_file()
    assert second.changed is False, "内容没变时不得重复入队"
    assert second.skipped == ["a"]
    assert len(qm.tasks) == 1


def test_import_replaces_when_definition_changes(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), command="python v1.py")])
    qm.import_queue_file()
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), command="python v2.py")])
    result = qm.import_queue_file()
    assert result.replaced == ["a"]
    assert qm.require("a").command == "python v2.py"
    assert len(qm.tasks) == 1


def test_import_never_interrupts_running_task(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), command="python v1.py")])
    qm.import_queue_file()
    task = qm.require("a")
    qm.mark_running(task, gpu_ids=[3], pid=4242, pgid=4242, proc_start_time=None, log_file="/l.log")

    write_tasks(cfg, [spec(name="a", cwd=str(workdir), command="python v2.py")])
    result = qm.import_queue_file()

    assert result.replaced_running == ["a"]
    assert result.replaced == []
    current = qm.require("a")
    assert current.command == "python v1.py", "运行中的任务定义不得被改写"
    assert current.status == RUNNING


def test_import_cancel_directive_cancels_pending(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir))])
    qm.import_queue_file()
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), status=CANCELLED)])
    result = qm.import_queue_file()
    assert result.cancelled == ["a"]
    assert qm.require("a").status == CANCELLED


def test_import_cancel_directive_only_requests_for_running(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir))])
    qm.import_queue_file()
    task = qm.require("a")
    qm.mark_running(task, gpu_ids=[0], pid=777, pgid=777, proc_start_time=None, log_file="/l")

    write_tasks(cfg, [spec(name="a", cwd=str(workdir), status=CANCELLED)])
    qm.import_queue_file()

    assert task.status == RUNNING
    assert task.stop_requested is True, "只置停止请求，由 daemon 优雅停止"


def test_import_disabled_directive_skipped(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), status="disabled")])
    result = qm.import_queue_file()
    assert result.skipped == ["a"]
    assert qm.tasks == {}


def test_import_invalid_status_reported(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="a", cwd=str(workdir), status="weird")])
    result = qm.import_queue_file()
    assert result.errors and "非法" in result.errors[0]
    assert qm.tasks == {}


def test_import_invalid_task_collects_error(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="ok", cwd=str(workdir)), spec(name="bad name", cwd=str(workdir))])
    result = qm.import_queue_file()
    assert result.added == ["ok"]
    assert result.errors, "非法任务名必须报告，而不是静默丢弃"


def test_import_bad_yaml_does_not_raise(cfg, workdir):
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="keep", cwd=str(workdir))])
    qm.import_queue_file()

    cfg.queue_file.write_text("tasks: [ {name: 'x'\n  broken", encoding="utf-8")
    result = qm.import_queue_file()
    assert result.errors
    assert len(qm.tasks) == 1, "YAML 损坏时保留上一次有效状态，不影响调度"


def test_import_top_level_list_supported(cfg, workdir):
    qm = make_qm(cfg)
    cfg.queue_file.write_text(
        yaml.safe_dump([spec(name="a", cwd=str(workdir))], allow_unicode=True), encoding="utf-8"
    )
    assert qm.import_queue_file().added == ["a"]


def test_import_missing_file_is_noop(cfg):
    qm = make_qm(cfg)
    cfg.queue_file.unlink()
    assert qm.import_queue_file().changed is False


# --------------------------------------------------------------------------- #
# 取消 / 移除 / 重新入队
# --------------------------------------------------------------------------- #
def test_cancel_pending_sets_cancelled(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.cancel("a")
    assert task.status == CANCELLED
    assert task.finished_at is not None


def test_cancel_running_only_requests_stop(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.mark_running(task, gpu_ids=[2], pid=555, pgid=555, proc_start_time=None, log_file="/l")
    qm.cancel("a")
    assert task.status == RUNNING, "取消运行中任务不能立即改状态，要等进程真正退出"
    assert task.stop_requested is True


def test_cancel_terminal_rejected(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.mark_finished(task, status=COMPLETED, exit_code=0)
    with pytest.raises(TaskError, match="终态"):
        qm.cancel("a")


def test_remove_refuses_running(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.mark_running(task, gpu_ids=[1], pid=9, pgid=9, proc_start_time=None, log_file="/l")
    with pytest.raises(TaskError, match="正在运行"):
        qm.remove("a")
    qm.mark_finished(task, status=CANCELLED, exit_code=None)
    qm.remove("a")
    assert "a" not in qm.tasks


def test_requeue_resets_runtime_fields(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a"))
    qm.mark_running(task, gpu_ids=[4], pid=42, pgid=42, proc_start_time=1.0, log_file="/l")
    qm.mark_finished(task, status=FAILED, exit_code=1, error="boom")
    qm.requeue("a")
    assert task.status == PENDING
    assert task.gpu_ids == []
    assert task.pid is None
    assert task.exit_code is None
    assert task.error == ""
    assert task.finished_at is None


def test_dump_and_load_roundtrip(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="a", gpu_num=2, priority=7))
    qm.mark_running(task, gpu_ids=[1, 2], pid=123, pgid=123, proc_start_time=5.0, log_file="/l.log")
    state: dict = {}
    qm.dump(state)

    restored = make_qm(cfg)
    restored.load(state)
    other = restored.require("a")
    assert other.gpu_ids == [1, 2]
    assert other.pid == 123
    assert other.status == RUNNING
    assert other.priority == 7
    assert restored.next_seq == qm.next_seq


def test_load_skips_corrupt_entries(cfg):
    qm = make_qm(cfg)
    qm.load({"tasks": [{"name": "ok", "cwd": "/x", "command": "c"}, {"nope": 1}, "junk"], "next_seq": 5})
    assert list(qm.tasks) == ["ok"]
    assert qm.next_seq == 5
