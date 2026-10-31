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


# --------------------------------------------------------------------------- #
# 终态记录清理（queue.finished_retention_hours）
# --------------------------------------------------------------------------- #
def test_purge_finished_removes_only_expired_terminal_tasks(cfg):
    """只删超窗的终态记录；pending/running 与未超窗的终态都必须留下。"""
    qm = make_qm(cfg)
    now = 1_700_000_000.0
    old_done = qm.add(spec(name="old_done"))
    qm.mark_finished(old_done, status=COMPLETED, exit_code=0)
    old_done.finished_at = now - 100 * 3600
    old_cancelled = qm.add(spec(name="old_cancelled"))
    qm.mark_finished(old_cancelled, status=CANCELLED, exit_code=None)
    old_cancelled.finished_at = now - 90 * 3600
    recent = qm.add(spec(name="recent_done"))
    qm.mark_finished(recent, status=FAILED, exit_code=1, error="boom")
    recent.finished_at = now - 3600
    pending = qm.add(spec(name="still_pending"))
    running = qm.add(spec(name="still_running"))
    qm.mark_running(running, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")

    removed = qm.purge_finished(before=now - 72 * 3600)

    assert sorted(removed) == ["old_cancelled", "old_done"]
    assert qm.get("old_done") is None and qm.get("old_cancelled") is None
    assert qm.get("recent_done") is not None
    assert qm.get("still_pending") is not None and qm.get("still_running") is not None
    assert pending.name in qm.tasks


def test_purge_finished_respects_protect_list(cfg):
    """调用方仍在引用的任务即使超窗也不删（避免状态机引用已删除对象）。"""
    qm = make_qm(cfg)
    now = 1_700_000_000.0
    task = qm.add(spec(name="stopping"))
    qm.mark_finished(task, status=CANCELLED, exit_code=None)
    task.finished_at = now - 100 * 3600

    assert qm.purge_finished(before=now - 72 * 3600, protect=["stopping"]) == []
    assert qm.get("stopping") is not None


def test_purge_finished_falls_back_to_created_at(cfg):
    """旧状态文件里终态任务可能没有 finished_at → 用 created_at 兜底判断。"""
    qm = make_qm(cfg)
    now = 1_700_000_000.0
    task = qm.add(spec(name="legacy"))
    qm.mark_finished(task, status=COMPLETED, exit_code=0)
    task.finished_at = None
    task.created_at = now - 200 * 3600

    assert qm.purge_finished(before=now - 72 * 3600) == ["legacy"]
    assert qm.get("legacy") is None


# --------------------------------------------------------------------------- #
# 已结束任务的「墓碑」：记录被 retention 删掉后不得被 tasks.yaml 重新导入
#
# 回归测试: 曾经 purge 掉一条已完成记录后，tasks.yaml 里那条没写 status 的旧条目
# 会被当成**新提交**重新入队，于是同一个实验被无限重训（实测浪费 171 GPU·小时）。
# --------------------------------------------------------------------------- #
def test_purged_task_is_not_reimported_from_queue_file(cfg, workdir):
    qm = make_qm(cfg)
    now = 1_700_000_000.0
    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), command="python main.py --epochs 100")])
    qm.import_queue_file()
    done = qm.require("exp")
    qm.mark_finished(done, status=COMPLETED, exit_code=0)
    done.finished_at = now - 100 * 3600

    assert qm.purge_finished(before=now - 72 * 3600) == ["exp"]
    assert qm.get("exp") is None, "记录已删除（list 看不到）"

    result = qm.import_queue_file()

    assert result.added == [], "同一条定义不得被当成新提交"
    assert result.archived == ["exp"]
    assert qm.get("exp") is None
    assert "exp" in qm.finished, "墓碑必须记住它跑过"


def test_purged_task_reruns_only_when_definition_changes(cfg, workdir):
    """想重跑同一个实验 → 改 command（新的实验参数）或换名字，而不是改 priority。"""
    qm = make_qm(cfg)
    now = 1_700_000_000.0
    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), command="python main.py --epochs 100")])
    qm.import_queue_file()
    done = qm.require("exp")
    qm.mark_finished(done, status=COMPLETED, exit_code=0)
    done.finished_at = now - 100 * 3600
    qm.purge_finished(before=now - 72 * 3600)

    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), command="python main.py --epochs 100 --seed 1")])
    result = qm.import_queue_file()

    assert result.added == ["exp"]
    assert qm.require("exp").status == PENDING
    assert "exp" not in qm.finished, "真的重跑了，墓碑要清掉"


def test_finished_task_cosmetic_edit_does_not_requeue(cfg, workdir):
    """priority / notes 只影响排队与展示，改了不该让一个已完成实验重跑。"""
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), priority=10, notes="v1")])
    qm.import_queue_file()
    done = qm.require("exp")
    qm.mark_finished(done, status=COMPLETED, exit_code=0)

    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), priority=99, notes="改个备注")])
    result = qm.import_queue_file()

    assert result.replaced == [] and result.added == []
    assert qm.require("exp").status == COMPLETED, "终态不得被改写回 pending"
    assert qm.require("exp").priority == 99, "展示字段仍然更新"
    assert qm.require("exp").notes == "改个备注"


def test_finished_task_real_edit_still_requeues(cfg, workdir):
    """真改了 command 才算「新的实验」——这是文档承诺的行为，保留。"""
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), command="python v1.py")])
    qm.import_queue_file()
    done = qm.require("exp")
    qm.mark_finished(done, status=CANCELLED, exit_code=None)

    write_tasks(cfg, [spec(name="exp", cwd=str(workdir), command="python v2.py")])
    result = qm.import_queue_file()

    assert result.replaced == ["exp"]
    assert qm.require("exp").status == PENDING
    assert qm.require("exp").command == "python v2.py"


def test_finished_archive_survives_state_roundtrip(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="exp"))
    qm.mark_finished(task, status=COMPLETED, exit_code=0)
    qm.purge_finished(before=task.finished_at + 1)

    state: dict[str, Any] = {}
    qm.dump(state)
    restored = make_qm(cfg)
    restored.load(state)

    assert "exp" in restored.finished
    assert restored.finished["exp"]["status"] == COMPLETED


def test_finished_archive_respects_max_entries(cfg):
    cfg.queue.archive_max_entries = 3
    qm = make_qm(cfg)
    for i in range(5):
        task = qm.add(spec(name=f"t{i}"))
        qm.mark_finished(task, status=COMPLETED, exit_code=0)
        task.finished_at = 1000.0 + i
        qm.purge_finished(before=10_000.0)

    assert len(qm.finished) == 3
    assert sorted(qm.finished) == ["t2", "t3", "t4"], "留下最新的 3 条"


def test_remove_terminal_task_leaves_tombstone(cfg, workdir):
    """CLI remove 掉一条终态任务后，tasks.yaml 里的同名旧条目也不得把它复活。"""
    qm = make_qm(cfg)
    write_tasks(cfg, [spec(name="exp", cwd=str(workdir))])
    qm.import_queue_file()
    done = qm.require("exp")
    qm.mark_finished(done, status=COMPLETED, exit_code=0)

    qm.remove("exp")
    result = qm.import_queue_file()

    assert result.added == [] and result.archived == ["exp"]
    assert qm.get("exp") is None


def test_explicit_add_clears_tombstone(cfg, workdir):
    """显式重新 add 是「我确实要再跑一遍」→ 墓碑必须清掉。"""
    qm = make_qm(cfg)
    task = qm.add(spec(name="exp"))
    qm.mark_finished(task, status=CANCELLED, exit_code=None)
    qm.purge_finished(before=task.finished_at + 1)
    assert "exp" in qm.finished

    qm.add(spec(name="exp"))

    assert "exp" not in qm.finished
    assert qm.require("exp").status == PENDING


# --------------------------------------------------------------------------- #
# 暂停调度（停止后重新排队）
# --------------------------------------------------------------------------- #
def test_request_pause_marks_running_tasks_only(cfg):
    qm = make_qm(cfg)
    running = qm.add(spec(name="run1"))
    qm.mark_running(running, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")
    pending = qm.add(spec(name="wait1"))

    marked = qm.request_pause()

    assert marked == ["run1"]
    assert running.stop_requested and running.pause_requested
    assert running.stop_reason == "pause"
    assert not pending.stop_requested and not pending.pause_requested


def test_requeue_after_pause_keeps_seq_and_retries_but_clears_runtime(cfg):
    qm = make_qm(cfg)
    task = qm.add(spec(name="t1", priority=7))
    seq = task.seq
    created = task.created_at
    qm.mark_running(task, gpu_ids=[0, 1], pid=123, pgid=123, proc_start_time=5.0, log_file="/l.log")
    task.retries = 2
    task.progress = {"epoch|500": [{"done": 10}]}
    task.stage = "training"
    task.pause_requested = True
    qm.mark_finished(task, status=COMPLETED, exit_code=0)

    qm.requeue_after_pause(task)

    assert task.status == PENDING
    assert task.seq == seq and task.created_at == created, "暂停不该让任务插队或重置提交时间"
    assert task.retries == 2, "暂停不是任务的错，不消耗重试次数"
    assert task.pid is None and task.gpu_ids == [] and task.exit_code is None
    assert task.finished_at is None and task.started_at is None
    assert task.progress == {} and task.stage == "", "重跑从第 0 轮开始，旧进度必须清掉"
    assert not task.pause_requested and not task.stop_requested


def test_cancel_clears_pause_intent(cfg):
    """暂停期间用户改主意取消 → 不应该再被当成「暂停后重排队」。"""
    qm = make_qm(cfg)
    task = qm.add(spec(name="t1"))
    qm.mark_running(task, gpu_ids=[0], pid=1, pgid=1, proc_start_time=None, log_file="")
    qm.request_pause()
    assert task.pause_requested

    qm.cancel("t1")
    assert not task.pause_requested
    assert task.stop_requested and task.stop_reason == "cancel"
