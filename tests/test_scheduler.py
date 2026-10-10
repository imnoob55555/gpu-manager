"""调度决策测试。

核心行为: priority + first fit，且**大任务资源不够时不阻塞整个队列**。
"""

from __future__ import annotations

from gpu_manager.scheduler import GpuCandidate, describe_plan, plan_assignments
from gpu_manager.task_queue import Task


def make_task(name: str, gpu_num: int, priority: int = 0, seq: int = 0, **kw) -> Task:
    """构造任务。调度器是纯函数，不会访问文件系统，因此 cwd 只作占位。"""
    return Task(
        name=name,
        cwd="/path/to/project",
        command="python x.py",
        gpu_num=gpu_num,
        priority=priority,
        seq=seq,
        **kw,
    )


def pending(tasks: list[Task]) -> list[Task]:
    """模拟 QueueManager.pending_tasks(): priority 降序 + FIFO。"""
    return sorted(tasks, key=lambda t: (-t.priority, t.seq, t.name))


def cands(*ids: int, free_mb: int = 40000) -> list[GpuCandidate]:
    return [GpuCandidate(gpu_id=i, free_mb=free_mb) for i in ids]


def test_priority_order_high_first():
    tasks = pending([make_task("low", 1, priority=1, seq=0), make_task("high", 1, priority=10, seq=1)])
    plan = plan_assignments(tasks, cands(4))
    assert [a.task.name for a in plan.assignments] == ["high"]
    assert plan.assignments[0].gpu_ids == [4]


def test_spec_example_big_task_does_not_block_queue():
    """例子: A 需要 4 卡、B 需要 2 卡、C 需要 1 卡，当前只有 3 卡空闲 → 跑 B 和 C。"""
    tasks = pending(
        [
            make_task("A", 4, priority=10, seq=0),
            make_task("B", 2, priority=5, seq=1),
            make_task("C", 1, priority=5, seq=2),
        ]
    )
    plan = plan_assignments(tasks, cands(4, 5, 6))
    assert [a.task.name for a in plan.assignments] == ["B", "C"]
    assert plan.assignments[0].gpu_ids == [4, 5]
    assert plan.assignments[1].gpu_ids == [6]
    reasons = dict(plan.waiting)
    assert "A" in reasons and "需要 4 张" in reasons["A"]


def test_enough_gpus_lets_big_task_run():
    tasks = pending([make_task("A", 4, priority=10, seq=0), make_task("C", 1, seq=1)])
    plan = plan_assignments(tasks, cands(3, 4, 5, 6, 7))
    assert [a.task.name for a in plan.assignments] == ["A", "C"]
    assert plan.assignments[0].gpu_ids == [3, 4, 5, 6]
    assert plan.assignments[1].gpu_ids == [7]


def test_gpus_are_never_double_allocated():
    tasks = pending([make_task("a", 2, seq=0), make_task("b", 2, seq=1), make_task("c", 2, seq=2)])
    plan = plan_assignments(tasks, cands(0, 1, 2, 3))
    used: list[int] = []
    for assignment in plan.assignments:
        used.extend(assignment.gpu_ids)
    assert len(used) == len(set(used)), "同一张卡不能分配给两个任务"
    assert sorted(used) == [0, 1, 2, 3]
    assert [a.task.name for a in plan.assignments] == ["a", "b"]
    assert dict(plan.waiting)["c"].startswith("需要 2 张")


def test_first_fit_picks_lowest_indices():
    plan = plan_assignments(pending([make_task("t", 2)]), cands(2, 5, 7))
    assert plan.assignments[0].gpu_ids == [2, 5]


def test_max_concurrent_tasks_limit():
    tasks = pending([make_task("a", 1, seq=0), make_task("b", 1, seq=1), make_task("c", 1, seq=2)])
    plan = plan_assignments(tasks, cands(0, 1, 2), running_count=0, max_concurrent=2)
    assert [a.task.name for a in plan.assignments] == ["a", "b"]
    assert plan.blocked_by_concurrency == ["c"]


def test_max_concurrent_accounts_for_running_tasks():
    tasks = pending([make_task("a", 1, seq=0)])
    plan = plan_assignments(tasks, cands(0, 1), running_count=2, max_concurrent=2)
    assert plan.assignments == []
    assert plan.blocked_by_concurrency == ["a"]


def test_min_free_memory_filters_candidates():
    tasks = pending([make_task("big", 1, min_free_memory_mb=40000)])
    plan = plan_assignments(tasks, [GpuCandidate(0, 10000), GpuCandidate(1, 45000)])
    assert plan.assignments[0].gpu_ids == [1]

    plan_none = plan_assignments(tasks, [GpuCandidate(0, 10000)])
    assert plan_none.assignments == []
    assert "显存" in dict(plan_none.waiting)["big"]


def test_min_free_memory_multi_gpu_needs_all_slots():
    tasks = pending([make_task("m", 2, min_free_memory_mb=30000)])
    plan = plan_assignments(
        tasks, [GpuCandidate(0, 45000), GpuCandidate(1, 20000), GpuCandidate(2, 44000)]
    )
    assert plan.assignments[0].gpu_ids == [0, 2]


def test_retry_backoff_task_is_skipped_but_others_run():
    now = 1000.0
    tasks = pending(
        [
            make_task("retrying", 1, priority=10, seq=0, retry_not_before=now + 60),
            make_task("fresh", 1, priority=1, seq=1),
        ]
    )
    plan = plan_assignments(tasks, cands(0), now=now)
    assert [a.task.name for a in plan.assignments] == ["fresh"]
    assert "重试等待" in dict(plan.waiting)["retrying"]


def test_cpu_task_disabled_by_default():
    plan = plan_assignments(pending([make_task("cpu", 0)]), [])
    assert plan.assignments == []
    assert "CPU 任务未启用" in dict(plan.waiting)["cpu"]


def test_cpu_task_allowed_does_not_consume_gpu():
    plan = plan_assignments(pending([make_task("cpu", 0)]), cands(0), allow_cpu_tasks=True)
    assert plan.assignments[0].gpu_ids == []


def test_empty_inputs():
    assert plan_assignments([], cands(0, 1)).assignments == []
    plan = plan_assignments(pending([make_task("t", 1)]), [])
    assert plan.assignments == []
    assert "当前可用 0" in dict(plan.waiting)["t"]


def test_describe_plan_is_human_readable():
    plan = plan_assignments(pending([make_task("t", 1)]), cands(3))
    text = describe_plan(plan)
    assert "t" in text and "3" in text
    empty = plan_assignments([], [])
    assert "无可启动" in describe_plan(empty)


def test_waiting_text_limits_output():
    tasks = pending([make_task(f"t{i}", 2, seq=i) for i in range(8)])
    plan = plan_assignments(tasks, cands(0))
    text = plan.waiting_text(limit=3)
    assert "另有 5 个任务在等待" in text
