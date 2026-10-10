"""调度决策。

刻意做成**纯函数**：不碰 GPU、不碰进程、不写文件，因此可以脱离真实显卡单测。

默认策略 ``priority_first_fit``:
  1. 按 priority 降序排序（同优先级 FIFO，由 QueueManager.pending_tasks 保证）
  2. 每个任务从可用池里取「编号最小的 N 张卡」
  3. 已分配给前面任务的卡从池中移除（不超卖）
  4. 大任务资源不够时**跳过它继续看后面的任务**，不阻塞整个队列

这一点很重要: TaskA 需要 4 卡而当前只有 3 卡空闲时，若队列后面有只需 1 卡的
TaskC，就先跑 TaskC。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .task_queue import Task


@dataclass(frozen=True)
class GpuCandidate:
    """可调度候选卡。free_mb 用于满足任务的显存下限要求。"""

    gpu_id: int
    free_mb: int = 0


@dataclass
class Assignment:
    task: Task
    gpu_ids: list[int]


@dataclass
class Plan:
    assignments: list[Assignment] = field(default_factory=list)
    waiting: list[tuple[str, str]] = field(default_factory=list)
    blocked_by_concurrency: list[str] = field(default_factory=list)

    @property
    def total_gpus(self) -> int:
        return sum(len(a.gpu_ids) for a in self.assignments)

    def waiting_text(self, limit: int = 5) -> str:
        if not self.waiting:
            return "无"
        lines = [f"  {name}: {reason}" for name, reason in self.waiting[:limit]]
        if len(self.waiting) > limit:
            lines.append(f"  ...另有 {len(self.waiting) - limit} 个任务在等待")
        return "\n".join(lines)


def plan_assignments(
    pending: list[Task],
    claimable: list[GpuCandidate],
    *,
    running_count: int = 0,
    max_concurrent: int = 0,
    now: float = 0.0,
    allow_cpu_tasks: bool = False,
) -> Plan:
    """生成本轮启动计划。

    pending 必须是已排序的待执行任务（priority 降序 + FIFO）。
    """
    plan = Plan()
    pool = sorted(claimable, key=lambda c: c.gpu_id)
    slots_left = None if max_concurrent <= 0 else max(0, max_concurrent - running_count)
    concurrency_full = False

    for task in pending:
        if task.retry_not_before and task.retry_not_before > now:
            remaining = int(task.retry_not_before - now)
            plan.waiting.append((task.name, f"失败重试等待中（约 {remaining}s 后）"))
            continue

        if slots_left is not None and slots_left <= 0:
            concurrency_full = True
            plan.blocked_by_concurrency.append(task.name)
            continue

        # --- 纯 CPU 任务 ---
        if task.gpu_num == 0:
            if not allow_cpu_tasks:
                plan.waiting.append((task.name, "gpu_num=0 的 CPU 任务未启用（scheduler.allow_cpu_tasks=false）"))
                continue
            plan.assignments.append(Assignment(task=task, gpu_ids=[]))
            if slots_left is not None:
                slots_left -= 1
            continue

        # --- 显存下限过滤 ---
        eligible = [c for c in pool if c.free_mb >= task.min_free_memory_mb]
        if len(eligible) < task.gpu_num:
            reason = f"需要 {task.gpu_num} 张确认空闲卡，当前可用 {len(eligible)}"
            if task.min_free_memory_mb:
                reason += f"（且单卡空闲显存 >= {task.min_free_memory_mb}MiB）"
            plan.waiting.append((task.name, reason))
            continue

        chosen = eligible[: task.gpu_num]
        pool = [c for c in pool if c.gpu_id not in {c2.gpu_id for c2 in chosen}]
        plan.assignments.append(Assignment(task=task, gpu_ids=sorted(c.gpu_id for c in chosen)))
        if slots_left is not None:
            slots_left -= 1

    if concurrency_full:
        # 并发额度已满时，其余任务不再逐条给理由，避免通知噪声
        plan.waiting = [(n, r) for n, r in plan.waiting if n not in set(plan.blocked_by_concurrency)]

    return plan


def describe_plan(plan: Plan) -> str:
    if not plan.assignments:
        return "本轮无可启动任务"
    parts = [f"{a.task.name}→GPU {','.join(str(g) for g in a.gpu_ids) or 'CPU'}" for a in plan.assignments]
    return "本轮计划: " + "; ".join(parts)
