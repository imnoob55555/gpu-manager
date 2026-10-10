"""任务定义、校验与队列管理。

职责边界（重要）:
  - ``tasks.yaml`` 是**提交入口**（只读，程序绝不改写它，因此不会与你的手改冲突）。
    其中 ``status`` 只是提交指令: ``pending``(默认) / ``cancelled`` / ``disabled``。
  - ``data/state.json`` 是**权威运行状态**（running/completed/failed/cancelled + pid/gpu/时间戳）。
    ``gpu-manager list`` 展示的就是它。
  - 同一任务名在 tasks.yaml 中内容发生变化（签名不同）会被视为一次新提交并重新入队；
    若该任务正在运行，则拒绝替换并提示先 ``cancel``（绝不偷偷杀掉正在跑的实验）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Config, TaskDefaults
from .utils import abs_path, gpu_ids_text, uniq_keep_order

RUNNING = "running"
PENDING = "pending"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

ACTIVE_STATUSES = frozenset({PENDING, RUNNING})
TERMINAL_STATUSES = frozenset({COMPLETED, FAILED, CANCELLED})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES

#: tasks.yaml 中 status 允许的取值（其余视为非法）
FILE_DIRECTIVES = frozenset({PENDING, CANCELLED, "disabled"})

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

SIGNATURE_FIELDS = (
    "command", "cwd", "env", "env_vars", "gpu_num", "priority",
    "min_free_memory_mb", "max_retries", "timeout_seconds",
)


class TaskError(ValueError):
    """任务定义错误。"""


# --------------------------------------------------------------------------- #
# 任务模型
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    name: str
    cwd: str
    command: str
    env: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    gpu_num: int = 1
    priority: int = 0
    min_free_memory_mb: int = 0
    max_retries: int = 0
    retry_delay_seconds: int = 60
    timeout_seconds: int = 0
    notes: str = ""

    # ---- 运行状态 ----
    status: str = PENDING
    seq: int = 0
    gpu_ids: list[int] = field(default_factory=list)
    pid: int | None = None
    pgid: int | None = None
    proc_start_time: float | None = None
    log_file: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str = ""
    retries: int = 0
    source: str = "cli"
    signature: str = ""
    stop_requested: bool = False
    stop_requested_at: float = 0.0
    stop_reason: str = ""
    retry_not_before: float = 0.0
    notify_wait_at: float = 0.0
    #: ETA 用的进度采样，按「阶段」分组: {阶段键: [样本, ...]}
    progress: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    progress_updated_at: float = 0.0
    #: 额外的进度日志文件（相对于 cwd 或绝对路径）。
    #: 训练脚本常把输出重定向到自己的日志（如 console.log），
    #: 此时管理器的横幅日志里看不到进度，需要在这里声明。
    eta_logs: list[str] = field(default_factory=list)
    #: 整轮总轮数（epoch）。用于把「第几轮 + 轮内进度」型日志换成整轮进度；
    #: 0 = 未知（此时只报轮次，不编造结束时间）
    total_epochs: int = 0
    #: 阶段标记文件（相对于 cwd 或绝对路径），例如 stage.status（内容 training/evaluating）
    stage_file: str = ""
    #: 当前阶段（由 stage_file 读取，用于展示并在阶段切换时令旧进度失效）
    stage: str = ""
    #: 上一次采样的进度来源文件及其「指纹」（mtime_ns:size）。
    #: 源文件没变化就不记新样本——否则日志里最后一行旧进度会被配上新时间戳，
    #: 在 eval 等阶段用训练速度外推，得出错误 ETA。
    progress_source: str = ""
    progress_source_sig: str = ""

    # ------------------------------------------------------------------ #
    @property
    def safe_name(self) -> str:
        """用于日志目录/文件名的安全名字。"""
        cleaned = SAFE_NAME_RE.sub("_", self.name).strip("._-") or "task"
        if cleaned != self.name:
            cleaned = f"{cleaned}-{self.signature[:6] or 'x'}"
        return cleaned

    @property
    def is_pending(self) -> bool:
        return self.status == PENDING

    @property
    def is_running(self) -> bool:
        return self.status == RUNNING

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def gpu_text(self) -> str:
        return gpu_ids_text(self.gpu_ids)

    def runtime_seconds(self, now: float | None = None) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else (now or time.time())
        return max(0.0, end - self.started_at)

    def wait_seconds(self, now: float | None = None) -> float:
        return max(0.0, (now or time.time()) - self.created_at)

    def compute_signature(self) -> str:
        """内容指纹，用于判断 tasks.yaml 中的提交是否发生变化（非安全用途，用 sha256）。"""
        payload = {k: getattr(self, k) for k in SIGNATURE_FIELDS}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cwd": self.cwd,
            "command": self.command,
            "env": self.env,
            "env_vars": dict(self.env_vars),
            "gpu_num": self.gpu_num,
            "priority": self.priority,
            "min_free_memory_mb": self.min_free_memory_mb,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "notes": self.notes,
            "status": self.status,
            "seq": self.seq,
            "gpu_ids": list(self.gpu_ids),
            "pid": self.pid,
            "pgid": self.pgid,
            "proc_start_time": self.proc_start_time,
            "log_file": self.log_file,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "retries": self.retries,
            "source": self.source,
            "signature": self.signature,
            "stop_requested": self.stop_requested,
            "stop_requested_at": self.stop_requested_at,
            "stop_reason": self.stop_reason,
            "retry_not_before": self.retry_not_before,
            "notify_wait_at": self.notify_wait_at,
            "progress": dict(self.progress),
            "progress_updated_at": self.progress_updated_at,
            "eta_logs": list(self.eta_logs),
            "total_epochs": self.total_epochs,
            "stage_file": self.stage_file,
            "stage": self.stage,
            "progress_source": self.progress_source,
            "progress_source_sig": self.progress_source_sig,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        task = cls(**payload)  # type: ignore[arg-type]
        if not isinstance(task.env_vars, dict):
            task.env_vars = {}
        if not isinstance(task.progress, dict):
            task.progress = {}
        if not isinstance(task.eta_logs, list):
            task.eta_logs = []
        if not isinstance(task.gpu_ids, list):
            task.gpu_ids = []
        task.gpu_ids = [int(g) for g in task.gpu_ids]
        if task.status not in ALL_STATUSES:
            task.status = PENDING
        if not task.signature:
            task.signature = task.compute_signature()
        return task


# --------------------------------------------------------------------------- #
# 校验与构造
# --------------------------------------------------------------------------- #
def _as_int(value: Any, where: str, default: int, *, minimum: int | None = None) -> int:
    if value is None or value == "":
        value = default
    if isinstance(value, bool):
        raise TaskError(f"{where} 必须是整数，实际是布尔值")
    try:
        out = int(str(value).strip())
    except (ValueError, TypeError) as exc:
        raise TaskError(f"{where} 必须是整数，实际是 {value!r}") from exc
    if minimum is not None and out < minimum:
        raise TaskError(f"{where} 必须 >= {minimum}，实际是 {out}")
    return out


def _as_str(value: Any, where: str, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        raise TaskError(f"{where} 必须是字符串，实际是列表（多行命令请用 YAML 的 > 或 | 块）")
    if not isinstance(value, str):
        raise TaskError(f"{where} 必须是字符串，实际是 {type(value).__name__}")
    return value


def _parse_env(value: Any, where: str) -> tuple[str, dict[str, str]]:
    """env 支持 conda 环境名字符串，或 {环境名: ..., vars: {...}} / 直接给环境变量字典。"""
    if value is None or value == "":
        return "", {}
    if isinstance(value, str):
        return value.strip(), {}
    if isinstance(value, dict):
        env_name = str(value.get("name", value.get("conda", "")) or "").strip()
        vars_raw = value.get("vars") or value.get("env_vars") or {}
        if not env_name and not vars_raw:
            # 视为纯环境变量表 {"CUDA_HOME": "..."}
            return "", {str(k): str(v) for k, v in value.items()}
        if not isinstance(vars_raw, dict):
            raise TaskError(f"{where}.vars 必须是映射")
        return env_name, {str(k): str(v) for k, v in vars_raw.items()}
    raise TaskError(f"{where} 必须是字符串或映射，实际是 {type(value).__name__}")


def build_task(spec: dict[str, Any], defaults: TaskDefaults, cfg: Config, source: str = "cli") -> Task:
    """把原始 dict 校验并转换为 Task。任何非法字段都抛 TaskError（错误信息面向用户）。"""
    if not isinstance(spec, dict):
        raise TaskError(f"任务定义必须是映射，实际是 {type(spec).__name__}")

    name = _as_str(spec.get("name"), "name").strip()
    if not name:
        raise TaskError("name 不能为空")
    if not NAME_RE.match(name):
        raise TaskError(
            f"任务名 {name!r} 非法：只允许字母/数字开头，后续为字母、数字、点、下划线、连字符，长度 1..64"
        )

    cwd_raw = _as_str(spec.get("cwd"), f"{name}.cwd").strip()
    if not cwd_raw:
        raise TaskError(f"{name}.cwd 不能为空（必须是绝对路径）")
    cwd = str(abs_path(cwd_raw))

    command = _as_str(spec.get("command"), f"{name}.command").strip()
    if not command:
        raise TaskError(f"{name}.command 不能为空")

    env_name, env_vars = _parse_env(spec.get("env"), f"{name}.env")
    if not env_name:
        env_name = defaults.env
    extra_vars = spec.get("env_vars")
    if isinstance(extra_vars, dict):
        env_vars.update({str(k): str(v) for k, v in extra_vars.items()})
    elif extra_vars is not None:
        raise TaskError(f"{name}.env_vars 必须是映射")

    gpu_num = _as_int(spec.get("gpu_num"), f"{name}.gpu_num", defaults.gpu_num, minimum=0)
    if gpu_num == 0 and not cfg.scheduler.allow_cpu_tasks:
        raise TaskError(
            f"{name}.gpu_num=0（纯 CPU 任务）需要显式开启 scheduler.allow_cpu_tasks: true，"
            "否则请指定 >= 1"
        )

    # ETA 相关: 额外进度日志 / 整轮总轮数 / 阶段文件
    eta_logs_raw = spec.get("eta_logs")
    if eta_logs_raw is None:
        eta_logs: list[str] = []
    elif isinstance(eta_logs_raw, (list, tuple)):
        eta_logs = [str(p).strip() for p in eta_logs_raw if str(p).strip()]
    else:
        raise TaskError(f"{name}.eta_logs 必须是路径列表，例如 [console.log, log.txt]")

    task = Task(
        name=name,
        cwd=cwd,
        command=command,
        env=env_name,
        env_vars=env_vars,
        gpu_num=gpu_num,
        priority=_as_int(spec.get("priority"), f"{name}.priority", defaults.priority),
        min_free_memory_mb=_as_int(
            spec.get("min_free_memory_mb"), f"{name}.min_free_memory_mb", defaults.min_free_memory_mb, minimum=0
        ),
        max_retries=_as_int(spec.get("max_retries"), f"{name}.max_retries", defaults.max_retries, minimum=0),
        retry_delay_seconds=_as_int(
            spec.get("retry_delay_seconds"), f"{name}.retry_delay_seconds", defaults.retry_delay_seconds, minimum=0
        ),
        timeout_seconds=_as_int(
            spec.get("timeout_seconds"), f"{name}.timeout_seconds", defaults.timeout_seconds, minimum=0
        ),
        notes=_as_str(spec.get("notes"), f"{name}.notes"),
        eta_logs=eta_logs,
        total_epochs=_as_int(spec.get("total_epochs"), f"{name}.total_epochs", 0, minimum=0),
        stage_file=_as_str(spec.get("stage_file"), f"{name}.stage_file").strip(),
        source=source,
    )
    task.signature = task.compute_signature()
    return task


def validate_cwd(task: Task) -> str | None:
    """返回错误说明；None 表示工作目录可用。"""
    path = Path(task.cwd)
    if not path.exists():
        return f"工作目录不存在: {task.cwd}"
    if not path.is_dir():
        return f"工作目录不是目录: {task.cwd}"
    if not os.access(path, os.R_OK | os.X_OK):
        return f"工作目录无读取/进入权限: {task.cwd}"
    return None


# --------------------------------------------------------------------------- #
# 导入结果
# --------------------------------------------------------------------------- #
@dataclass
class ImportResult:
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    replaced_running: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.replaced or self.cancelled)

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"新增 {len(self.added)}: {', '.join(self.added)}")
        if self.replaced:
            parts.append(f"重新入队 {len(self.replaced)}: {', '.join(self.replaced)}")
        if self.cancelled:
            parts.append(f"按文件取消 {len(self.cancelled)}: {', '.join(self.cancelled)}")
        if self.replaced_running:
            parts.append(f"运行中拒绝替换 {len(self.replaced_running)}: {', '.join(self.replaced_running)}")
        if self.errors:
            parts.append(f"错误 {len(self.errors)}: {'; '.join(self.errors)}")
        if self.skipped:
            parts.append(f"跳过 {len(self.skipped)}: {', '.join(self.skipped)}")
        return " | ".join(parts) if parts else "无变化"


# --------------------------------------------------------------------------- #
# 队列
# --------------------------------------------------------------------------- #
class QueueManager:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.tasks: dict[str, Task] = {}
        self.next_seq: int = 1

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def load(self, state: dict[str, Any]) -> None:
        self.tasks = {}
        self.next_seq = int(state.get("next_seq", 1) or 1)
        raw_tasks = state.get("tasks") or []
        if not isinstance(raw_tasks, list):
            self.logger.error("state.tasks 不是列表，忽略（状态可能已损坏）")
            return
        for item in raw_tasks:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                task = Task.from_dict(item)
            except (TypeError, ValueError) as exc:
                self.logger.warning("恢复任务失败: %s (%s)", item.get("name"), exc)
                continue
            self.tasks[task.name] = task
        self.next_seq = max(self.next_seq, max((t.seq for t in self.tasks.values()), default=0) + 1)

    def dump(self, state: dict[str, Any]) -> None:
        state["tasks"] = [t.to_dict() for t in self.ordered_tasks()]
        state["next_seq"] = self.next_seq

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Task | None:
        return self.tasks.get(name)

    def require(self, name: str) -> Task:
        task = self.tasks.get(name)
        if task is None:
            raise TaskError(f"任务不存在: {name}")
        return task

    def ordered_tasks(self) -> list[Task]:
        """稳定顺序: 先按 seq（提交顺序）。"""
        return sorted(self.tasks.values(), key=lambda t: (t.seq, t.name))

    def pending_tasks(self) -> list[Task]:
        """调度顺序: priority 降序，同优先级 FIFO。"""
        pending = [t for t in self.tasks.values() if t.is_pending]
        return sorted(pending, key=lambda t: (-t.priority, t.seq, t.name))

    def running_tasks(self) -> list[Task]:
        return [t for t in self.ordered_tasks() if t.is_running]

    def terminal_tasks(self) -> list[Task]:
        return [t for t in self.ordered_tasks() if t.is_terminal]

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(sorted(ALL_STATUSES), 0)
        for task in self.tasks.values():
            out[task.status] = out.get(task.status, 0) + 1
        out["total"] = len(self.tasks)
        return out

    def busy_gpu_ids(self) -> set[int]:
        """已被运行中任务占用的 GPU（防止重复分配）。"""
        busy: set[int] = set()
        for task in self.tasks.values():
            if task.is_running:
                busy.update(task.gpu_ids)
        return busy

    # ------------------------------------------------------------------ #
    # 变更
    # ------------------------------------------------------------------ #
    def _take_seq(self) -> int:
        seq = self.next_seq
        self.next_seq += 1
        return seq

    def add(self, spec: dict[str, Any], source: str = "cli", *, replace: bool = False) -> Task:
        task = build_task(spec, self.cfg.task_defaults, self.cfg, source)
        existing = self.tasks.get(task.name)
        if existing is not None:
            if existing.is_running:
                raise TaskError(
                    f"任务 {task.name} 正在运行（GPU {existing.gpu_text}）。"
                    f"如要替换请先 `gpu-manager cancel {task.name}`，程序不会自动杀掉运行中的实验"
                )
            if not replace and not existing.is_terminal:
                raise TaskError(f"任务已存在: {task.name}（当前状态 {existing.status}）")
        task.seq = self._take_seq()
        task.status = PENDING
        self.tasks[task.name] = task
        return task

    def cancel(self, name: str, *, force_terminal: bool = True) -> Task:
        """请求取消。运行中 → 标记 stop_requested 由 daemon 优雅停止；未运行 → 直接置 cancelled。"""
        task = self.require(name)
        if task.is_terminal:
            raise TaskError(f"任务已是终态，无需取消: {name} (status={task.status})")
        if task.is_running:
            task.stop_requested = True
            self.logger.info("任务 %s 已标记停止请求（daemon 将优雅终止其进程组）", name)
        else:
            task.status = CANCELLED
            task.finished_at = time.time()
            task.error = task.error or "用户取消"
            self.logger.info("任务 %s 已取消（尚未运行）", name)
        return task

    def remove(self, name: str) -> Task:
        task = self.require(name)
        if task.is_running:
            raise TaskError(f"任务正在运行，不能删除: {name}（先 cancel 等它结束）")
        del self.tasks[name]
        return task

    def requeue(self, name: str) -> Task:
        task = self.require(name)
        if task.is_running:
            raise TaskError(f"任务正在运行，不能重新入队: {name}")
        task.status = PENDING
        task.gpu_ids = []
        task.pid = None
        task.pgid = None
        task.proc_start_time = None
        task.exit_code = None
        task.error = ""
        task.started_at = None
        task.finished_at = None
        task.stop_requested = False
        task.retry_not_before = 0.0
        task.created_at = time.time()
        task.notify_wait_at = 0.0
        return task
    def mark_running(
        self,
        task: Task,
        *,
        gpu_ids: Iterable[int],
        pid: int,
        pgid: int | None,
        proc_start_time: float | None,
        log_file: str,
    ) -> None:
        task.status = RUNNING
        task.gpu_ids = sorted(int(g) for g in gpu_ids)
        task.pid = pid
        task.pgid = pgid
        task.proc_start_time = proc_start_time
        task.log_file = log_file
        task.started_at = time.time()
        task.finished_at = None
        task.exit_code = None
        task.error = ""
        task.stop_requested = False
        task.notify_wait_at = 0.0

    def mark_finished(self, task: Task, *, status: str, exit_code: int | None, error: str = "") -> None:
        if status not in TERMINAL_STATUSES:
            raise TaskError(f"非法终态: {status}")
        task.status = status
        task.exit_code = exit_code
        task.finished_at = time.time()
        task.error = error
        task.stop_requested = False

    # ------------------------------------------------------------------ #
    # 从 tasks.yaml 导入（幂等）
    # ------------------------------------------------------------------ #
    def read_queue_file(self, path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
        """读取 tasks.yaml；解析失败时抛 TaskError（daemon 会保留上一次有效状态）。"""
        queue_path = abs_path(path or self.cfg.queue_file)
        if not queue_path.exists():
            return []
        try:
            raw = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
            raise TaskError(f"读取 {queue_path} 失败: {exc}") from exc
        if isinstance(raw, list):
            specs = raw
        elif isinstance(raw, dict):
            specs = raw.get("tasks") or []
        else:
            raise TaskError(f"{queue_path} 顶层必须是 tasks 列表或带 tasks 键的映射")
        if not isinstance(specs, list):
            raise TaskError(f"{queue_path} 的 tasks 必须是列表")
        out: list[dict[str, Any]] = []
        for item in specs:
            if isinstance(item, dict):
                out.append(item)
            else:
                raise TaskError(f"{queue_path} 中存在非映射的任务条目: {item!r}")
        return out

    def import_queue_file(self, path: str | os.PathLike[str] | None = None) -> ImportResult:
        """把 tasks.yaml 的提交同步进队列。幂等：内容未变则不动。"""
        result = ImportResult()
        try:
            specs = self.read_queue_file(path)
        except TaskError as exc:
            result.errors.append(str(exc))
            self.logger.warning("%s", exc)
            return result

        for spec in specs:
            raw_name = str(spec.get("name", "")).strip()
            directive = str(spec.get("status", PENDING) or PENDING).strip().lower()
            if directive not in FILE_DIRECTIVES:
                result.errors.append(
                    f"{raw_name or '(未命名)'}.status={directive!r} 非法（tasks.yaml 只接受 "
                    f"{'/'.join(sorted(FILE_DIRECTIVES))}）"
                )
                continue
            if directive == "disabled":
                result.skipped.append(raw_name or "(未命名)")
                continue
            if not raw_name:
                result.errors.append("存在缺少 name 的任务条目")
                continue

            existing = self.tasks.get(raw_name)

            # 文件里显式取消
            if directive == CANCELLED:
                if existing is None or existing.is_terminal:
                    result.skipped.append(raw_name)
                    continue
                if existing.is_running:
                    existing.stop_requested = True
                else:
                    existing.status = CANCELLED
                    existing.finished_at = time.time()
                    existing.error = existing.error or "用户取消（tasks.yaml）"
                result.cancelled.append(raw_name)
                continue

            try:
                candidate = build_task(spec, self.cfg.task_defaults, self.cfg, source="tasks.yaml")
            except TaskError as exc:
                result.errors.append(str(exc))
                continue

            if existing is None:
                candidate.seq = self._take_seq()
                self.tasks[candidate.name] = candidate
                result.added.append(candidate.name)
                continue

            if existing.signature == candidate.signature:
                result.skipped.append(candidate.name)
                continue

            # 内容变了 → 新提交
            if existing.is_running:
                result.replaced_running.append(existing.name)
                self.logger.warning(
                    "tasks.yaml 中 %s 的定义已变化，但它正在运行（GPU %s）——保持不动。"
                    "需要生效请先 cancel 再重新入队",
                    existing.name, existing.gpu_text,
                )
                continue

            candidate.seq = self._take_seq() if existing.is_terminal else existing.seq
            self.tasks[candidate.name] = candidate
            result.replaced.append(candidate.name)

        return result

    # ------------------------------------------------------------------ #
    # 从 tasks.yaml 移除后的同步（文件里删掉的任务不会自动删状态，仅记录）
    # ------------------------------------------------------------------ #
    def names_in_queue_file(self, path: str | os.PathLike[str] | None = None) -> set[str]:
        try:
            return {str(s.get("name", "")).strip() for s in self.read_queue_file(path) if s.get("name")}
        except TaskError:
            return set()

    def unique_names(self, names: Iterable[str]) -> list[str]:
        return uniq_keep_order(names)
