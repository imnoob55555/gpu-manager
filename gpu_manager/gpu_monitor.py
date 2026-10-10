"""GPU 状态采集与空闲判定。

设计原则:
  - **单次采集即快照**（Snapshot），避免同一轮决策里读到不一致的 GPU 状态。
  - 采集失败时保守处理: 所有 GPU 视为「不可用」，绝不因读取失败而误判空闲并抢占。
  - compute process 与 graphics process(Xorg) 严格区分: Xorg 不占用计算资源，
    不算「明显 compute process」。
  - 进程归属通过 /proc/<pid> 的 uid 判断，这是「不占用他人 GPU」的硬依据。
"""

from __future__ import annotations

import csv
import io
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .utils import current_uid, gpu_ids_text, proc_uid

CSV_GPU_QUERY = (
    "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,persistence_mode"
)
CSV_PROC_QUERY = "gpu_uuid,pid,process_name,used_gpu_memory"

_NA_VALUES = {"", "n/a", "[n/a]", "na", "[not supported]", "not supported", "unknown", "[unknown]"}


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text.lower() in _NA_VALUES:
        return None
    # nvidia-smi 可能返回 "49140 MiB" 或 "49140"
    for unit in ("Mib", "MiB", "MB", "W", "%", "C", "MiB "):
        if text.endswith(unit):
            text = text[: -len(unit)].strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def _split_csv_line(line: str) -> list[str]:
    """nvidia-smi CSV 输出中 process_name 可能含逗号，故用 csv 模块解析。"""
    try:
        row = next(csv.reader(io.StringIO(line), skipinitialspace=True))
    except (StopIteration, csv.Error):
        return []
    return [c.strip() for c in row]


@dataclass
class ComputeProcess:
    """GPU 上的计算进程。"""

    pid: int
    name: str
    memory_mb: int | None = None
    gpu_index: int | None = None
    uid: int | None = None
    ours: bool = False

    @property
    def min_user(self) -> str:
        """供展示的用户标识（无法读取时给未知）。"""
        return f"uid{self.uid}" if self.uid is not None else "unknown"


@dataclass
class GpuInfo:
    """单张 GPU 的瞬时状态。"""

    index: int
    uuid: str = ""
    name: str = ""
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    utilization: int | None = None
    temperature_c: int | None = None
    persistence_mode: str = ""
    compute_processes: list[ComputeProcess] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def memory_free_mb(self) -> int:
        if self.memory_total_mb is None:
            return 0
        return max(0, self.memory_total_mb - (self.memory_used_mb or 0))

    @property
    def foreign_processes(self) -> list[ComputeProcess]:
        """不属于当前用户的 compute 进程。"""
        return [p for p in self.compute_processes if not p.ours]

    @property
    def own_processes(self) -> list[ComputeProcess]:
        return [p for p in self.compute_processes if p.ours]

    @property
    def ours_only(self) -> bool:
        return not self.foreign_processes

    @property
    def process_pids(self) -> list[int]:
        return [p.pid for p in self.compute_processes]

    def idle_reasons(self, max_utilization: int, max_memory_mb: int, require_no_process: bool) -> list[str]:
        """返回「不满足空闲」的原因列表；为空则表示空闲。"""
        reasons: list[str] = []
        utilization = self.utilization
        used = self.memory_used_mb

        if utilization is None or used is None:
            reasons.append("util/显存读取失败")
        else:
            if utilization > max_utilization:
                reasons.append(f"util {utilization}% > {max_utilization}%")
            if used > max_memory_mb:
                reasons.append(f"mem {used}MiB > {max_memory_mb}MiB")
        if require_no_process and self.compute_processes:
            who = ", ".join(f"pid{p.pid}({'ours' if p.ours else p.min_user})" for p in self.compute_processes[:4])
            reasons.append(f"有 compute 进程: {who}")
        return reasons

    def summary(self) -> str:
        util = "-" if self.utilization is None else f"{self.utilization}%"
        mem = "-" if self.memory_used_mb is None else f"{self.memory_used_mb}MiB"
        total = "-" if self.memory_total_mb is None else f"{self.memory_total_mb}MiB"
        procs = len(self.compute_processes)
        return f"GPU{self.index} util={util} mem={mem}/{total} compute_procs={procs}"


@dataclass
class Snapshot:
    """一次 nvidia-smi 采集的完整结果。"""

    gpus: list[GpuInfo] = field(default_factory=list)
    error: str = ""
    timestamp: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.gpus)

    def by_index(self) -> dict[int, GpuInfo]:
        return {g.index: g for g in self.gpus}

    def get(self, index: int) -> GpuInfo | None:
        for g in self.gpus:
            if g.index == index:
                return g
        return None

    def all_indices(self) -> list[int]:
        return sorted(g.index for g in self.gpus)


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def _run_nvidia_smi(args: list[str], timeout: float) -> tuple[str, str]:
    cmd = ["nvidia-smi", *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
        )
    except FileNotFoundError:
        return "", "nvidia-smi 不可执行（未安装或不在 PATH）"
    except subprocess.TimeoutExpired:
        return "", f"nvidia-smi 超时（{timeout}s）: {' '.join(cmd)}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = err[0] if err else f"exit={proc.returncode}"
        return "", f"nvidia-smi 失败: {detail}"
    return proc.stdout, ""


def _data_lines(stdout: str) -> list[str]:
    """返回所有非空行 —— **刻意不做表头跳过**。

    这里有一个曾经存在的真实 bug（已回归测试锁定）：
    我们用 --format=csv,noheader,nounits，nvidia-smi 不输出表头，
    但早期代码无条件 `lines[1:]`，于是第一张 GPU 和第一条 compute 进程被丢掉。
    后者是安全隐患：某张卡唯一的占用进程被忽略 → 有可能被误判为空闲并占用。

    修法：不靠启发式猜表头，而是让每行的 schema 校验自然过滤
    （表头行的 index 不是数字、pid 字段也不是数字，会被 _to_int 判为 None 而跳过）。
    这样带表头 / 不带表头两种输出都正确。
    """
    return [ln for ln in stdout.splitlines() if ln.strip()]


def _parse_gpus(stdout: str) -> tuple[list[GpuInfo], dict[str, int]]:
    gpus: list[GpuInfo] = []
    uuid_to_index: dict[str, int] = {}
    for line in _data_lines(stdout):
        row = _split_csv_line(line)
        if len(row) < 6:
            continue
        index = _to_int(row[0])
        if index is None:
            continue
        gpu = GpuInfo(
            index=index,
            uuid=row[1].strip(),
            name=row[2].strip(),
            memory_total_mb=_to_int(row[3]),
            memory_used_mb=_to_int(row[4]),
            utilization=_to_int(row[5]),
            temperature_c=_to_int(row[6]) if len(row) > 6 else None,
            persistence_mode=row[7].strip() if len(row) > 7 else "",
        )
        gpus.append(gpu)
        if gpu.uuid:
            uuid_to_index[gpu.uuid] = index
    return gpus, uuid_to_index


def _parse_processes(stdout: str, uuid_to_index: dict[str, int], uid: int) -> dict[int, list[ComputeProcess]]:
    by_index: dict[int, list[ComputeProcess]] = {}
    for line in _data_lines(stdout):
        row = _split_csv_line(line)
        if len(row) < 2:
            continue
        gpu_uuid = row[0].strip()
        pid = _to_int(row[1])
        if pid is None or pid <= 0:
            continue
        name = row[2].strip() if len(row) > 2 else ""
        mem = _to_int(row[3]) if len(row) > 3 else None
        index = uuid_to_index.get(gpu_uuid)
        if index is None:
            continue
        proc_owner = proc_uid(pid)
        by_index.setdefault(index, []).append(
            ComputeProcess(
                pid=pid, name=name, memory_mb=mem, gpu_index=index,
                uid=proc_owner, ours=(proc_owner == uid),
            )
        )
    return by_index


def collect(timeout: float = 10.0, *, include_processes: bool = True) -> Snapshot:
    """执行一次完整采集。任何一步失败都返回带 error 的 Snapshot（保守，不判空闲）。"""
    snap = Snapshot(timestamp=time.time())

    gpu_out, err = _run_nvidia_smi(
        [f"--query-gpu={CSV_GPU_QUERY}", "--format=csv,noheader,nounits"], timeout
    )
    if err:
        snap.error = err
        return snap
    gpus, uuid_to_index = _parse_gpus(gpu_out)
    if not gpus:
        snap.error = "nvidia-smi 未返回任何 GPU"
        return snap

    if include_processes:
        proc_out, err = _run_nvidia_smi(
            [f"--query-compute-apps={CSV_PROC_QUERY}", "--format=csv,noheader,nounits"], timeout
        )
        if err:
            # 进程信息读不到 → 保守: 认为所有卡都可能被占用
            for gpu in gpus:
                gpu.compute_processes.append(
                    ComputeProcess(pid=-1, name="<unknown: nvidia-smi query-compute-apps failed>", ours=False)
                )
            snap.error = err
        else:
            uid = current_uid()
            by_index = _parse_processes(proc_out, uuid_to_index, uid)
            for gpu in gpus:
                gpu.compute_processes = by_index.get(gpu.index, [])

    snap.gpus = gpus
    return snap


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def idle_gpu_ids(
    snapshot: Snapshot,
    *,
    max_utilization: int,
    max_memory_mb: int,
    require_no_process: bool = True,
    monitor_gpus: Iterable[int] | None = None,
) -> list[int]:
    """满足空闲阈值的 GPU（仅依据一次快照，不含连续确认）。"""
    if not snapshot.ok:
        return []
    monitored = set(monitor_gpus) if monitor_gpus is not None else None
    out: list[int] = []
    for gpu in snapshot.gpus:
        if monitored is not None and gpu.index not in monitored:
            continue
        if not gpu.idle_reasons(max_utilization, max_memory_mb, require_no_process):
            out.append(gpu.index)
    return sorted(out)


def idle_report(
    snapshot: Snapshot,
    *,
    max_utilization: int,
    max_memory_mb: int,
    require_no_process: bool = True,
    monitor_gpus: Iterable[int] | None = None,
) -> dict[int, list[str]]:
    """每张受监控 GPU 的「不空闲原因」，供 status/日志解释判定结果。"""
    report: dict[int, list[str]] = {}
    if not snapshot.ok:
        return report
    monitored = set(monitor_gpus) if monitor_gpus is not None else None
    for gpu in snapshot.gpus:
        if monitored is not None and gpu.index not in monitored:
            continue
        report[gpu.index] = gpu.idle_reasons(max_utilization, max_memory_mb, require_no_process)
    return report


def still_free(
    snapshot: Snapshot,
    gpu_ids: Iterable[int],
    *,
    max_utilization: int,
    max_memory_mb: int,
    require_no_process: bool = True,
) -> tuple[bool, str]:
    """启动前二次检查。返回 (是否可以启动, 说明)。"""
    ids = sorted({int(g) for g in gpu_ids})
    if not ids:
        return False, "未选择任何 GPU"
    if not snapshot.ok:
        return False, f"nvidia-smi 采集失败: {snapshot.error}"
    index = snapshot.by_index()
    problems: list[str] = []
    for gid in ids:
        gpu = index.get(gid)
        if gpu is None:
            problems.append(f"GPU{gid} 不在快照中")
            continue
        if gpu.foreign_processes:
            who = ", ".join(f"{p.name}(pid {p.pid}, {p.min_user})" for p in gpu.foreign_processes[:3])
            problems.append(f"GPU{gid} 已被他人占用: {who}")
            continue
        reasons = gpu.idle_reasons(max_utilization, max_memory_mb, require_no_process)
        if reasons:
            problems.append(f"GPU{gid} " + "; ".join(reasons))
    if problems:
        return False, " | ".join(problems)
    return True, f"GPU {gpu_ids_text(ids)} 二次检查通过"


def as_dict_rows(snapshot: Snapshot, gpu_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """给 CLI/JSON 用的结构化输出。"""
    wanted = set(gpu_ids) if gpu_ids is not None else None
    rows: list[dict[str, Any]] = []
    for gpu in snapshot.gpus:
        if wanted is not None and gpu.index not in wanted:
            continue
        rows.append(
            {
                "index": gpu.index,
                "name": gpu.name,
                "utilization": gpu.utilization,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_total_mb": gpu.memory_total_mb,
                "memory_free_mb": gpu.memory_free_mb,
                "temperature_c": gpu.temperature_c,
                "compute_processes": [
                    {"pid": p.pid, "name": p.name, "memory_mb": p.memory_mb, "ours": p.ours, "uid": p.uid}
                    for p in gpu.compute_processes
                ],
            }
        )
    return rows
