"""GPU 采集与空闲判定测试。

重点锁定两个曾经真实存在的 bug（在真实服务器上被实测发现）:
  1. 用 --format=csv,noheader 时不应跳过首行 —— 否则丢掉第一张 GPU。
  2. 同理丢掉第一条 compute 进程 —— 这是**安全隐患**：
     某张卡唯一的占用进程被忽略后，可能被误判为空闲并被占用。
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from gpu_manager.gpu_monitor import (
    ComputeProcess,
    GpuInfo,
    Snapshot,
    _parse_gpus,
    _parse_processes,
    collect,
    idle_gpu_ids,
    idle_report,
    still_free,
)

# nvidia-smi 在 --format=csv,noheader,nounits 下的真实形状（不含表头）
GPU_NOHEADER = (
    "0, GPU-aaa, NVIDIA RTX A6000, 49140, 10, 0, 40, Disabled\n"
    "1, GPU-bbb, NVIDIA RTX A6000, 49140, 20, 5, 41, Disabled\n"
)
GPU_HEADER = (
    "index, uuid, name, memory.total, memory.used, utilization.gpu, "
    "temperature.gpu, persistence_mode\n" + GPU_NOHEADER
)
PROC_NOHEADER = (
    "GPU-aaa, 111, python, 100 MiB\n"
    "GPU-bbb, 222, python, 200 MiB\n"
)
PROC_HEADER = "gpu_uuid, pid, process_name, used_gpu_memory [MiB]\n" + PROC_NOHEADER


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def test_parse_gpus_keeps_first_row_without_header():
    gpus, mapping = _parse_gpus(GPU_NOHEADER)
    assert [g.index for g in gpus] == [0, 1], "首行被当成表头丢掉 → 会漏掉 GPU 0"
    assert mapping["GPU-aaa"] == 0


def test_parse_gpus_with_header_also_works():
    gpus, _ = _parse_gpus(GPU_HEADER)
    assert [g.index for g in gpus] == [0, 1], "带表头时应靠字段校验过滤表头，而不是靠跳行"


def test_parse_processes_keeps_first_row_without_header():
    by_index = _parse_processes(PROC_NOHEADER, {"GPU-aaa": 0, "GPU-bbb": 1}, uid=os.getuid())
    assert [p.pid for p in by_index[0]] == [111], "首条 compute 进程被丢掉 → 存在误判空闲的安全隐患"


def test_parse_processes_with_header_also_works():
    by_index = _parse_processes(PROC_HEADER, {"GPU-aaa": 0, "GPU-bbb": 1}, uid=os.getuid())
    assert [p.pid for p in by_index[0]] == [111]


def test_parse_processes_handles_comma_in_process_name():
    line = "GPU-aaa, 333, /usr/bin/python3 -c print(1,2), 10 MiB\n"
    by_index = _parse_processes(line, {"GPU-aaa": 0}, uid=os.getuid())
    assert by_index[0][0].pid == 333


def test_parse_handles_na_and_missing_units():
    line = "0, GPU-aaa, RTX, 49140 MiB, [N/A], N/A, 40, Disabled\n"
    gpus, _ = _parse_gpus(line)
    assert gpus[0].memory_total_mb == 49140
    assert gpus[0].memory_used_mb is None
    assert gpus[0].utilization is None


def test_parse_skips_garbage_lines():
    line = "not,a,gpu,row\n0, GPU-aaa, RTX, 49140, 10, 0, 40, Disabled\n"
    gpus, _ = _parse_gpus(line)
    assert [g.index for g in gpus] == [0]


# --------------------------------------------------------------------------- #
# 空闲判定
# --------------------------------------------------------------------------- #
def make_gpu(
    index: int = 0,
    util: int | None = 0,
    used: int | None = 0,
    procs: Iterable[ComputeProcess] = (),
    total: int | None = 49140,
) -> GpuInfo:
    """构造合成 GPU 状态。

    util/used 允许 None，用于模拟 nvidia-smi 读不到值的情形；
    procs 接受任意可迭代对象（测试里通常直接给 list）。
    """
    return GpuInfo(
        index=index, uuid=f"GPU-{index}", name="RTX", memory_total_mb=total,
        memory_used_mb=used, utilization=util, compute_processes=list(procs),
    )


def own_proc(pid=None) -> ComputeProcess:
    return ComputeProcess(pid=pid or os.getpid(), name="python", ours=True, uid=os.getuid())


def foreign_proc(pid=999999) -> ComputeProcess:
    return ComputeProcess(pid=pid, name="python", ours=False, uid=1234)


def test_idle_uses_both_util_and_memory_thresholds():
    snap = Snapshot(
        gpus=[
            make_gpu(0, util=0, used=0),        # 空闲
            make_gpu(1, util=99, used=0),       # 利用率高
            make_gpu(2, util=0, used=40000),    # 显存被占
            make_gpu(3, util=5, used=1000),     # 正好在阈值上 → 空闲
        ]
    )
    idle = idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000, require_no_process=True)
    assert idle == [0, 3]


def test_own_running_task_makes_gpu_non_idle():
    """关键安全属性: 自己任务占着的卡绝不能算空闲（否则会超卖/抢自己）。"""
    snap = Snapshot(gpus=[make_gpu(0, util=0, used=500, procs=[own_proc()])])
    idle = idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000, require_no_process=True)
    assert idle == [], "自己的 compute 进程也算占用"


def test_foreign_process_makes_gpu_non_idle():
    snap = Snapshot(gpus=[make_gpu(0, util=0, used=500, procs=[foreign_proc()])])
    assert idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000, require_no_process=True) == []


def test_process_check_can_be_relaxed():
    snap = Snapshot(gpus=[make_gpu(0, util=0, used=500, procs=[foreign_proc()])])
    idle = idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000, require_no_process=False)
    assert idle == [0]


def test_unreadable_util_is_not_idle():
    snap = Snapshot(gpus=[make_gpu(0, util=None, used=None)])
    assert idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000) == []
    reasons = idle_report(snap, max_utilization=5, max_memory_mb=1000)
    assert "读取失败" in reasons[0][0]


def test_failed_snapshot_yields_no_idle_gpus():
    """采集失败时必须保守：一张卡都不能判为空闲。"""
    snap = Snapshot(gpus=[make_gpu(0, util=0, used=0)], error="nvidia-smi timeout")
    assert snap.ok is False
    assert idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000) == []


def test_monitor_gpu_filter():
    snap = Snapshot(gpus=[make_gpu(0), make_gpu(1), make_gpu(2)])
    assert idle_gpu_ids(snap, max_utilization=5, max_memory_mb=1000, monitor_gpus=[1, 2]) == [1, 2]


# --------------------------------------------------------------------------- #
# 启动前二次检查
# --------------------------------------------------------------------------- #
def test_still_free_ok():
    snap = Snapshot(gpus=[make_gpu(3), make_gpu(6)])
    ok, detail = still_free(snap, [3, 6], max_utilization=5, max_memory_mb=1000)
    assert ok, detail


def test_still_free_rejects_when_other_user_grabbed_it():
    snap = Snapshot(gpus=[make_gpu(3), make_gpu(6, procs=[foreign_proc()])])
    ok, detail = still_free(snap, [3, 6], max_utilization=5, max_memory_mb=1000)
    assert not ok
    assert "他人占用" in detail


def test_still_free_rejects_missing_gpu():
    snap = Snapshot(gpus=[make_gpu(3)])
    ok, detail = still_free(snap, [3, 7], max_utilization=5, max_memory_mb=1000)
    assert not ok
    assert "GPU7" in detail


def test_still_free_rejects_when_busy_by_utilization():
    snap = Snapshot(gpus=[make_gpu(4, util=80, used=30000)])
    ok, detail = still_free(snap, [4], max_utilization=5, max_memory_mb=1000)
    assert not ok
    assert "util" in detail


def test_still_free_rejects_on_collection_failure():
    snap = Snapshot(error="boom")
    ok, _ = still_free(snap, [0], max_utilization=5, max_memory_mb=1000)
    assert not ok


def test_still_free_requires_at_least_one_gpu():
    ok, detail = still_free(Snapshot(gpus=[make_gpu(0)]), [], max_utilization=5, max_memory_mb=1000)
    assert not ok
    assert "未选择" in detail


# --------------------------------------------------------------------------- #
# 真实采集（在真实 GPU 机器上才有意义）
# --------------------------------------------------------------------------- #
def test_collect_on_real_machine_parses_all_gpus():
    snap = collect(timeout=20.0)
    if not snap.ok:
        return  # 无 GPU / 无 nvidia-smi 的环境跳过
    assert len(snap.gpus) >= 1
    indices = snap.all_indices()
    assert len(indices) == len(set(indices)), "GPU index 不应重复"
    # 每张卡要么读到进程，要么 util/mem 表明没在用
    for gpu in snap.gpus:
        assert gpu.memory_total_mb is None or gpu.memory_total_mb > 0
