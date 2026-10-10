"""日志保留与磁盘占用控制。

设计要点:
  - 以「一次运行的产物」为单位删除：``<key>.log`` / ``<key>.log.1`` / ``<key>.run.sh``
    / ``<key>.exit`` 同属一组，避免删掉日志却留下孤立的 run.sh。
  - **绝不动正在运行任务的日志**（daemon 传入活跃日志路径，命中即整组跳过）。
  - 两级约束:
      1. 每个任务最多保留最近 ``task_log_keep_files_per_task`` 次运行
      2. 整个 ``logs/`` 目录总量不超过 ``task_log_dir_max_mb``（超出按最旧优先删）
    总量超限时默认**先删成功任务的日志**，失败任务的留久一点（排错要用）。
  - ``events.jsonl`` 与 daemon 的 stdout 文件没有天然轮转，这里按大小裁剪，
    只保留尾部完整行。
  - 所有数字来自 config.yaml 的 ``storage`` 段；0 表示不限制。
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .utils import abs_path, atomic_write_text

#: 视作「同一组」的扩展名（其余归属该组的文件也会一起删）
_RUN_SUFFIXES = (".log", ".run.sh", ".exit")


@dataclass
class RunGroup:
    """一次运行的产物集合。"""

    task: str
    key: str
    files: list[Path]
    size: int
    mtime: float


@dataclass
class PruneReport:
    removed_files: list[str] = field(default_factory=list)
    removed_bytes: int = 0
    trimmed_bytes: int = 0
    remaining_bytes: int = 0
    skipped_active: int = 0
    dry_run: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.removed_files) or self.trimmed_bytes > 0

    def summary(self) -> str:
        prefix = "[dry-run] " if self.dry_run else ""
        parts = [
            f"删除 {len(self.removed_files)} 个文件 / {human_size(self.removed_bytes)}",
        ]
        if self.trimmed_bytes:
            parts.append(f"裁剪 {human_size(self.trimmed_bytes)}")
        parts.append(f"logs/ 剩余 {human_size(self.remaining_bytes)}")
        if self.skipped_active:
            parts.append(f"跳过运行中 {self.skipped_active} 组")
        return prefix + "，".join(parts)

    def detail_lines(self, limit: int = 20) -> list[str]:
        lines = [f"  - {name}" for name in self.removed_files[:limit]]
        if len(self.removed_files) > limit:
            lines.append(f"  ...另有 {len(self.removed_files) - limit} 个")
        return lines


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def human_size(num_bytes: float) -> str:
    value = float(max(0.0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"  # pragma: no cover


def dir_usage_bytes(root: str | os.PathLike[str]) -> int:
    """目录内普通文件总大小（不计软链，避免重复计算）。"""
    path = abs_path(root)
    if not path.is_dir():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            entry = Path(dirpath) / name
            if entry.is_symlink():
                continue
            with contextlib.suppress(OSError):
                total += entry.stat().st_size
    return total


def _safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _run_key(name: str) -> str:
    """文件名 → 「一次运行」的组键（去掉第一个点之后的部分）。"""
    return name.split(".", 1)[0]


def _collect_groups(root: Path, active: set[str]) -> tuple[list[RunGroup], int]:
    """收集所有可清理的运行组；返回 (组列表, 跳过组数)。"""
    groups: list[RunGroup] = []
    skipped = 0
    if not root.is_dir():
        return groups, skipped
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        by_key: dict[str, list[Path]] = {}
        for entry in sorted(task_dir.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                continue
            by_key.setdefault(_run_key(entry.name), []).append(entry)
        for key, files in by_key.items():
            resolved = {str(p.resolve()) for p in files}
            if active & resolved:
                skipped += 1
                continue
            size = 0
            mtime = 0.0
            for path in files:
                stat = _safe_stat(path)
                if stat is None:
                    continue
                size += stat.st_size
                mtime = max(mtime, stat.st_mtime)
            groups.append(RunGroup(task=task_dir.name, key=key, files=files, size=size, mtime=mtime))
    return groups, skipped


def _delete_group(group: RunGroup, report: PruneReport, dry_run: bool) -> None:
    for path in group.files:
        stat = _safe_stat(path)
        size = stat.st_size if stat else 0
        report.removed_files.append(str(path))
        report.removed_bytes += size
        if not dry_run:
            with contextlib.suppress(OSError):
                path.unlink()


def _cleanup_symlinks_and_empty_dirs(root: Path, active: set[str], dry_run: bool) -> None:
    """删掉指向已不存在目标的 latest.* 软链，以及变空的子目录。"""
    if not root.is_dir():
        return
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for link in sorted(task_dir.glob("latest.*")):
            if not link.is_symlink():
                continue
            if str(link.resolve()) in active:
                continue
            target_exists = link.exists()
            if not target_exists and not dry_run:
                with contextlib.suppress(OSError):
                    link.unlink()
        with contextlib.suppress(OSError):
            if not dry_run and not any(task_dir.iterdir()):
                task_dir.rmdir()


# --------------------------------------------------------------------------- #
# 裁剪（events.jsonl / daemon stdout）
# --------------------------------------------------------------------------- #
def trim_tail(path: str | os.PathLike[str], max_bytes: int) -> int:
    """把文本文件裁剪到 max_bytes 之内，只保留**尾部完整行**。返回释放的字节数。

    多字节字符在 seek 边界被切开时只会影响被丢弃的首行，不影响保留内容。
    """
    target = abs_path(path)
    if max_bytes <= 0:
        return 0
    stat = _safe_stat(target)
    if stat is None or stat.st_size <= max_bytes:
        return 0
    keep_bytes = max(1, max_bytes // 2)
    try:
        with open(target, encoding="utf-8", errors="replace") as fh:
            fh.seek(max(0, stat.st_size - keep_bytes))
            fh.readline()  # 丢弃可能被截断的首行
            tail = fh.read()
    except OSError:
        return 0
    atomic_write_text(target, tail)
    return max(0, stat.st_size - len(tail.encode("utf-8")))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def prune_task_logs(
    cfg: Config,
    *,
    active_log_paths: Iterable[str] = (),
    failed_task_names: Iterable[str] = (),
    dry_run: bool = False,
) -> PruneReport:
    """按 storage 配置清理任务日志。"""
    storage = cfg.storage
    root = abs_path(cfg.task_log_dir)
    report = PruneReport(dry_run=dry_run)
    active = {str(Path(p).resolve()) for p in active_log_paths if p}
    failed = {str(name) for name in failed_task_names}

    groups, skipped = _collect_groups(root, active)
    report.skipped_active = skipped
    if not groups:
        report.remaining_bytes = dir_usage_bytes(root)
        return report

    doomed: list[RunGroup] = []
    doomed_keys: set[tuple[str, str]] = set()

    # ---- 第一级: 每个任务保留最近 N 次运行 ----
    keep_n = int(storage.task_log_keep_files_per_task)
    if keep_n > 0:
        by_task: dict[str, list[RunGroup]] = {}
        for group in groups:
            by_task.setdefault(group.task, []).append(group)
        for items in by_task.values():
            items.sort(key=lambda g: g.mtime, reverse=True)  # 最新的在前
            for group in items[keep_n:]:
                doomed.append(group)
                doomed_keys.add((group.task, group.key))

    # ---- 第二级: logs/ 总量上限 ----
    cap = int(storage.task_log_dir_max_mb) * 1024 * 1024
    remaining = sum(g.size for g in groups if (g.task, g.key) not in doomed_keys)
    if cap > 0 and remaining > cap:
        prefer_failed = bool(storage.prefer_keep_failed_logs)
        candidates = [g for g in groups if (g.task, g.key) not in doomed_keys]
        # 排序即删除顺序: 先删「非失败任务」的，再按时间从旧到新
        candidates.sort(
            key=lambda g: (1 if (prefer_failed and g.task in failed) else 0, g.mtime)
        )
        for group in candidates:
            if remaining <= cap:
                break
            doomed.append(group)
            doomed_keys.add((group.task, group.key))
            remaining -= group.size

    for group in doomed:
        _delete_group(group, report, dry_run)

    _cleanup_symlinks_and_empty_dirs(root, active, dry_run)
    report.remaining_bytes = dir_usage_bytes(root) if not dry_run else dir_usage_bytes(root)
    return report


def prune_aux_files(cfg: Config, *, dry_run: bool = False) -> PruneReport:
    """裁剪 events.jsonl、daemon stdout 这类无轮转的文件。"""
    storage = cfg.storage
    report = PruneReport(dry_run=dry_run)
    targets = (
        (cfg.event_log, int(storage.events_max_mb)),
        (cfg.log_file.with_suffix(".out.log"), int(storage.daemon_out_max_mb)),
    )
    for path, max_mb in targets:
        if max_mb <= 0:
            continue
        freed = 0 if dry_run else trim_tail(path, max_mb * 1024 * 1024)
        if dry_run:
            stat = _safe_stat(abs_path(path))
            limit = max_mb * 1024 * 1024
            freed = max(0, (stat.st_size if stat else 0) - limit) if stat else 0
        report.trimmed_bytes += freed
        if freed:
            report.removed_files.append(f"{path}（裁剪 {human_size(freed)}）")
    return report


def prune_all(
    cfg: Config,
    logger: logging.Logger | None = None,
    *,
    active_log_paths: Iterable[str] = (),
    failed_task_names: Iterable[str] = (),
    dry_run: bool = False,
) -> PruneReport:
    """清理全部日志（任务日志 + 辅助文件 + 看门狗诊断）。"""
    task_report = prune_task_logs(
        cfg,
        active_log_paths=active_log_paths,
        failed_task_names=failed_task_names,
        dry_run=dry_run,
    )
    aux_report = prune_aux_files(cfg, dry_run=dry_run)
    diag_report = prune_watchdog_diagnostics(cfg, dry_run=dry_run)
    task_report.trimmed_bytes += aux_report.trimmed_bytes
    task_report.removed_files.extend(aux_report.removed_files)
    task_report.removed_bytes += diag_report.removed_bytes
    task_report.removed_files.extend(diag_report.removed_files)
    if logger is not None and task_report.changed:
        logger.info("日志清理: %s", task_report.summary())
    return task_report


def prune_watchdog_diagnostics(cfg: Config, *, dry_run: bool = False) -> PruneReport:
    """清理看门狗诊断快照（上限见 ``watchdog.diagnostics_*``）。

    目录布局: ``<paths.diagnostics_dir>/<任务>/<YYYYmmdd-HHMMSS[-N]>/``。

    两条约束:
      1. 每个任务保留最近 ``diagnostics_keep_per_task`` 份
      2. 整个诊断目录不超过 ``diagnostics_max_mb``，超出时从最旧开始删

    **每个任务最新的那份永远不删**（哪怕上限更小）——最新一次告警的现场是
    最需要保留的，与日志清理「绝不删运行中任务日志」同一思路。
    """
    report = PruneReport(dry_run=dry_run)
    wcfg = cfg.watchdog
    if not wcfg.diagnostics_enabled:
        return report
    root = abs_path(cfg.diagnostics_dir)
    snapshots = _diagnostic_snapshots(root)
    if not snapshots:
        return report

    keep = max(0, int(wcfg.diagnostics_keep_per_task))
    max_bytes = max(0, int(wcfg.diagnostics_max_mb)) * 1024 * 1024

    by_task: dict[Path, list[tuple[Path, int, float]]] = {}
    for item in snapshots:
        by_task.setdefault(item[0].parent, []).append(item)

    doomed: dict[Path, int] = {}
    survivors: list[tuple[Path, int, float]] = []
    for items in by_task.values():
        # 目录名即时间戳 → 名称倒序就是「新 → 旧」
        items.sort(key=lambda it: (it[0].name, it[2]), reverse=True)
        for idx, item in enumerate(items):
            if idx < max(1, keep):
                survivors.append(item)
            else:
                doomed[item[0]] = item[1]

    if max_bytes:
        total = sum(size for _path, size, _mtime in survivors)
        newest = {items[0][0] for items in by_task.values() if items}
        for path, size, _mtime in sorted(survivors, key=lambda it: (it[2], it[0].name)):
            if total <= max_bytes:
                break
            if path in newest:
                continue
            doomed[path] = size
            total -= size

    for path in sorted(doomed):
        size = doomed[path]
        if dry_run:
            report.removed_files.append(f"{path}（{human_size(size)}）")
        else:
            shutil.rmtree(path, ignore_errors=True)
            report.removed_files.append(str(path))
            report.removed_bytes += size

    if not dry_run:
        for task_dir in list(by_task):
            with contextlib.suppress(OSError):
                if task_dir.is_dir() and not any(task_dir.iterdir()):
                    task_dir.rmdir()
        report.remaining_bytes = dir_usage_bytes(root)
    return report


def _diagnostic_snapshots(root: Path) -> list[tuple[Path, int, float]]:
    """收集诊断快照目录: ``[(目录, 字节数, mtime)]``（只认两层目录结构）。"""
    out: list[tuple[Path, int, float]] = []
    if not root.is_dir():
        return out
    for task_dir in sorted(root.iterdir()):
        if task_dir.is_symlink() or not task_dir.is_dir():
            continue
        for snap in sorted(task_dir.iterdir()):
            if snap.is_symlink() or not snap.is_dir():
                continue
            stat = _safe_stat(snap)
            out.append((snap, dir_usage_bytes(snap), stat.st_mtime if stat else 0.0))
    return out


def usage_overview(cfg: Config) -> list[tuple[str, int]]:
    """给 `gpu-manager prune --usage` 用的占用概览（只含真实占用项）。"""
    out_log = cfg.log_file.with_suffix(".out.log")
    rows = [
        (f"任务日志 {cfg.task_log_dir}", dir_usage_bytes(cfg.task_log_dir)),
        (f"daemon 日志 {cfg.log_file.name}", _file_size(cfg.log_file)),
        (f"事件流 {cfg.event_log.name}", _file_size(cfg.event_log)),
        (f"stdout {out_log.name}", _file_size(out_log)),
    ]
    diag_dir = abs_path(cfg.diagnostics_dir)
    if diag_dir.is_dir():
        rows.append((f"看门狗诊断 {diag_dir}", dir_usage_bytes(diag_dir)))
    return rows


def limits_text(cfg: Config) -> str:
    """把生效的上限拼成一行，供 CLI 展示。"""
    storage = cfg.storage
    parts: list[str] = []
    if storage.task_log_dir_max_mb:
        parts.append(f"logs/ <= {storage.task_log_dir_max_mb}MB")
    if storage.task_log_keep_files_per_task:
        parts.append(f"每任务留最近 {storage.task_log_keep_files_per_task} 次")
    if storage.events_max_mb:
        parts.append(f"events <= {storage.events_max_mb}MB")
    if storage.daemon_log_max_mb:
        parts.append(f"daemon 日志 <= {storage.daemon_log_max_mb}MB x{storage.daemon_log_backups}")
    if storage.daemon_out_max_mb:
        parts.append(f"stdout <= {storage.daemon_out_max_mb}MB")
    wcfg = cfg.watchdog
    if wcfg.diagnostics_enabled and (wcfg.diagnostics_keep_per_task or wcfg.diagnostics_max_mb):
        bits: list[str] = []
        if wcfg.diagnostics_keep_per_task:
            bits.append(f"每任务 {wcfg.diagnostics_keep_per_task} 份")
        if wcfg.diagnostics_max_mb:
            bits.append(f"{wcfg.diagnostics_max_mb}MB")
        parts.append("诊断快照 <= " + " / ".join(bits))
    return "，".join(parts) if parts else "未设置任何上限（storage.* 全为 0）"


def _file_size(path: str | os.PathLike[str]) -> int:
    stat = _safe_stat(abs_path(path))
    return stat.st_size if stat else 0
