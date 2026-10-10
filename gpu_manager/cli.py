"""命令行接口（需求: 配置 + CLI 管理队列）。

设计要点:
  - daemon 在线时所有变更走 AF_UNIX socket（立即生效且不会与 daemon 抢写 state.json）。
  - daemon 离线时退化为直接读写 state.json（带文件锁），下次启动自动接管。
  - 任何写操作都不需要 sudo，也不触碰系统级配置。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any, TextIO

import yaml

from . import __version__
from .config import Config, ConfigError, load_config
from .gpu_monitor import as_dict_rows, collect
from .ipc import send_request, socket_in_use
from .retention import (
    dir_usage_bytes,
    human_size,
    limits_text,
    prune_all,
    usage_overview,
)
from .runner import TaskRunner
from .state_store import StateStore
from .task_queue import (
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    QueueManager,
    Task,
    TaskError,
)
from .utils import (
    build_logger,
    fmt_duration,
    fmt_ts,
    gpu_ids_text,
    proc_uid,
    process_alive,
    resolve_within,
    truncate,
    wait_until,
)
from .watchdog import format_status as format_watchdog_status

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATUS_ICON = {
    PENDING: "◻",
    RUNNING: "▶",
    COMPLETED: "✔",
    FAILED: "✘",
    CANCELLED: "⊘",
}

LEVEL_ICON = {"info": "·", "warning": "!", "error": "✘", "ok": "✔", "fail": "✘"}


# --------------------------------------------------------------------------- #
# 终端渲染（中文按双宽计算，避免表格错位）
# --------------------------------------------------------------------------- #
def display_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - display_width(text))
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def render_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    if not rows:
        return "(空)"
    cols = len(headers)
    aligns = aligns or ["left"] * cols
    widths = [display_width(h) for h in headers]
    for row in rows:
        for idx in range(cols):
            cell = row[idx] if idx < len(row) else ""
            widths[idx] = max(widths[idx], display_width(cell))
    out = ["  ".join(pad(headers[i], widths[i]) for i in range(cols))]
    out.append("  ".join("─" * widths[i] for i in range(cols)))
    for row in rows:
        cells = [(row[i] if i < len(row) else "") for i in range(cols)]
        out.append("  ".join(pad(cells[i], widths[i], aligns[i]) for i in range(cols)))
    return "\n".join(out)


def _size_changed(fd: int, expected: int) -> bool:
    """fd 对应文件的大小是否已不等于 expected（tail -f 等新内容用）。

    做成模块级函数而不是循环内的闭包: 避免闭包捕获循环变量，也便于单测。
    """
    try:
        return os.fstat(fd).st_size != expected
    except OSError:
        return True  # 文件被删/不可读时不再傻等


class Style:
    """极简 ANSI 着色（非 TTY 或 NO_COLOR 时自动关闭）。"""

    CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36", "dim": "2"}

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def __call__(self, text: str, color: str) -> str:
        if not self.enabled:
            return text
        code = self.CODES.get(color)
        return f"\033[{code}m{text}\033[0m" if code else text


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-manager",
        description="实验室共享 GPU 服务器上的空闲提醒 + 训练任务自动调度（只服务当前用户）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  gpu-manager start                    # 后台常驻（nohup 等价，SSH 断开不影响）\n"
            "  gpu-manager status                   # 看 GPU/模式/队列\n"
            "  gpu-manager add --name exp1 --cmd 'python main.py' --gpu-num 2 --env undetr\n"
            "  gpu-manager list                     # 任务状态\n"
            "  gpu-manager cancel exp1              # 优雅停止自己的任务\n"
            "  gpu-manager logs exp1 -f             # 跟踪任务日志\n"
        ),
    )
    parser.add_argument("--config", "-c", default=None, help="配置文件路径（默认 config.yaml）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本处理）")
    parser.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    parser.add_argument("--version", action="version", version=f"gpu-manager {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- daemon（内部使用） ----
    p = sub.add_parser("daemon", help="前台运行 daemon（内部命令，通常用 start）")
    p.add_argument("--dry-run", action="store_true", help="不真正发送飞书消息")
    p.add_argument("--no-launch", action="store_true", help="只做调度决策，不启动任务")
    p.add_argument("--once", action="store_true", help="只跑一个 tick 后退出（调试用）")

    # ---- start ----
    p = sub.add_parser("start", help="启动 daemon（默认后台常驻）")
    p.add_argument("--foreground", "-f", action="store_true", help="前台运行（Ctrl-C 退出）")
    p.add_argument("--dry-run", action="store_true", help="不真正发送飞书消息（写日志）")
    p.add_argument("--no-launch", action="store_true", help="只做调度决策，不启动任务（验证用）")
    p.add_argument("--wait", type=float, default=20.0, help="等待就绪的秒数（默认 20）")

    # ---- stop ----
    p = sub.add_parser("stop", help="停止 daemon（不会杀死正在运行的训练任务）")
    p.add_argument("--force", action="store_true", help="超时后 SIGKILL（仅针对 daemon 进程）")
    p.add_argument("--timeout", type=float, default=30.0, help="等待退出的秒数（默认 30）")

    sub.add_parser("restart", help="重启 daemon（等价 stop + start）")
    sub.add_parser("reload", help="重新加载配置与 tasks.yaml")

    # ---- status ----
    p = sub.add_parser("status", help="查看 GPU / 模式 / 队列状态")
    p.add_argument("--watch", type=float, nargs="?", const=2.0, default=None, help="每 N 秒刷新（默认 2）")
    p.add_argument("--stale", action="store_true", help="使用 daemon 记录的快照，不重新执行 nvidia-smi")

    # ---- list / show ----
    p = sub.add_parser("list", help="列出任务")
    p.add_argument("--status", choices=[PENDING, RUNNING, COMPLETED, FAILED, CANCELLED], help="按状态过滤")
    p.add_argument("--all", action="store_true", help="包含终态任务（默认全含）")
    p.add_argument("--pending", action="store_true", help="只看待执行")

    p = sub.add_parser("show", help="查看单个任务详情（含等效命令）")
    p.add_argument("name")

    # ---- add ----
    p = sub.add_parser("add", help="提交新任务")
    p.add_argument("--name", required=True, help="任务名（唯一；字母数字._-）")
    p.add_argument("--cmd", required=True, help="训练命令（会通过 bash 执行）")
    p.add_argument("--cwd", default=None, help="工作目录（默认当前目录）")
    p.add_argument("--env", default=None, help="conda 环境名（或环境目录路径）")
    p.add_argument("--gpu-num", type=int, default=None, help="需要的 GPU 数（默认取 task_defaults.gpu_num）")
    p.add_argument("--priority", type=int, default=None, help="优先级，越大越先调度（默认 0）")
    p.add_argument("--min-free-memory-mb", type=int, default=None, help="单卡最低空闲显存要求（MiB）")
    p.add_argument("--max-retries", type=int, default=None, help="失败自动重试次数")
    p.add_argument("--retry-delay-seconds", type=int, default=None, help="重试等待秒数")
    p.add_argument("--timeout-seconds", type=int, default=None, help="运行超时秒数（0=不限）")
    p.add_argument("--env-var", action="append", default=[], metavar="K=V", help="附加环境变量，可重复")
    p.add_argument("--notes", default="", help="备注")
    p.add_argument("--replace", action="store_true", help="覆盖同名终态任务")
    p.add_argument("--dry-run", action="store_true", help="只校验并打印，不真正入队")

    # ---- cancel / remove / requeue ----
    p = sub.add_parser("cancel", help="取消任务（运行中则优雅停止其进程组）")
    p.add_argument("name")
    p = sub.add_parser("remove", help="从队列移除终态任务")
    p.add_argument("name")
    p = sub.add_parser("requeue", help="把终态任务重新置为 pending")
    p.add_argument("name")

    # ---- logs ----
    p = sub.add_parser("logs", help="查看 daemon 或任务日志")
    p.add_argument("name", nargs="?", default=None, help="任务名；省略则看 daemon 日志")
    p.add_argument("-n", "--lines", type=int, default=40, help="显示最后 N 行（默认 40）")
    p.add_argument("-f", "--follow", action="store_true", help="持续跟踪（Ctrl-C 退出）")

    # ---- events ----
    p = sub.add_parser("events", help="查看事件流（daemon 启动/任务状态变化等）")
    p.add_argument("-n", "--lines", type=int, default=30, help="显示最后 N 条（默认 30）")

    # ---- config ----
    p = sub.add_parser("config", help="打印生效配置（密钥已打码）")
    p.add_argument("--raw", action="store_true", help="打印原始配置文件")

    # ---- doctor ----
    p = sub.add_parser("doctor", help="环境自检（GPU / conda / 飞书 / 权限 / 任务定义）")
    p.add_argument("--notify-test", action="store_true", help="实际发送一条测试飞书消息")

    # ---- prune ----
    p = sub.add_parser("prune", help="清理过期日志，控制磁盘占用")
    p.add_argument("--dry-run", action="store_true", help="只列出会删什么，不真删")
    p.add_argument("--usage", action="store_true", help="只看当前占用与生效上限")

    # ---- watchdog ----
    p = sub.add_parser("watchdog", help="看门狗状态与诊断快照（只读）")
    p.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=["status", "diagnostics"],
        help="status=监视状态（默认）；diagnostics=列出/查看诊断快照",
    )
    p.add_argument("task", nargs="?", help="diagnostics 时可只看某个任务")
    p.add_argument("--show", action="store_true", help="打印最新一份诊断的 summary.txt 全文")

    # 把全局选项也补到每个子命令上。
    #
    # 原因: argparse 默认只接受这些选项写在子命令**之前**，于是
    # `gpu-manager start --config X` 会报 "unrecognized arguments: --config"，
    # 而这正是用户最自然的写法（也曾是 `gpu-manager start` 后台模式直接失败的原因）。
    # 子命令上的定义必须用 default=argparse.SUPPRESS：否则子解析器的默认值会在
    # 解析后覆盖主解析器已取到的值（argparse 经典陷阱）。
    for sub_parser in sub.choices.values():
        group = sub_parser.add_argument_group("全局选项（写在子命令前后都可以）")
        group.add_argument(
            "--config", "-c", default=argparse.SUPPRESS,
            help="配置文件路径（默认 config.yaml）",
        )
        group.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS,
            help="以 JSON 输出（便于脚本处理）",
        )
        group.add_argument(
            "--no-color", action="store_true", default=argparse.SUPPRESS,
            help="关闭彩色输出",
        )

    return parser


def build_daemon_argv(
    config_path: str | os.PathLike[str],
    *,
    dry_run: bool = False,
    no_launch: bool = False,
    python: str | None = None,
) -> list[str]:
    """构造后台 daemon 子进程的 argv。

    强调顺序：``--config`` 是主命令的选项，必须写在子命令 ``daemon`` **之前**。
    写成后面会让 daemon 子解析器报 "unrecognized arguments: --config" 而直接退出
    （这正是 `gpu-manager start` 曾经一启动就失败的原因）。
    tests/test_cli.py 会用**真实 parser** 解析这里返回的 argv 来锁定这个顺序。
    """
    argv = [
        python or sys.executable,
        "-m", "gpu_manager",
        "--config", str(config_path),
        "daemon",
    ]
    if dry_run:
        argv.append("--dry-run")
    if no_launch:
        argv.append("--no-launch")
    return argv


# --------------------------------------------------------------------------- #
# CLI 主体
# --------------------------------------------------------------------------- #
class Cli:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.style = Style(enabled=not args.no_color)
        self.logger = build_logger("gpu_manager.cli", None, level="WARNING", console=False)
        self.cfg: Config = load_config(args.config)
        self.store = StateStore(self.cfg.state_file, self.logger)
        self.queue = QueueManager(self.cfg, self.logger)
        self.runner = TaskRunner(self.cfg, self.logger)

    # ------------------------------------------------------------------ #
    # 通用
    # ------------------------------------------------------------------ #
    def out(self, text: str = "") -> None:
        print(text)

    def err(self, text: str) -> None:
        print(text, file=sys.stderr)

    def daemon_request(
        self, command: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any] | None:
        return send_request(self.cfg.socket_file, command, params, timeout=timeout)

    def daemon_pid(self) -> int | None:
        return self.store.read_pid(self.cfg.pid_file)

    def pid_is_our_daemon(self, pid: int | None) -> bool:
        """确认 pid 真的是我们的 daemon（owner + cmdline 双重校验，避免误杀）。"""
        if not pid or pid <= 0:
            return False
        if not process_alive(pid):
            return False
        if proc_uid(pid) != os.getuid():
            return False
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        args = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if not args:
            return False
        return any("gpu_manager" in a or "gpu-manager" in a for a in args)

    def daemon_online(self) -> bool:
        return socket_in_use(self.cfg.socket_file)

    def offline_queue(self) -> QueueManager:
        """daemon 离线时：直接从 state.json 载入队列（加锁，避免与其它 CLI 冲突）。"""
        if not self.queue.tasks:
            state = self.store.read()
            self.queue.load(state)
        return self.queue

    def persist_offline(self, reason: str) -> None:
        with self.store._lock:  # noqa: SLF001 - 同一进程内共享锁对象
            state = self.store.read()
            self.queue.dump(state)
            self.store.write(state)
        self.store.append_event({"event": reason, "source": "cli-offline"}, self.cfg.event_log)

    # ------------------------------------------------------------------ #
    # start / stop / reload
    # ------------------------------------------------------------------ #
    def cmd_start(self, args: argparse.Namespace) -> int:
        pid = self.daemon_pid()
        if self.pid_is_our_daemon(pid):
            self.out(f"daemon 已在运行: pid={pid}")
            return 0
        if pid:
            self.err(f"发现陈旧的 PID 文件（pid={pid} 不是本程序的 daemon），已清理")
            self.store.clear_pid(self.cfg.pid_file)

        if args.foreground:
            from .daemon import Daemon

            return Daemon(
                self.cfg,
                dry_run=args.dry_run,
                no_launch=args.no_launch,
                once=False,
                console=True,
            ).run()

        # 后台常驻: 等价 nohup + setsid，父进程退出/SSH 断开都不影响
        if not args.json:
            self.out("正在启动 daemon（后台常驻，SSH 断开不影响）…")
        out_path = self.cfg.log_file.with_suffix(".out.log")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = build_daemon_argv(
            self.cfg.source_path or (PROJECT_ROOT / "config.yaml"),
            dry_run=args.dry_run,
            no_launch=args.no_launch,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)

        # 子进程 stdout 只用于捕获意外崩溃的堆栈；文件位置由用户配置的 log_file 派生
        # pi-lens-ignore: python-path-traversal -- 路径来自配置，非外部输入拼接
        with open(out_path, "a", encoding="utf-8") as out_fh:
            proc = subprocess.Popen(  # noqa: S603 - argv 显式构造
                argv,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        def ready() -> bool:
            return bool(self.daemon_pid()) and self.daemon_online()

        # 谓词等待而不是固定 sleep: 就绪即返回，子进程早退也能立即发现
        wait_until(
            lambda: proc.poll() is not None or ready(),
            max(1.0, args.wait),
            interval=0.25,
        )
        if proc.poll() is not None:
            self.err(f"daemon 启动失败（退出码 {proc.returncode}），最后日志:")
            self.err("\n".join(self._tail_file(out_path, 25)))
            return 1
        new_pid = self.daemon_pid()
        if new_pid and self.daemon_online():
            if args.json:
                print(json.dumps({"ok": True, "pid": new_pid}, ensure_ascii=False))
            else:
                self.out(self.style(f"✔ daemon 已启动: pid={new_pid}", "green"))
                self.out(f"  日志: {self.cfg.log_file}")
                self.out("  控制: gpu-manager status / logs -f / stop")
            return 0

        self.err("daemon 启动超时（进程仍在运行，请用 `gpu-manager status` 检查）")
        self.err("\n".join(self._tail_file(out_path, 15)))
        return 1

    def cmd_stop(self, args: argparse.Namespace) -> int:
        pid = self.daemon_pid()
        running_tasks: list[str] = []
        state = self.store.read()
        running_tasks = [t.get("name", "?") for t in state.get("tasks", []) if t.get("status") == RUNNING]

        if not self.pid_is_our_daemon(pid):
            if pid:
                self.store.clear_pid(self.cfg.pid_file)
            if self.daemon_online():
                self.err("socket 存在但 PID 文件无效；请检查是否有另一个 daemon（gpu-manager status）")
                return 1
            self.out("daemon 未在运行")
            return 0

        if pid is None:
            self.out("daemon 未在运行")
            return 0

        graceful = False
        response = self.daemon_request("stop_daemon", timeout=10.0)
        if response and response.get("ok"):
            graceful = True
        else:
            self.err("socket 无响应，改用 SIGTERM（只针对 daemon 进程本身）")
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)

        if not graceful:
            self.err("（已回退到 SIGTERM，仅针对 daemon 本身）")

        # 复用全项目唯一的轮询原语（不手写 while + sleep：条件一满足就立即返回）
        if wait_until(lambda: not process_alive(pid), max(1.0, args.timeout), interval=0.4):
            self.store.clear_pid(self.cfg.pid_file)
            self.out(self.style(f"✔ daemon 已停止 (pid={pid})", "green"))
            self._warn_running_tasks(running_tasks)
            return 0

        if args.force:
            self.err("daemon 未在超时内退出，发送 SIGKILL（仅 daemon 本身）")
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
            wait_until(lambda: not process_alive(pid), 5.0, interval=0.2)
            if not process_alive(pid):
                self.store.clear_pid(self.cfg.pid_file)
                self.out("✔ daemon 已被强制停止")
                self._warn_running_tasks(running_tasks)
                return 0
        self.err(f"daemon (pid={pid}) 仍未退出；可用 `gpu-manager stop --force`")
        return 1

    def _warn_running_tasks(self, running: list[str]) -> None:
        if running:
            self.out(
                self.style(
                    f"注意: {len(running)} 个训练任务仍在运行（未被杀死）: {', '.join(running[:5])}",
                    "yellow",
                )
            )
            self.out("这些任务在 daemon 重启前不会被更新状态；重启后会自动接管监控。")

    def cmd_restart(self, args: argparse.Namespace) -> int:
        stop_args = argparse.Namespace(force=False, timeout=30.0, json=getattr(args, "json", False))
        self.cmd_stop(stop_args)
        start_args = argparse.Namespace(
            foreground=False, dry_run=False, no_launch=False, wait=20.0, json=getattr(args, "json", False)
        )
        return self.cmd_start(start_args)

    def cmd_reload(self, args: argparse.Namespace) -> int:
        response = self.daemon_request("reload", timeout=30.0)
        if response is None:
            self.err("daemon 未在运行（离线模式下配置在下次启动时自然生效）")
            return 1
        if not response.get("ok"):
            self.err(f"重载失败: {response.get('error')}")
            return 1
        data = response.get("data", {})
        self.out(f"✔ 配置已重载；tasks.yaml 同步: {data.get('import', '')}")
        return 0

    # ------------------------------------------------------------------ #
    # status
    # ------------------------------------------------------------------ #
    def cmd_status(self, args: argparse.Namespace) -> int:
        if args.watch is not None:
            return self._watch_status(args)
        return self._print_status(args)

    def _daemon_payload(self, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
        response = self.daemon_request("status", timeout=8.0)
        if response and response.get("ok") and isinstance(response.get("data"), dict):
            return response["data"], "daemon"
        state = self.store.read()
        counts: dict[str, int] = dict.fromkeys([PENDING, RUNNING, COMPLETED, FAILED, CANCELLED], 0)
        for task in state.get("tasks", []):
            status = str(task.get("status", PENDING))
            counts[status] = counts.get(status, 0) + 1
        payload = {
            "daemon": {
                "pid": state.get("daemon_pid"),
                "version": state.get("daemon_version", "?"),
                "started_at": state.get("started_at"),
                "uptime_seconds": (time.time() - state["started_at"]) if state.get("started_at") else 0,
                "dry_run": False,
                "no_launch": False,
                "offline": True,
            },
            "mode": state.get("mode") or "IDLE",
            "mode_since": state.get("mode_since") or 0,
            "server": self.cfg.server.resolved_name(),
            "user": self.cfg.server.resolved_user(),
            "counts": counts,
            "stats": state.get("stats") or {},
            "tasks": state.get("tasks") or [],
            "idle": {
                "required_checks": self.cfg.gpu.idle_confirm_checks,
                "counts": (state.get("idle") or {}).get("counts", {}),
                "since": (state.get("idle") or {}).get("since", {}),
                "confirmed": [],
            },
            "waiting": state.get("waiting") or [],
            "plan": {"assignments": []},
            "watchdog": state.get("watchdog") or {},
            "source": "state.json",
        }
        return payload, "state"

    def _print_status(self, args: argparse.Namespace) -> int:
        payload, source = self._daemon_payload(args)
        offline = source == "state"

        if args.stale and offline:
            snapshot_rows = (self.store.read().get("last_snapshot") or {}).get("gpus") or []
            snapshot_error = (self.store.read().get("last_snapshot") or {}).get("error") or ""
        else:
            snapshot = collect(timeout=15.0, include_processes=True)
            snapshot_rows = as_dict_rows(snapshot)
            snapshot_error = snapshot.error

        if self.args.json:
            print(
                json.dumps(
                    {
                        "daemon": payload.get("daemon"),
                        "mode": payload.get("mode"),
                        "counts": payload.get("counts"),
                        "tasks": payload.get("tasks"),
                        "gpus": snapshot_rows,
                        "gpu_error": snapshot_error,
                        "source": source,
                        "gpu_policy": payload.get("gpu_policy"),
                        "interactive": payload.get("interactive")
                        or {"enabled": self.cfg.feishu.interactive_enabled, "running": False},
                        "watchdog": payload.get("watchdog") or {},
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

        allowed = set(self.cfg.gpu.schedulable_set([int(r["index"]) for r in snapshot_rows]))
        idle_counts = payload.get("idle", {}).get("counts", {})
        required = payload.get("idle", {}).get("required_checks", self.cfg.gpu.idle_confirm_checks)

        self.out(self.style(f"GPU Manager {__version__} · {payload.get('server')} (用户 {payload.get('user')})", "cyan"))
        daemon_info = payload.get("daemon") or {}
        pid = daemon_info.get("pid")
        if source == "daemon":
            uptime = fmt_duration(float(daemon_info.get("uptime_seconds") or 0))
            flags = []
            if daemon_info.get("dry_run"):
                flags.append("dry-run")
            if daemon_info.get("no_launch"):
                flags.append("no-launch")
            flag_text = f" [{', '.join(flags)}]" if flags else ""
            self.out(f"daemon: 运行中 pid={pid} 已运行 {uptime}{flag_text}")
        else:
            self.out(self.style("daemon: 未运行（以下为上次持久化状态）", "yellow"))
        mode = payload.get("mode", "IDLE")
        mode_color = {"IDLE": "green", "QUEUE": "yellow", "RUNNING": "blue"}.get(mode, "cyan")
        mode_since = float(payload.get("mode_since") or 0)
        since_text = f"（持续 {fmt_duration(time.time() - mode_since)}）" if mode_since else ""
        counts = payload.get("counts", {})
        self.out(
            f"模式: {self.style(mode, mode_color)} {since_text}   "
            f"队列: pending={counts.get('pending', 0)} running={counts.get('running', 0)} "
            f"completed={counts.get('completed', 0)} failed={counts.get('failed', 0)} "
            f"cancelled={counts.get('cancelled', 0)}"
        )
        # 交互查询（默认关闭；只开了才显示，避免干扰日常输出）
        interactive = payload.get("interactive")
        if not isinstance(interactive, dict):
            interactive = {"enabled": self.cfg.feishu.interactive_enabled, "running": False}
        if interactive.get("enabled"):
            state_text = (
                f"运行中（已授权 {interactive.get('allowed_users', 0)} 人）"
                if interactive.get("running")
                else "已启用但未运行 —— 用 `gpu-manager doctor` 检查凭证 / SDK"
            )
            self.out(f"交互查询: {state_text}")
        # 看门狗：只在真有任务被判定异常时提示一句（正常时保持输出干净）
        watchdog = payload.get("watchdog")
        if isinstance(watchdog, dict) and watchdog.get("enabled"):
            for name, info in (watchdog.get("tasks") or {}).items():
                if not isinstance(info, dict) or not info.get("alert_kind"):
                    continue
                self.out(
                    self.style(
                        f"看门狗: {name} → {info.get('alert_reason', '异常')}"
                        f"（诊断: {info.get('last_snapshot') or '—'}）",
                        "yellow",
                    )
                )
        self.out("")

        if snapshot_error and not snapshot_rows:
            self.out(self.style(f"GPU 状态不可用: {snapshot_error}", "red"))
        else:
            headers = ["GPU", "UTIL", "显存已用/总量", "空闲确认", "可自动调度", "占用者(compute)"]
            rows: list[list[str]] = []
            for row in snapshot_rows:
                index = int(row["index"])
                streak = int(idle_counts.get(str(index), 0))
                procs = row.get("compute_processes") or []
                owners = ", ".join(
                    f"pid {p['pid']}({'自己' if p.get('ours') else 'uid' + str(p.get('uid'))})"
                    for p in procs[:2]
                ) or "-"
                if len(procs) > 2:
                    owners += f" +{len(procs) - 2}"
                util = row.get("utilization")
                util_text = "n/a" if util is None else f"{util}%"
                util_disp = f"{util_text}"
                if util is not None and util > self.cfg.gpu.max_idle_utilization:
                    util_disp = self.style(util_text, "yellow")
                confirm_text = f"{streak}/{required}" + (" ✔" if streak >= required else "")
                rows.append(
                    [
                        f"{index}",
                        util_disp,
                        f"{row.get('memory_used_mb')}/{row.get('memory_total_mb')} MiB",
                        confirm_text,
                        "是" if index in allowed else "否",
                        owners,
                    ]
                )
            self.out(render_table(headers, rows, ["right", "right", "right", "right", "center", "left"]))
            if snapshot_error:
                self.out(self.style(f"注意: 采集有告警: {snapshot_error}", "yellow"))

        waiting = payload.get("waiting") or []
        tasks = payload.get("tasks") or []
        pending_tasks = [t for t in tasks if t.get("status") == PENDING]
        running_tasks = [t for t in tasks if t.get("status") == RUNNING]
        if pending_tasks:
            self.out("")
            self.out(self.style(f"等待中的任务 ({len(pending_tasks)})", "yellow"))
            headers = ["任务", "需要GPU", "优先级", "已等待", "等待原因"]
            reason_map = {str(w.get("name")): str(w.get("reason", "")) for w in waiting}
            rows = [
                [
                    str(t.get("name")),
                    str(t.get("gpu_num")),
                    str(t.get("priority")),
                    fmt_duration(time.time() - float(t.get("created_at") or time.time())),
                    truncate(reason_map.get(str(t.get("name")), "资源检查中"), 46),
                ]
                for t in sorted(pending_tasks, key=lambda x: (-int(x.get("priority") or 0), int(x.get("seq") or 0)))
            ]
            self.out(render_table(headers, rows, ["left", "right", "right", "right", "left"]))
        if running_tasks:
            self.out("")
            self.out(self.style(f"运行中的任务 ({len(running_tasks)})", "blue"))
            headers = ["任务", "GPU", "PID", "已运行", "日志"]
            rows = [
                [
                    str(t.get("name")),
                    gpu_ids_text(t.get("gpu_ids") or []),
                    str(t.get("pid") or "-"),
                    fmt_duration(time.time() - float(t.get("started_at") or time.time())),
                    str(t.get("log_file") or "-"),
                ]
                for t in running_tasks
            ]
            self.out(render_table(headers, rows))
        return 0

    def _watch_status(self, args: argparse.Namespace) -> int:
        interval = max(1.0, float(args.watch or 2.0))
        # 用 Event.wait 代替 sleep：刷新间隔语义不变，但可被立即打断
        wake = threading.Event()
        with contextlib.suppress(OSError, ValueError):
            signal.signal(signal.SIGINT, lambda *_: wake.set())
        try:
            while not wake.is_set():
                if not args.json:
                    print("\033[2J\033[H", end="")
                self._print_status(args)
                if not args.json:
                    self.out("")
                    self.out(self.style(f"每 {interval:g}s 刷新，Ctrl-C 退出", "dim"))
                wake.wait(interval)
        except KeyboardInterrupt:
            self.out("")  # Ctrl-C: 换行结束这一屏
        return 0

    # ------------------------------------------------------------------ #
    # 任务管理
    # ------------------------------------------------------------------ #
    def _load_tasks_for_read(self) -> tuple[list[Task], dict[str, Any]]:
        response = self.daemon_request("status", timeout=8.0)
        if response and response.get("ok"):
            data = response.get("data") or {}
            tasks = [Task.from_dict(t) for t in data.get("tasks", []) if isinstance(t, dict)]
            return tasks, data
        state = self.store.read()
        queue = QueueManager(self.cfg, self.logger)
        queue.load(state)
        return queue.ordered_tasks(), {
            "mode": state.get("mode") or "IDLE",
            "waiting": state.get("waiting") or [],
            "source": "state.json",
        }

    def cmd_list(self, args: argparse.Namespace) -> int:
        tasks, meta = self._load_tasks_for_read()
        if args.status:
            tasks = [t for t in tasks if t.status == args.status]
        if args.pending:
            tasks = [t for t in tasks if t.is_pending]
        if not args.all and not args.status and not args.pending:
            # 默认: 活跃任务在前，终态任务在后（不隐藏，便于一眼看全）
            tasks = sorted(tasks, key=lambda t: (t.is_terminal, -t.priority, t.seq))

        if args.json:
            print(json.dumps([t.to_dict() for t in tasks], ensure_ascii=False, indent=2, default=str))
            return 0
        if not tasks:
            self.out("(无任务)")
            return 0

        headers = ["", "任务", "状态", "优先级", "GPU数", "GPU", "PID", "已运行/耗时", "退出码", "备注"]
        rows: list[list[str]] = []
        for task in tasks:
            runtime = task.runtime_seconds()
            runtime_text = fmt_duration(runtime) if runtime is not None else fmt_duration(task.wait_seconds())
            note = task.error or task.notes or ""
            if task.is_pending and task.retries:
                note = f"重试 {task.retries}/{task.max_retries} " + note
            rows.append(
                [
                    STATUS_ICON.get(task.status, "?"),
                    task.name,
                    task.status,
                    str(task.priority),
                    str(task.gpu_num),
                    task.gpu_text,
                    str(task.pid or "-"),
                    runtime_text,
                    "-" if task.exit_code is None else str(task.exit_code),
                    truncate(note, 40),
                ]
            )
        self.out(render_table(headers, rows, ["center", "left", "left", "right", "right", "right", "right", "right", "right", "left"]))
        counts = {}
        for task in self.queue.ordered_tasks() if False else tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        self.out("")
        self.out(self.style("总计: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "dim"))
        if meta.get("source") == "state.json":
            self.out(self.style("(daemon 未运行，状态来自 state.json；重启 daemon 后继续调度)", "yellow"))
        return 0

    def cmd_show(self, args: argparse.Namespace) -> int:
        tasks, _meta = self._load_tasks_for_read()
        task = next((t for t in tasks if t.name == args.name), None)
        if task is None:
            self.err(f"任务不存在: {args.name}")
            return 1
        if args.json:
            print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2, default=str))
            return 0
        self.out(self.style(f"任务 {task.name}", "cyan"))
        rows = [
            ["状态", task.status],
            ["优先级", str(task.priority)],
            ["需要 GPU 数", str(task.gpu_num)],
            ["已分配 GPU", task.gpu_text if task.gpu_ids else "-"],
            ["工作目录", task.cwd],
            ["conda 环境", task.env or "(未指定)"],
            ["额外环境变量", ", ".join(f"{k}={v}" for k, v in task.env_vars.items()) or "-"],
            ["提交来源", task.source],
            ["创建时间", fmt_ts(task.created_at)],
            ["开始时间", fmt_ts(task.started_at) if task.started_at else "-"],
            ["结束时间", fmt_ts(task.finished_at) if task.finished_at else "-"],
            ["运行时长", fmt_duration(task.runtime_seconds()) if task.started_at else "-"],
            ["PID", str(task.pid or "-")],
            ["退出码", "-" if task.exit_code is None else str(task.exit_code)],
            ["重试", f"{task.retries}/{task.max_retries}"],
            ["最低空闲显存", f"{task.min_free_memory_mb} MiB" if task.min_free_memory_mb else "-"],
            ["超时", f"{task.timeout_seconds}s" if task.timeout_seconds else "-"],
            ["日志", task.log_file or "-"],
            ["错误/备注", task.error or task.notes or "-"],
        ]
        self.out(render_table(["字段", "值"], rows))
        self.out("")
        self.out("实际执行的命令:")
        self.out(self.style(self.runner.describe_command(task), "dim"))
        return 0

    def cmd_add(self, args: argparse.Namespace) -> int:
        env_vars: dict[str, str] = {}
        for item in args.env_var:
            if "=" not in item:
                self.err(f"--env-var 需要 K=V 格式: {item}")
                return 2
            key, _, value = item.partition("=")
            env_vars[key.strip()] = value

        defaults = self.cfg.task_defaults
        spec: dict[str, Any] = {
            "name": args.name,
            "cwd": args.cwd or os.getcwd(),
            "command": args.cmd,
            "env": args.env if args.env is not None else defaults.env,
            "gpu_num": args.gpu_num if args.gpu_num is not None else defaults.gpu_num,
            "priority": args.priority if args.priority is not None else defaults.priority,
            "min_free_memory_mb": (
                args.min_free_memory_mb if args.min_free_memory_mb is not None else defaults.min_free_memory_mb
            ),
            "max_retries": args.max_retries if args.max_retries is not None else defaults.max_retries,
            "retry_delay_seconds": (
                args.retry_delay_seconds if args.retry_delay_seconds is not None else defaults.retry_delay_seconds
            ),
            "timeout_seconds": (
                args.timeout_seconds if args.timeout_seconds is not None else defaults.timeout_seconds
            ),
            "notes": args.notes,
            "env_vars": env_vars,
        }

        # 先本地校验（daemon 在线也先校验，快速报错）
        from .task_queue import build_task, validate_cwd

        try:
            preview = build_task(spec, defaults, self.cfg, source="cli")
        except TaskError as exc:
            self.err(f"任务定义有误: {exc}")
            return 2
        cwd_error = validate_cwd(preview)
        if cwd_error:
            if args.json:
                print(json.dumps({"ok": False, "error": cwd_error}, ensure_ascii=False))
            self.err(f"任务定义有误: {cwd_error}")
            return 2
        if preview.env:
            exists, detail = self.runner.env_exists(preview.env)
            if not exists:
                self.err(f"警告: conda 环境 {preview.env!r} 未找到（{detail}）")

        if args.dry_run:
            if args.json:
                print(json.dumps({"ok": True, "dry_run": True, "task": preview.to_dict()}, ensure_ascii=False, indent=2, default=str))
            else:
                self.out("校验通过（--dry-run 未入队）:")
                self.out(f"  任务: {preview.name}")
                self.out(f"  命令: {preview.command}")
                self.out(f"  目录: {preview.cwd}")
                self.out(f"  环境: {preview.env or '(未指定)'}")
                self.out(f"  GPU : {preview.gpu_num} 卡, 优先级 {preview.priority}")
            return 0

        response = self.daemon_request("add", {"spec": spec, "replace": args.replace})
        if response is None:
            queue = self.offline_queue()
            try:
                task = queue.add(spec, source="cli", replace=args.replace)
            except TaskError as exc:
                self.err(f"入队失败: {exc}")
                return 2
            self.persist_offline("task_added")
            if args.json:
                print(json.dumps({"ok": True, "task": task.to_dict(), "offline": True}, ensure_ascii=False, indent=2, default=str))
            else:
                self.out(self.style(f"✔ 任务已入队: {task.name}", "green"))
                self.out(self.style("  daemon 未运行 —— 启动 daemon 后会自动开始调度: gpu-manager start", "yellow"))
            return 0

        if not response.get("ok"):
            self.err(f"入队失败: {response.get('error')}")
            return 2
        task_data = (response.get("data") or {}).get("task") or {}
        if args.json:
            print(json.dumps({"ok": True, "task": task_data}, ensure_ascii=False, indent=2, default=str))
        else:
            self.out(self.style(f"✔ 任务已入队: {task_data.get('name')}", "green"))
            self.out(f"  GPU: {task_data.get('gpu_num')} 卡 · 优先级 {task_data.get('priority')}")
            self.out("  满足资源后会尽快自动启动；可随时 `gpu-manager list` 查看")
        return 0

    def cmd_cancel(self, args: argparse.Namespace) -> int:
        response = self.daemon_request("cancel", {"name": args.name})
        if response is None:
            queue = self.offline_queue()
            try:
                task = queue.cancel(args.name)
            except TaskError as exc:
                self.err(str(exc))
                return 1
            self.persist_offline("task_cancel_requested")
            if args.json:
                print(json.dumps({"ok": True, "task": task.to_dict()}, ensure_ascii=False, default=str))
            else:
                self.out(f"✔ 已取消 {task.name}（daemon 未运行，不会影响任何进程）")
            return 0
        if not response.get("ok"):
            self.err(f"取消失败: {response.get('error')}")
            return 1
        data = response.get("data") or {}
        if args.json:
            print(json.dumps({"ok": True, "task": data.get("task")}, ensure_ascii=False, default=str))
        else:
            self.out(f"✔ {data.get('message', '已请求取消')}: {args.name}")
            self.out("  只对自己启动的进程组发送 SIGTERM（grace 后 SIGKILL），不触碰其他用户")
        return 0

    def cmd_remove(self, args: argparse.Namespace) -> int:
        return self._simple_task_op("remove", args, f"✔ 已移除 {args.name}")

    def cmd_requeue(self, args: argparse.Namespace) -> int:
        return self._simple_task_op("requeue", args, f"✔ 已重新入队 {args.name}")

    def _simple_task_op(self, op: str, args: argparse.Namespace, message: str) -> int:
        response = self.daemon_request(op, {"name": args.name})
        if response is None:
            queue = self.offline_queue()
            try:
                task = getattr(queue, op)(args.name)
            except TaskError as exc:
                self.err(str(exc))
                return 1
            self.persist_offline(f"task_{op}")
            if args.json:
                print(json.dumps({"ok": True, "task": task.to_dict()}, ensure_ascii=False, default=str))
            else:
                self.out(message)
            return 0
        if not response.get("ok"):
            self.err(f"操作失败: {response.get('error')}")
            return 1
        if args.json:
            print(json.dumps({"ok": True, "task": (response.get('data') or {}).get("task")}, ensure_ascii=False, default=str))
        else:
            self.out(message)
        return 0

    # ------------------------------------------------------------------ #
    # 日志 / 事件 / 配置
    # ------------------------------------------------------------------ #
    @staticmethod
    @contextlib.contextmanager
    def _open_log(path: Path) -> Iterator[TextIO]:
        """打开日志文件（读取用）—— 全项目读取日志的**唯一入口**。

        路径只有两个来源: daemon 自己配置的日志路径，或已经过 resolve_within()
        校验的任务日志路径（见 _task_log_path）。不存在由外部输入拼出的路径。
        """
        # pi-lens-ignore: python-path-traversal -- 路径来自配置或已验证的任务日志路径
        with open(path, encoding="utf-8", errors="replace") as handle:
            yield handle

    @classmethod
    def _tail_file(cls, path: Path, lines: int) -> list[str]:
        if not path.exists():
            return []
        try:
            with cls._open_log(path) as fh:
                return fh.read().splitlines()[-max(1, lines):]
        except OSError:
            return []

    def _task_log_path(self, name: str) -> Path | None:
        """定位任务日志。只接受任务日志目录内的路径（防路径穿越/误读其他文件）。"""
        tasks, _ = self._load_tasks_for_read()
        task = next((t for t in tasks if t.name == name), None)
        if task and task.log_file:
            try:
                path = resolve_within(task.log_file, self.cfg.task_log_dir)
            except ValueError:
                path = None
            if path is not None and path.exists():
                return path
        # 退化: 按目录找最新日志
        safe = task.safe_name if task else name
        try:
            log_dir = resolve_within(self.cfg.task_log_dir / safe, self.cfg.task_log_dir)
        except ValueError:
            return None
        if not log_dir.is_dir():
            return None
        candidates = sorted(
            (p for p in log_dir.iterdir() if p.is_file() and p.suffix == ".log" and p.name != "latest.log"),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    def cmd_logs(self, args: argparse.Namespace) -> int:
        if args.name:
            path = self._task_log_path(args.name)
            if path is None:
                self.err(f"找不到任务 {args.name} 的日志（任务不存在或尚未运行过）")
                return 1
        else:
            path = self.cfg.log_file
            if not path.exists():
                self.err(f"daemon 日志不存在: {path}（daemon 还没运行过？）")
                return 1
        lines = self._tail_file(path, args.lines)
        self.out(self.style(f"# {path}", "dim"))
        for line in lines:
            self.out(line)
        if args.follow:
            try:
                with self._open_log(path) as fh:
                    fh.seek(0, os.SEEK_END)
                    observed = os.fstat(fh.fileno()).st_size
                    while True:
                        chunk = fh.readline()
                        if chunk:
                            self.out(chunk.rstrip("\n"))
                            continue
                        size = os.fstat(fh.fileno()).st_size
                        if size != observed:
                            observed = size
                            continue
                        # 等文件增长（不是固定 sleep）: 有新内容立刻继续输出
                        wait_until(
                            partial(_size_changed, fh.fileno(), observed),
                            60.0,
                            interval=0.3,
                        )
            except KeyboardInterrupt:
                self.out("")  # Ctrl-C: 换行结束
            except OSError as exc:
                self.err(f"跟踪日志失败: {exc}")
                return 1
        return 0

    def cmd_events(self, args: argparse.Namespace) -> int:
        events = StateStore.read_events(self.cfg.event_log, args.lines)
        if args.json:
            print(json.dumps(events, ensure_ascii=False, indent=2, default=str))
            return 0
        if not events:
            self.out("(无事件记录)")
            return 0
        for event in events:
            ts = fmt_ts(float(event.get("ts") or 0))
            kind = str(event.get("event", "?"))
            detail = {k: v for k, v in event.items() if k not in ("ts", "event")}
            icon = {
                "task_finished": "✔",
                "task_started": "▶",
                "task_added": "＋",
                "task_cancel_requested": "⊘",
                "task_retry": "↻",
                "mode_change": "⇄",
                "daemon_start": "⏻",
                "daemon_stop": "⏼",
                "launch_aborted": "✘",
            }.get(kind, LEVEL_ICON.get("info", "·"))
            self.out(f"{ts}  {icon} {kind:<22} {truncate(json.dumps(detail, ensure_ascii=False), 110)}")
        return 0

    def cmd_config(self, args: argparse.Namespace) -> int:
        if args.raw:
            if not self.cfg.source_path or not self.cfg.source_path.exists():
                self.err("配置文件不存在")
                return 1
            self.out(self.cfg.source_path.read_text(encoding="utf-8"))
            return 0
        data = self.cfg.as_dict()
        # 打码密钥
        feishu = data.get("feishu")
        if isinstance(feishu, dict):
            for key in ("app_secret", "app_id", "receive_id"):
                if feishu.get(key):
                    feishu[key] = _mask(str(feishu[key]))
        data["resolved_paths"] = {
            "config": str(self.cfg.source_path) if self.cfg.source_path else "(默认值)",
            "state_file": str(self.cfg.state_file),
            "log_file": str(self.cfg.log_file),
            "task_log_dir": str(self.cfg.task_log_dir),
            "queue_file": str(self.cfg.queue_file),
            "pid_file": str(self.cfg.pid_file),
            "socket_file": str(self.cfg.socket_file),
        }
        data["computed_gpu_scope"] = {
            "monitor_gpus": self.cfg.gpu.monitor_gpu_ids or "all",
            "scheduler_allowed_gpus": self.cfg.gpu.scheduler_allowed_gpus or "all",
            "reserved_gpus": self.cfg.gpu.reserved_gpu_ids,
        }
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        return 0

    # ------------------------------------------------------------------ #
    # doctor
    # ------------------------------------------------------------------ #
    def cmd_doctor(self, args: argparse.Namespace) -> int:
        checks: list[tuple[str, bool, str]] = []

        # 1) 配置文件
        checks.append(("配置文件解析", True, str(self.cfg.source_path or "(内置默认值)")))

        # 2) nvidia-smi
        snapshot = collect(timeout=15.0, include_processes=True)
        if snapshot.ok:
            names = {g.name for g in snapshot.gpus}
            checks.append(
                ("nvidia-smi", True, f"{len(snapshot.gpus)} 张卡: {', '.join(sorted(names))}")
            )
        else:
            checks.append(("nvidia-smi", False, snapshot.error))

        # 3) 进程归属读取能力（安全边界前提）
        can_read_uid = all(proc_uid(p.pid) is not None for g in snapshot.gpus for p in g.compute_processes[:2])
        checks.append(
            (
                "进程归属可见性(/proc)",
                can_read_uid,
                "可读取其他用户进程 uid —— 判断「是否他人占用」可靠"
                if can_read_uid
                else "无法读取部分进程 uid（可能受 hidepid 影响），判定会更保守",
            )
        )

        # 4) conda
        conda = self.runner.conda_bin()
        if conda:
            envs = self.runner.list_envs()
            checks.append(("conda", True, f"{conda}（{len(envs)} 个环境: {', '.join(envs[:8])}）"))
        else:
            checks.append(("conda", False, "未找到 conda；如任务需要 conda 环境请在 runner.conda_bin 指定"))

        # 5) 目录可写
        for label, path in (("状态目录", self.cfg.state_file.parent), ("任务日志目录", self.cfg.task_log_dir)):
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / f".write-test-{os.getpid()}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                checks.append((f"{label}可写", True, str(path)))
            except OSError as exc:
                checks.append((f"{label}可写", False, f"{path} ({exc})"))

        # 6) daemon 状态
        pid = self.daemon_pid()
        if self.pid_is_our_daemon(pid):
            checks.append(("daemon", True, f"运行中 pid={pid}（socket {'在线' if self.daemon_online() else '离线'}）"))
        else:
            checks.append(("daemon", True, "未运行（gpu-manager start 可启动）"))

        # 6b) 看门狗（只读检查；没有任务时会自动关闭，不是问题）
        wcfg = self.cfg.watchdog
        if not wcfg.enabled:
            checks.append(("看门狗", True, "已禁用（watchdog.enabled=false）"))
        else:
            diag_dir = self.cfg.diagnostics_dir
            checks.append(
                (
                    "看门狗",
                    True,
                    f"{wcfg.summary()}；诊断快照 {human_size(dir_usage_bytes(diag_dir))}"
                    f"（{diag_dir}）",
                )
            )

        # 7) 飞书
        if not self.cfg.feishu.enabled:
            checks.append(("飞书通知", True, "已禁用（feishu.enabled=false）"))
        elif not (self.cfg.feishu.app_id and self.cfg.feishu.app_secret):
            checks.append(
                ("飞书通知", False, "缺少 app_id/app_secret（开放平台 → 自建应用 → 凭证与基础信息）")
            )
        else:
            target = self.cfg.feishu.resolved_receive_id()
            detail = (
                f"自建应用, card={self.cfg.feishu.card}, "
                f"目标={target or '未配置（填 receive_id 或 allowed_users）'}"
            )
            if args.notify_test:
                from .notifier import build_notifier

                notifier = build_notifier(self.cfg.feishu, self.logger, dry_run=False)
                ok_sent, msg = notifier.send(
                    "🧪 GPU Manager 连通性测试",
                    f"**服务器**: {self.cfg.server.resolved_name()}\n\n"
                    "如果你看到这条消息，说明飞书通知配置正确。",
                    level="info",
                )
                checks.append(("飞书发送测试", ok_sent, msg))
            else:
                checks.append(("飞书通知", True, detail + "（加 --notify-test 实测发送）"))

        # 7b) 飞书交互式查询（与上面的通知共用同一份自建应用凭证）
        fcfg = self.cfg.feishu
        if not fcfg.interactive_enabled:
            checks.append(("飞书交互查询", True, "已关闭（feishu.interactive_enabled=false）"))
        elif not (fcfg.app_id and fcfg.app_secret):
            checks.append(
                (
                    "飞书交互查询",
                    False,
                    "feishu.interactive_enabled=true 但 app_id/app_secret 未填 → 不会启动"
                    "（开放平台 → 自建应用 → 凭证与基础信息）",
                )
            )
        else:
            notes: list[str] = []
            # 实测换一次 tenant_access_token（只读）:
            # 用来区分「凭证错」与「凭证对但事件没推过来」—— 后者是开放平台侧配置问题。
            from .notifier import _http_post_json

            ok_http, data = _http_post_json(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                {"app_id": fcfg.app_id, "app_secret": fcfg.app_secret},
                timeout=float(self.cfg.feishu.timeout_seconds),
                allowed_hosts=self.cfg.feishu.allowed_hosts,
                allow_any_host=self.cfg.feishu.allow_any_host,
            )
            token_ok = bool(
                ok_http
                and isinstance(data, dict)
                and data.get("code") in (0, None)
                and data.get("tenant_access_token")
            )
            if not ok_http:
                notes.append(f"无法访问飞书开放平台: {data}")
            elif not token_ok:
                reason = data.get("msg") if isinstance(data, dict) else data
                notes.append(f"凭证无效: {reason}")
            else:
                notes.append("凭证有效 ✓")
            if importlib.util.find_spec("lark_oapi") is None:
                notes.append("未安装 SDK: python3 -m pip install lark-oapi")
            if fcfg.allowed_users:
                notes.append(f"已授权 {len(fcfg.allowed_users)} 人")
            else:
                notes.append("白名单为空 → 拒绝所有人；先发一条「状态」，再从日志取 open_id")
                if token_ok:
                    notes.append(
                        "若发消息后日志里连「收到未授权消息」都没有 → "
                        "查开放平台: ① 事件订阅已加 im.message.receive_v1 且订阅方式为「长连接」"
                        " ② 已授予 im:message ③ 版本已发布生效（群里需 @机器人）"
                    )
            checks.append(("飞书交互查询", token_ok, "；".join(notes)))

        # 8) tasks.yaml
        try:
            specs = self.queue.read_queue_file()
            tasks_ok = True
            notes: list[str] = []
            for spec in specs:
                name = str(spec.get("name", "?"))
                try:
                    task = __import__("gpu_manager.task_queue", fromlist=["build_task"]).build_task(
                        spec, self.cfg.task_defaults, self.cfg, source="tasks.yaml"
                    )
                except TaskError as exc:
                    tasks_ok = False
                    notes.append(f"{name}: {exc}")
                    continue
                cwd_error = __import__("gpu_manager.task_queue", fromlist=["validate_cwd"]).validate_cwd(task)
                if cwd_error:
                    tasks_ok = False
                    notes.append(f"{name}: {cwd_error}")
                elif task.env:
                    exists, detail = self.runner.env_exists(task.env)
                    if not exists:
                        tasks_ok = False
                        notes.append(f"{name}: conda 环境 {task.env} 不存在（{detail}）")
            checks.append(
                (
                    f"tasks.yaml ({len(specs)} 个任务)",
                    tasks_ok,
                    "; ".join(notes[:5]) if notes else "全部任务定义有效",
                )
            )
        except TaskError as exc:
            checks.append(("tasks.yaml", False, str(exc)))

        # 输出
        if args.json:
            print(
                json.dumps(
                    [{"check": c, "ok": o, "detail": d} for c, o, d in checks],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if all(o for _c, o, _d in checks) else 1
        self.out(self.style(f"GPU Manager {__version__} 环境自检", "cyan"))
        self.out("")
        rows = [
            [self.style("✔", "green") if ok else self.style("✘", "red"), check, detail]
            for check, ok, detail in checks
        ]
        self.out(render_table(["", "检查项", "结果"], rows))
        self.out("")
        if all(ok for _c, ok, _d in checks):
            self.out(self.style("全部通过。", "green"))
            return 0
        self.out(self.style("存在问题，请按上面提示修复。", "yellow"))
        return 1

    # ------------------------------------------------------------------ #
    # watchdog
    # ------------------------------------------------------------------ #
    def cmd_watchdog(self, args: argparse.Namespace) -> int:
        """看门狗状态与诊断快照（只读）。

        status      —— 是否在监视、每个任务的「最后进展」与告警情况
        diagnostics —— 列出（或查看）告警时留下的诊断快照
        """
        action = getattr(args, "action", "status") or "status"
        if action == "diagnostics":
            return self._watchdog_diagnostics(args)
        return self._watchdog_status(args)

    def _watchdog_status(self, args: argparse.Namespace) -> int:
        payload, source = self._daemon_payload(args)
        data = payload.get("watchdog") or {}
        if args.json:
            print(
                json.dumps(
                    {"watchdog": data, "source": source},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0
        for line in format_watchdog_status(data, now=time.time()):
            self.out(line)
        if not data:
            self.out("")
            hint = (
                "提示: 运行中的 daemon 是升级前启动的旧版本 —— `gpu-manager restart` 后即可看到看门狗状态"
                if source == "daemon"
                else "提示: 未读到看门狗状态（daemon 未运行，且 state.json 里还没有 watchdog 段）"
            )
            self.out(self.style(hint, "dim"))
        return 0

    def _watchdog_diagnostics(self, args: argparse.Namespace) -> int:
        root = self.cfg.diagnostics_dir
        task_filter = getattr(args, "task", None)
        entries: list[tuple[str, str, Path, int, float]] = []
        if root.is_dir():
            for task_dir in sorted(root.iterdir()):
                if task_dir.is_symlink() or not task_dir.is_dir():
                    continue
                if task_filter and task_dir.name != task_filter:
                    continue
                for snap in sorted(task_dir.iterdir()):
                    if snap.is_symlink() or not snap.is_dir():
                        continue
                    stat = snap.stat()
                    entries.append(
                        (task_dir.name, snap.name, snap, dir_usage_bytes(snap), stat.st_mtime)
                    )
        # 目录名就是快照时间戳（同一秒的重复抓取带 -N 后缀）——以它为主键排序，
        # 保证「最新一份」是确定性的；mtime 只作为同名时的补充键。
        entries.sort(key=lambda e: (e[1], e[4]), reverse=True)

        if args.json:
            print(
                json.dumps(
                    {
                        "diagnostics_dir": str(root),
                        "limits": limits_text(self.cfg),
                        "count": len(entries),
                        "bytes": sum(e[3] for e in entries),
                        "snapshots": [
                            {
                                "task": task,
                                "snapshot": name,
                                "path": str(path),
                                "bytes": size,
                                "mtime": mtime,
                            }
                            for task, name, path, size, mtime in entries
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

        if not entries:
            self.out(f"没有诊断快照（目录: {root}）")
            if task_filter:
                self.out(self.style(f"（已按任务名筛选: {task_filter}）", "dim"))
            self.out(
                self.style(
                    "看门狗只在「持续无进展」或「进程异常」告警时才写诊断快照。",
                    "dim",
                )
            )
            return 0

        if getattr(args, "show", False):
            _task, _name, path, _size, _mtime = entries[0]
            self.out(self.style(f"# {path}", "cyan"))
            summary = path / "summary.txt"
            if summary.exists():
                print(summary.read_text(encoding="utf-8", errors="replace").rstrip())
            else:
                self.out("（该快照没有 summary.txt，目录内容:）")
                for child in sorted(path.iterdir()):
                    self.out(f"  {child.name}")
            return 0

        rows = [
            [task, name, fmt_ts(mtime) if mtime else "-", human_size(size)]
            for task, name, _path, size, mtime in entries
        ]
        self.out(render_table(["任务", "快照", "时间", "大小"], rows, ["left", "left", "left", "right"]))
        self.out("")
        self.out(
            self.style(
                f"共 {len(entries)} 份 / {human_size(sum(e[3] for e in entries))}（{root}）",
                "dim",
            )
        )
        self.out(self.style(f"保留上限: {limits_text(self.cfg)}", "dim"))
        self.out(
            self.style(
                "--show 打印最新一份的摘要；进程详情/日志尾部/GPU 快照在该目录里的其它文件",
                "dim",
            )
        )
        return 0

    # ------------------------------------------------------------------ #
    # prune
    # ------------------------------------------------------------------ #
    def cmd_prune(self, args: argparse.Namespace) -> int:
        """清理日志。上限全部来自 config.yaml 的 storage 段。

        默认行为（不用带任何参数）:
          · 每个任务只留最近 N 次运行的日志文件
          · logs/ 总量超过上限就从最旧开始删（优先删成功任务的）
          · events.jsonl / daemon stdout 超过上限则只保留尾部
        正在运行任务的日志绝不会被删。
        """
        tasks, _meta = self._load_tasks_for_read()
        failed = [t.safe_name for t in tasks if t.status == FAILED]
        active = [t.log_file for t in tasks if t.is_running and t.log_file]

        if args.usage:
            overview = usage_overview(self.cfg)
            rows = [[name, human_size(size)] for name, size in overview]
            if args.json:
                print(
                    json.dumps(
                        {
                            "usage": dict(overview),
                            "limits": limits_text(self.cfg),
                            "storage": dict(self.cfg.storage.__dict__),
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                )
                return 0
            self.out(render_table(["项目", "占用"], rows, ["left", "right"]))
            self.out("")
            self.out(self.style(f"生效上限: {limits_text(self.cfg)}", "dim"))
            return 0

        before = dir_usage_bytes(self.cfg.task_log_dir)
        report = prune_all(
            self.cfg,
            self.logger,
            active_log_paths=active,
            failed_task_names=failed,
            dry_run=args.dry_run,
        )
        after = dir_usage_bytes(self.cfg.task_log_dir)

        if args.json:
            print(
                json.dumps(
                    {
                        "dry_run": args.dry_run,
                        "removed_files": report.removed_files,
                        "removed_bytes": report.removed_bytes,
                        "trimmed_bytes": report.trimmed_bytes,
                        "usage_before": before,
                        "usage_after": after,
                        "limits": limits_text(self.cfg),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0

        if report.skipped_active:
            self.out(
                self.style(f"跳过 {report.skipped_active} 组正在运行任务的日志（绝不删）", "dim")
            )
        if not report.changed:
            self.out(self.style("无需清理：当前占用在上限之内。", "green"))
        else:
            self.out(f"logs/ 占用: {human_size(before)} → {human_size(after)}")
            self.out(report.summary())
            for line in report.detail_lines(10):
                self.out(line)
        if args.dry_run:
            self.out(self.style("（--dry-run：没有删除任何文件）", "yellow"))
        self.out(self.style(f"生效上限: {limits_text(self.cfg)}", "dim"))
        return 0


def _mask(text: str) -> str:
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}…{text[-4:]}（已打码）"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    try:
        cli = Cli(args)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "start": cli.cmd_start,
        "daemon": lambda a: _run_daemon(cli, a),
        "stop": cli.cmd_stop,
        "restart": cli.cmd_restart,
        "reload": cli.cmd_reload,
        "status": cli.cmd_status,
        "list": cli.cmd_list,
        "show": cli.cmd_show,
        "add": cli.cmd_add,
        "cancel": cli.cmd_cancel,
        "remove": cli.cmd_remove,
        "requeue": cli.cmd_requeue,
        "logs": cli.cmd_logs,
        "events": cli.cmd_events,
        "config": cli.cmd_config,
        "doctor": cli.cmd_doctor,
        "prune": cli.cmd_prune,
        "watchdog": cli.cmd_watchdog,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse 已限制取值
        parser.print_help()
        return 2
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except TaskError as exc:
        print(f"操作失败: {exc}", file=sys.stderr)
        return 1


def _run_daemon(cli: Cli, args: argparse.Namespace) -> int:
    from .daemon import Daemon

    return Daemon(
        cli.cfg,
        dry_run=getattr(args, "dry_run", False),
        no_launch=getattr(args, "no_launch", False),
        once=getattr(args, "once", False),
        console=True,
    ).run()


__all__ = ["main", "build_parser", "Cli", "PROJECT_ROOT"]
