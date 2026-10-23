"""CLI 参数解析 + 后台启动/停止端到端测试。

存在的原因（真实 bug 回归）:
    ``--config/--json/--no-color`` 曾经只定义在主命令上，于是
    ``gpu-manager start --config X``、``status --json`` 等**文档化用法**
    会直接以 argparse 退出码 2 失败；更严重的是 ``gpu-manager start``
    在后台 spawn 时恰好把 ``--config`` 拼在子命令之后，导致**主入口完全不可用**。
    而当时只测了内部 ``daemon`` 子命令，所以没被发现。

    下面的用例把三件事锁死:
      1. 全部文档化命令都能被解析（不会 unrecognized arguments）
      2. 全局选项写在子命令前后都生效，且不会被 SUPPRESS 默认值覆盖
      3. 真实的 ``start``（后台 spawn）能起来，并能被 ``stop`` 干净收掉
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys

import pytest

from gpu_manager.cli import PROJECT_ROOT, Cli, build_daemon_argv, build_parser
from gpu_manager.config import load_config
from gpu_manager.state_store import StateStore
from gpu_manager.utils import process_alive, wait_until

# README 里出现的全部调用形式（含全局选项放在子命令之后的写法）
DOCUMENTED_INVOCATIONS: list[list[str]] = [
    ["daemon", "--once", "--dry-run", "--no-launch"],
    ["start"],
    ["start", "--foreground"],
    ["start", "--dry-run", "--no-launch"],
    ["start", "--config", "config.yaml"],
    ["stop"],
    ["stop", "--force"],
    ["restart"],
    ["reload"],
    ["status"],
    ["status", "--watch"],
    ["status", "--watch", "3"],
    ["status", "--json"],
    ["status", "--stale"],
    ["status", "--config", "config.yaml"],
    ["status", "--json", "--no-color"],
    ["list"],
    ["list", "--json"],
    ["list", "--status", "running"],
    ["list", "--pending"],
    ["list", "--no-color"],
    ["show", "exp1"],
    ["show", "exp1", "--json"],
    ["add", "--name", "n", "--cmd", "true"],
    ["add", "--dry-run", "--name", "n", "--cmd", "true", "--gpu-num", "2"],
    [
        "add", "--name", "exp1", "--cmd", "python main.py",
        "--cwd", "/data/project",   # 仅作解析测试，不会真去访问该路径
        "--env", "dino", "--gpu-num", "2", "--priority", "10", "--json",
    ],
    ["cancel", "exp1"],
    ["remove", "exp1"],
    ["requeue", "exp1"],
    ["logs"],
    ["logs", "exp1"],
    ["logs", "exp1", "-f", "-n", "5"],
    ["events"],
    ["events", "-n", "5"],
    ["events", "--json"],
    ["config"],
    ["config", "--json"],
    ["config", "--raw"],
    ["doctor"],
    ["doctor", "--notify-test"],
    ["doctor", "--json"],
    ["prune"],
    ["prune", "--dry-run"],
    ["prune", "--usage"],
    ["prune", "--usage", "--json"],
    ["watchdog"],
    ["watchdog", "status"],
    ["watchdog", "status", "--json"],
    ["watchdog", "diagnostics"],
    ["watchdog", "diagnostics", "task_x", "--show"],
    ["watchdog", "diagnostics", "--json"],
    # 全局选项写在子命令之前
    ["--config", "config.yaml", "status"],
    ["--json", "list"],
    ["--no-color", "status"],
]


def parse_ok(argv: list[str]) -> argparse.Namespace:
    """用真实 parser 解析；argparse 出错会抛 SystemExit(2)。"""
    try:
        return build_parser().parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - 失败路径
        pytest.fail(f"gpu-manager {' '.join(argv)} 解析失败（argparse 退出码 {exc.code}）")


@pytest.mark.parametrize("argv", DOCUMENTED_INVOCATIONS, ids=lambda a: " ".join(a))
def test_documented_invocations_parse(argv):
    namespace = parse_ok(argv)
    assert namespace.command == argv[0] or argv[0].startswith("--")


def test_global_flags_work_after_subcommand():
    assert parse_ok(["status", "--json"]).json is True
    assert parse_ok(["list", "--no-color"]).no_color is True
    assert parse_ok(["status", "--config", "x.yaml"]).config == "x.yaml"


def test_global_flags_work_before_subcommand():
    assert parse_ok(["--json", "list"]).json is True
    assert parse_ok(["--config", "x.yaml", "status"]).config == "x.yaml"


def test_global_defaults_are_not_clobbered_by_subparser():
    """子命令上的全局选项必须用 SUPPRESS，否则子解析器的默认值会覆盖主解析器的值。"""
    assert parse_ok(["list"]).json is False
    assert parse_ok(["list"]).config is None
    assert parse_ok(["list"]).no_color is False
    # 主命令给了值、子命令没给 → 必须保留主命令的值
    assert parse_ok(["--json", "list"]).json is True
    assert parse_ok(["--config", "a.yaml", "list"]).config == "a.yaml"


def test_subcommand_specific_flags_still_parse():
    ns = parse_ok(["logs", "exp1", "-n", "5", "-f"])
    assert ns.name == "exp1" and ns.lines == 5 and ns.follow is True
    ns = parse_ok(["add", "--name", "n", "--cmd", "c", "--env-var", "A=1", "--env-var", "B=2"])
    assert ns.env_var == ["A=1", "B=2"]


def test_invalid_arguments_still_fail():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["list", "--nonexistent-option"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args(["add", "--cmd", "true"])  # 缺 --name


def test_missing_command_shows_help_without_crashing():
    ns = build_parser().parse_args([])
    assert ns.command is None


# --------------------------------------------------------------------------- #
# 后台 spawn 的 argv（这是曾经出错的地方）
# --------------------------------------------------------------------------- #
def test_daemon_argv_places_config_before_subcommand(tmp_path):
    argv = build_daemon_argv(tmp_path / "config.yaml", dry_run=True, no_launch=True)
    assert argv[:3] == [sys.executable, "-m", "gpu_manager"]
    assert argv.index("--config") < argv.index("daemon"), "全局选项必须在子命令之前"

    # 用**真实 parser** 解析去掉 `python -m gpu_manager` 前缀后的参数，确保这个 argv 真的可用
    namespace = parse_ok(argv[3:])
    assert namespace.command == "daemon"
    assert namespace.config.endswith("config.yaml")
    assert namespace.dry_run is True
    assert namespace.no_launch is True
    assert namespace.once is False


def test_daemon_argv_minimal(tmp_path):
    argv = build_daemon_argv(tmp_path / "config.yaml")
    # 前 3 项是解释器与模块前缀，之后才是真正传给 CLI 的参数
    assert argv[1:3] == ["-m", "gpu_manager"]
    assert argv[3:] == ["--config", str(tmp_path / "config.yaml"), "daemon"]
    assert "--dry-run" not in argv and "--no-launch" not in argv


# --------------------------------------------------------------------------- #
# 端到端: 真实后台启动 + 查询 + 停止
# --------------------------------------------------------------------------- #
def _run_cli(cfg_path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "gpu_manager", "--config", str(cfg_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )


def test_start_status_stop_background_end_to_end(cfg_path):
    """覆盖用户的主入口 `gpu-manager start`（后台 spawn）与 `stop`。

    这个用例专治「后台 argv 拼装错误」：它真的会 fork 一个 daemon 子进程出来。
    """
    cfg = load_config(cfg_path)
    daemon_pid: int | None = None
    try:
        started = _run_cli(cfg_path, "start", "--wait", "30")
        assert started.returncode == 0, f"start 失败:\n{started.stdout}\n{started.stderr}"
        assert "已启动" in started.stdout

        daemon_pid = StateStore.read_pid(cfg.pid_file)
        assert daemon_pid, "start 成功后必须写出 PID 文件"
        assert process_alive(daemon_pid), "daemon 进程应当存活"

        # 重复启动应被识别为「已在运行」，而不是再造一个
        again = _run_cli(cfg_path, "start")
        assert "已在运行" in again.stdout

        # 状态查询（走 socket）应能连上，并报告同一个 pid
        status = _run_cli(cfg_path, "status", "--json")
        assert status.returncode == 0, status.stderr
        payload = json.loads(status.stdout)
        assert payload["daemon"]["pid"] == daemon_pid
        assert payload["mode"] in ("IDLE", "QUEUE", "RUNNING")

        stopped = _run_cli(cfg_path, "stop", "--timeout", "25")
        assert stopped.returncode == 0, f"stop 失败:\n{stopped.stdout}\n{stopped.stderr}"
        assert wait_until(lambda: not process_alive(daemon_pid), 15.0, interval=0.1), "stop 后 daemon 应退出"
        assert StateStore.read_pid(cfg.pid_file) is None, "退出后应清理 PID 文件"
    finally:
        # 无论断言是否失败，都不能留下野 daemon
        if daemon_pid and process_alive(daemon_pid):
            _run_cli(cfg_path, "stop", "--force", "--timeout", "10")
        if daemon_pid and process_alive(daemon_pid):
            with pytest.raises(OSError):
                os.kill(daemon_pid, signal.SIGKILL)


def test_start_foreground_once_exits(cfg_path):
    """前台单次 tick（调试用法）应正常退出，不留下 PID 文件。"""
    result = _run_cli(cfg_path, "daemon", "--once", "--dry-run", "--no-launch")
    assert result.returncode == 0, result.stderr
    cfg = load_config(cfg_path)
    assert StateStore.read_pid(cfg.pid_file) is None
    assert cfg.state_file.exists(), "单次 tick 也应持久化状态"


# --------------------------------------------------------------------------- #
# status 输出内容
# --------------------------------------------------------------------------- #
def test_status_json_includes_interactive_block(cfg_path, capsys):
    """回归: CLI 曾只挑选部分字段输出，把 daemon 已提供的 interactive / watchdog 丢掉了。"""
    cfg = load_config(cfg_path)
    store = StateStore(cfg.state_file, logging.getLogger("test"))
    state = store.read()
    state["watchdog"] = {
        "enabled": True,
        "active": False,
        "summary": "每 600s 检查 / 3600s 无进展告警",
        "tasks": {},
    }
    store.write(state)

    args = argparse.Namespace(
        config=str(cfg_path), json=True, no_color=True, command="status", stale=False
    )
    cli = Cli(args)

    code = cli._print_status(args)  # noqa: SLF001 - 直接验证渲染结果

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "interactive" in payload
    assert payload["interactive"]["enabled"] is False
    assert payload["interactive"]["running"] is False
    # 其他关键字段不能被顺手删掉
    for key in ("daemon", "mode", "counts", "tasks", "gpus", "gpu_policy", "watchdog"):
        assert key in payload
    # 看门狗段要能从状态里读到策略（skill / 自动化靠它判断）
    assert payload["watchdog"].get("enabled") is True
    assert "检查" in str(payload["watchdog"].get("summary", ""))


# --------------------------------------------------------------------------- #
# watchdog
# --------------------------------------------------------------------------- #
def _cli(cfg_path, **overrides) -> Cli:
    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog", **overrides
    )
    return Cli(args)


def test_watchdog_status_without_daemon_points_at_restart(cfg_path, capsys):
    """daemon 未运行（或运行中的是升级前旧版本）时不能报错，要给出可操作提示。"""
    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog", action="status"
    )
    code = Cli(args).cmd_watchdog(args)  # noqa: SLF001 - 直接验证渲染结果
    out = capsys.readouterr().out
    assert code == 0
    assert "看门狗" in out
    assert "watchdog" in out or "重启" in out


def test_watchdog_status_renders_persisted_state(cfg_path, capsys):
    """daemon 离线时读 state.json 里的 watchdog 段（静态展示也要有信息）。"""
    cfg = load_config(cfg_path)
    store = StateStore(cfg.state_file, __import__("logging").getLogger("test"))
    state = store.read()
    state["watchdog"] = {
        "enabled": True,
        "active": True,
        "summary": "每 600s 检查 / 3600s 无进展告警 / 每 3600s 重复",
        "startup_grace_seconds": 900,
        "anomaly_confirm_checks": 2,
        "diagnostics_dir": str(cfg.diagnostics_dir),
        "last_check_at": 1_700_000_000.0,
        "check_count": 7,
        "alert_total": 1,
        "watched": 1,
        "tasks": {
            "exp1": {
                "pid": 4242,
                "proc_state_text": "S（可中断睡眠）",
                "last_activity": 1_700_000_000.0,
                "last_activity_source": "捕获日志",
                "idle_seconds": 120.0,
                "alert_kind": "stall",
                "alert_reason": "已 1h 无任何进展",
                "alert_count": 1,
                "last_alert_at": 1_699_999_000.0,
                "signals": {},
                "last_snapshot": str(cfg.diagnostics_dir / "exp1" / "20260102-120000"),
            }
        },
    }
    store.write(state)

    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog", action="status"
    )
    code = Cli(args).cmd_watchdog(args)  # noqa: SLF001
    out = capsys.readouterr().out
    assert code == 0
    assert "exp1" in out
    assert "捕获日志" in out
    assert "已 1h 无任何进展" in out
    assert "第 1 次" in out


def test_watchdog_diagnostics_empty_and_listed(cfg_path, capsys):
    cfg = load_config(cfg_path)
    cli = _cli(cfg_path)

    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog",
        action="diagnostics", task=None, show=False,
    )
    assert cli.cmd_watchdog(args) == 0  # noqa: SLF001
    assert "没有诊断快照" in capsys.readouterr().out

    # 造两份快照
    for name in ("20260101-120000", "20260102-120000"):
        snap = cfg.diagnostics_dir / "exp1" / name
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "summary.txt").write_text("GPU Manager 看门狗诊断快照\n任务: exp1\n", encoding="utf-8")

    args = argparse.Namespace(
        config=str(cfg_path), json=True, no_color=True, command="watchdog",
        action="diagnostics", task=None, show=False,
    )
    assert cli.cmd_watchdog(args) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert {item["snapshot"] for item in payload["snapshots"]} == {
        "20260101-120000",
        "20260102-120000",
    }
    assert "诊断快照" in payload["limits"] or "每任务" in payload["limits"]

    # --show 打印最新一份的 summary.txt
    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog",
        action="diagnostics", task="exp1", show=True,
    )
    assert cli.cmd_watchdog(args) == 0  # noqa: SLF001
    assert "20260102-120000" in capsys.readouterr().out

    # 任务名筛选不匹配时也不报错
    args = argparse.Namespace(
        config=str(cfg_path), json=False, no_color=True, command="watchdog",
        action="diagnostics", task="nope", show=False,
    )
    assert cli.cmd_watchdog(args) == 0  # noqa: SLF001
    assert "没有诊断快照" in capsys.readouterr().out
