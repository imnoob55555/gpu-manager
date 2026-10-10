"""训练进程执行器。

安全要点:
  - 只启动自己的进程，用 ``start_new_session=True`` 放入独立进程组，
    停止任务时 killpg 不会波及 daemon 自身。
  - 停止前双重身份校验: (a) 进程 owner 是自己；(b) /proc 中的 starttime 与启动时记录一致。
    任何一条不满足都**拒绝发送 signal**，防止 PID 复用误杀他人进程。
  - 通过 CUDA_VISIBLE_DEVICES 做隔离，不修改任何系统级 CUDA 配置。

可靠性要点:
  - 每个任务生成 run.sh，并把退出码写入 ``*.exit`` 文件。
    这样即使 daemon 在任务运行期间被重启，也能准确判定 completed/failed，
    而不是靠猜（无人值守场景的关键）。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil
import stat as stat_module
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .config import Config
from .task_queue import Task
from .utils import (
    abs_path,
    current_uid,
    ensure_dir,
    fmt_ts,
    gpu_ids_text,
    is_own_process,
    proc_uid,
    process_alive,
    process_start_time,
    resolve_within,
    safe_terminate,
)


class RunnerError(RuntimeError):
    pass


@dataclass
class RunHandle:
    """一次训练运行的句柄（daemon 内存持有；持久化信息记录在 Task 上）。"""

    task_name: str
    pid: int
    pgid: int | None
    log_path: Path
    log_fh: TextIO
    exit_path: Path
    script_path: Path
    started_at: float
    proc_start_time: float | None
    command_line: str
    gpu_ids: list[int] = field(default_factory=list)
    _proc: subprocess.Popen | None = None

    def poll(self) -> int | None:
        if self._proc is None:
            return None
        code = self._proc.poll()
        if code is not None:
            self.flush_close()
        return code

    def read_exit_code(self) -> int | None:
        """从退出码文件读取（run.sh 保证写入）。"""
        try:
            text = self.exit_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return int(text) if text.lstrip("-").isdigit() else None

    def flush_close(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self.log_fh.flush()
        with contextlib.suppress(OSError, ValueError):
            self.log_fh.close()

    def log_size_mb(self) -> float:
        try:
            return self.log_path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def rotate_if_needed(self, max_mb: float) -> bool:
        """超限时把当前日志改名为 .1（保留一份），重开新日志继续写。

        max_mb 允许小数（<=0 表示不限制）。
        """
        if max_mb <= 0 or self.log_size_mb() < max_mb:
            return False
        with contextlib.suppress(OSError, ValueError):
            self.log_fh.flush()
        rotated = self.log_path.with_suffix(self.log_path.suffix + ".1")
        try:
            if rotated.exists():
                rotated.unlink()
            self.log_path.rename(rotated)
            self.log_fh.close()
            # 必须是长寿命句柄（作为子进程 stdout），不能包在 with 中
            self.log_fh = open(  # noqa: SIM115
                self.log_path, "a", encoding="utf-8", buffering=1
            )
            return True
        except OSError:
            return False


class TaskRunner:
    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        # 用「已探测」标志而不是 bool 哨兵: `str | None | bool` 无法被类型系统窄化，
        # 那种写法必须靠 type: ignore 才能过检，容易掩盖真问题。
        self._conda_bin: str | None = None
        self._conda_checked = False
        self._conda_base: str | None = None
        self._conda_base_checked = False
        self._env_cache: dict[str, tuple[float, bool, str]] = {}

    # ------------------------------------------------------------------ #
    # conda 探测与校验
    # ------------------------------------------------------------------ #
    def conda_bin(self) -> str | None:
        if self._conda_checked:
            return self._conda_bin
        candidates: list[str] = []
        if self.cfg.runner.conda_bin:
            candidates.append(os.path.expanduser(self.cfg.runner.conda_bin))
        found = shutil.which("conda")
        if found:
            candidates.append(found)
        home = os.path.expanduser("~")
        candidates += [
            os.path.join(home, "anaconda3", "bin", "conda"),
            os.path.join(home, "miniconda3", "bin", "conda"),
            os.path.join(home, "miniforge3", "bin", "conda"),
            "/opt/conda/bin/conda",
            "/usr/local/anaconda3/bin/conda",
        ]
        for cand in candidates:
            if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
                self._conda_bin = cand
                self._conda_checked = True
                return cand
        self._conda_bin = None
        self._conda_checked = True
        return None

    def conda_base(self) -> str | None:
        if self._conda_base_checked:
            return self._conda_base
        conda = self.conda_bin()
        base: str | None = None
        if conda:
            try:
                proc = subprocess.run(
                    [conda, "info", "--base"], capture_output=True, text=True, timeout=20, check=False
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    base = proc.stdout.strip().splitlines()[-1].strip()
            except (OSError, subprocess.TimeoutExpired):
                base = None
            if not base:
                base = str(Path(conda).parent.parent)
        self._conda_base = base
        self._conda_base_checked = True
        return base

    def conda_sh(self) -> str | None:
        base = self.conda_base()
        if not base:
            return None
        candidate = os.path.join(base, "etc", "profile.d", "conda.sh")
        return candidate if os.path.isfile(candidate) else None

    def list_envs(self) -> list[str]:
        conda = self.conda_bin()
        if not conda:
            return []
        try:
            proc = subprocess.run(
                [conda, "env", "list", "--json"], capture_output=True, text=True, timeout=30, check=False
            )
            data = json.loads(proc.stdout)
            return sorted({str(Path(p).name) for p in data.get("envs", [])})
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, AttributeError):
            return []

    def env_exists(self, env_name: str) -> tuple[bool, str]:
        """校验 conda 环境是否存在（5 分钟缓存）。返回 (存在与否, 说明)。"""
        now = time.time()
        cached = self._env_cache.get(env_name)
        if cached and now - cached[0] < 300:
            return cached[1], cached[2]

        if not self.conda_bin():
            return False, "未找到 conda 可执行文件（可在 runner.conda_bin 指定绝对路径）"
        envs = self.list_envs()
        if not envs:
            return False, "conda env list 无输出，无法校验环境是否存在"
        ok = env_name in envs
        detail = "ok" if ok else f"可用环境: {', '.join(envs[:12])}"
        self._env_cache[env_name] = (now, ok, detail)
        return ok, detail

    # ------------------------------------------------------------------ #
    # 命令与环境
    # ------------------------------------------------------------------ #
    def build_env(self, task: Task, gpu_ids: list[int]) -> dict[str, str]:
        env = dict(os.environ)
        # 1) Manager 默认值
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids_text(gpu_ids) if task.gpu_num > 0 else ""
        env["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env["PYTHONUNBUFFERED"] = "1"
        env["GPU_MANAGER_TASK"] = task.name
        env["GPU_MANAGER_GPU_IDS"] = gpu_ids_text(gpu_ids)
        env["GPU_MANAGER_SERVER"] = self.cfg.server.resolved_name()
        env["GPU_MANAGER_LOG"] = task.log_file
        # 2) 全局额外环境变量
        env.update({str(k): str(v) for k, v in self.cfg.runner.extra_env.items()})
        # 3) 任务级环境变量（优先级最高）
        env.update({str(k): str(v) for k, v in task.env_vars.items()})
        return env

    def build_script(self, task: Task) -> list[str]:
        """生成 run.sh 的脚本行（不含退出码记录部分）。失败抛 RunnerError。"""
        command = task.command.strip()
        env_spec = (task.env or "").strip()

        if not env_spec:
            return [command]

        # 形如 /path/to/env 或 ./venv: 按前缀目录处理
        if "/" in env_spec or env_spec.startswith("."):
            prefix = abs_path(env_spec)
            if not (prefix / "bin").is_dir():
                raise RunnerError(f"env 路径不是有效的 conda/venv 前缀（缺少 bin/）: {prefix}")
            return [
                f"export CONDA_PREFIX={shlex.quote(str(prefix))}",
                f'export PATH={shlex.quote(str(prefix))}/bin:"$PATH"',
                command,
            ]

        conda = self.conda_bin()
        if not conda:
            raise RunnerError(
                f"任务 {task.name} 指定了 conda 环境 {env_spec!r}，但未找到 conda。"
                "请在 config.yaml 中设置 runner.conda_bin"
            )
        conda_sh = self.conda_sh()
        if conda_sh:
            # 与交互式一致的完整 activate（脚本里依赖 CONDA_PREFIX 等变量时才正确）
            return [
                f"source {shlex.quote(conda_sh)} || exit 97",
                f"conda activate {shlex.quote(env_spec)} || exit 98",
                command,
            ]
        self.logger.warning("未找到 conda.sh，任务 %s 退化为 conda run（近似 activate）", task.name)
        return [
            f"exec {shlex.quote(conda)} run --no-capture-output -n {shlex.quote(env_spec)} bash -c "
            f"{shlex.quote(command)}"
        ]

    def describe_command(self, task: Task) -> str:
        """给 `gpu-manager show` 展示的等效 shell 命令。"""
        try:
            lines = self.build_script(task)
        except RunnerError as exc:
            return f"(无法构造命令: {exc})"
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 启动
    # ------------------------------------------------------------------ #
    def _prepare_artifacts(self, task: Task) -> tuple[Path, TextIO, Path, Path]:
        """准备 log / run.sh / exit 码文件。

        任务名会参与拼路径（logs/<任务名>/），所以这里做两层防护：
        task.safe_name 做字符白名单，resolve_within 再做目录包含校验。
        """
        try:
            log_dir = ensure_dir(resolve_within(self.cfg.task_log_dir / task.safe_name, self.cfg.task_log_dir))
        except ValueError as exc:
            raise RunnerError(f"任务 {task.name} 的日志目录非法: {exc}") from exc
        stamp = time.strftime("%Y%m%d-%H%M%S")
        unique = f"{stamp}-{os.getpid()}-{int(time.time() * 1000) % 1000}"

        log_path = log_dir / f"{unique}.log"
        script_path = log_dir / f"{unique}.run.sh"
        exit_path = log_dir / f"{unique}.exit"

        # 长寿命句柄: 作为子进程 stdout，生命周期由 RunHandle.flush_close() 管理
        # pi-lens-ignore: python-path-traversal -- log_dir 已由 resolve_within 限定在 cfg.task_log_dir 内
        log_fh = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

        for latest_name, target in (("latest.log", log_path), ("latest.run.sh", script_path), ("latest.exit", exit_path)):
            latest = log_dir / latest_name
            with contextlib.suppress(OSError):
                latest.unlink(missing_ok=True)
                latest.symlink_to(target.name)
        return log_path, log_fh, script_path, exit_path

    def _write_script(self, script_path: Path, lines: list[str], exit_path: Path, header: str) -> None:
        """生成 run.sh。退出码用 EXIT trap 记录（而不是写在脚本末尾）。

        原因: 用户的命令里很常见 `set -e` 或 `exit N`，这会让脚本提前结束，
        写在末尾的记录语句永远不会执行 → 退出码丢失（daemon 重启后就无法判断成败）。
        trap 在脚本以任何方式终止时都会触发，$? 就是真实退出码。
        路径通过变量传入，避免把带引号的路径嵌进 trap 字符串里。
        """
        trap_body = (
            "gpu_manager_exit=$?; "
            'printf "%s\\n" "$gpu_manager_exit" > "$GPU_MANAGER_EXIT_FILE"'
        )
        body = [
            "#!/usr/bin/env bash",
            "# 由 gpu-manager 自动生成：任务结束时通过 EXIT trap 记录退出码",
            f"# {header}",
            f"GPU_MANAGER_EXIT_FILE={shlex.quote(str(exit_path))}",
            f"trap '{trap_body}' EXIT",
            "",
            *lines,
            "",
        ]
        # pi-lens-ignore: python-path-traversal -- script_path 与 log_path 同目录，已由 resolve_within 校验
        script_path.write_text("\n".join(body), encoding="utf-8")
        os.chmod(script_path, 0o700)

    def launch(self, task: Task, gpu_ids: list[int]) -> RunHandle:
        """启动训练进程。失败抛 RunnerError（调用方负责把任务标记为 failed）。"""
        if task.gpu_num > 0 and len(gpu_ids) != task.gpu_num:
            raise RunnerError(f"GPU 数量不匹配: 需要 {task.gpu_num}，实际 {len(gpu_ids)} 个 ({gpu_ids})")

        script_lines = self.build_script(task)  # 可能抛 RunnerError（先于创建文件）
        log_path, log_fh, script_path, exit_path = self._prepare_artifacts(task)
        task.log_file = str(log_path)

        env = self.build_env(task, gpu_ids)
        header = (
            f"task={task.name} server={self.cfg.server.resolved_name()} "
            f"user={self.cfg.server.resolved_user()} gpus={gpu_ids_text(gpu_ids)}"
        )
        self._write_script(script_path, script_lines, exit_path, header)

        banner = (
            f"{'=' * 78}\n"
            f"[gpu-manager] task      : {task.name}\n"
            f"[gpu-manager] started   : {fmt_ts()}\n"
            f"[gpu-manager] server    : {self.cfg.server.resolved_name()} (user={self.cfg.server.resolved_user()})\n"
            f"[gpu-manager] cwd       : {task.cwd}\n"
            f"[gpu-manager] conda env : {task.env or '(未指定)'}\n"
            f"[gpu-manager] gpus      : {gpu_ids_text(gpu_ids)} (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})\n"
            f"[gpu-manager] command   : {task.command}\n"
            f"[gpu-manager] run script: {script_path}\n"
            f"[gpu-manager] signature : {task.signature[:12]}\n"
            f"{'=' * 78}\n"
        )

        argv = ["bash", str(script_path)]
        try:
            log_fh.write(banner)
            log_fh.flush()
            proc = subprocess.Popen(  # noqa: S603 - argv 为显式构造的脚本路径，无 shell 注入面
                argv,
                cwd=task.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # 独立会话/进程组
                close_fds=True,
            )
        except OSError as exc:
            log_fh.close()
            with contextlib.suppress(OSError):
                exit_path.unlink(missing_ok=True)
            raise RunnerError(f"启动进程失败: {type(exc).__name__}: {exc}") from exc

        start_time = process_start_time(proc.pid)
        if start_time is None:
            self.logger.warning(
                "无法读取 pid=%s 的 starttime，停止该任务时将退化为仅校验 owner", proc.pid
            )

        handle = RunHandle(
            task_name=task.name,
            pid=proc.pid,
            pgid=proc.pid,
            log_path=log_path,
            log_fh=log_fh,
            exit_path=exit_path,
            script_path=script_path,
            started_at=time.time(),
            proc_start_time=start_time,
            command_line=f"bash {script_path}",
            gpu_ids=sorted(int(g) for g in gpu_ids),
            _proc=proc,
        )
        self.logger.info(
            "已启动任务 %s pid=%s pgid=%s GPU=%s cwd=%s log=%s",
            task.name, proc.pid, proc.pid, gpu_ids_text(gpu_ids), task.cwd, log_path,
        )
        return handle

    # ------------------------------------------------------------------ #
    # 停止
    # ------------------------------------------------------------------ #
    def verify_identity(self, task: Task) -> tuple[bool, str]:
        """确认「记录中的 pid」仍是当初那个进程，且属于当前用户。"""
        if not task.pid:
            return False, "任务无 pid 记录"
        if not process_alive(task.pid):
            return False, f"进程 {task.pid} 已不存在"
        if not is_own_process(task.pid):
            return False, f"进程 {task.pid} 的 owner 是 uid{proc_uid(task.pid)}，不是当前用户（拒绝发送 signal）"
        if task.proc_start_time is not None:
            actual = process_start_time(task.pid)
            if actual is None:
                return False, f"无法读取进程 {task.pid} 的 starttime（拒绝发送 signal）"
            if abs(actual - task.proc_start_time) > 0.5:
                return False, (
                    f"进程 {task.pid} 疑似 PID 复用（记录 starttime={task.proc_start_time:.0f}，"
                    f"实际={actual:.0f}），拒绝发送 signal"
                )
        return True, "身份校验通过"

    def send_stop_signal(self, task: Task, *, hard: bool = False) -> str:
        """发送停止信号（不等待）。返回: sent|refused|gone。"""
        ok, detail = self.verify_identity(task)
        if not ok:
            self.logger.warning("停止任务 %s 被拒绝: %s", task.name, detail)
            return "refused"
        target = int(task.pgid or task.pid or 0)
        if target <= 0:
            return "gone"
        signal_name = "SIGKILL" if hard else "SIGTERM"
        try:
            if hard:
                os.killpg(target, 9)
            else:
                os.killpg(target, 15)
        except ProcessLookupError:
            return "gone"
        except PermissionError:
            self.logger.error("停止任务 %s 权限不足（不作任何降级尝试）", task.name)
            return "refused"
        except OSError as exc:
            self.logger.warning("停止任务 %s 失败: %s", task.name, exc)
            return "refused"
        self.logger.info("已向任务 %s 进程组 %s 发送 %s", task.name, target, signal_name)
        return "sent"

    def stop(self, task: Task, reason: str = "用户取消") -> str:
        """同步优雅停止（会阻塞 grace 秒，适合 CLI / 收尾路径）。

        与 send_stop_signal 一样**先做身份校验**：
        绝不向「非本人进程」或「PID 已被复用」的进程发送任何信号。
        这个校验曾经在本方法里被遗漏过（send_stop_signal 有、stop 没有），
        由 tests/test_runner_safety.py::test_stop_refuses_when_pid_reused 锁定。
        """
        target = int(task.pgid or task.pid or 0)
        if target <= 0 or not task.pid or not process_alive(task.pid):
            return "gone"
        ok, detail = self.verify_identity(task)
        if not ok:
            self.logger.warning("停止任务 %s 被拒绝: %s", task.name, detail)
            return "refused"
        self.logger.info(
            "正在停止任务 %s (pid=%s pgid=%s, 原因: %s)，grace=%ss",
            task.name, task.pid, target, reason, self.cfg.runner.term_grace_seconds,
        )
        result = safe_terminate(
            target,
            grace_seconds=self.cfg.runner.term_grace_seconds,
            kill_after_grace=self.cfg.runner.kill_after_grace,
        )
        self.logger.info("任务 %s 停止结果: %s", task.name, result)
        return result

    # ------------------------------------------------------------------ #
    def tail_log(self, task: Task, lines: int = 40) -> list[str]:
        """读取任务日志末尾几行。仅允许读任务日志目录内的文件（防任意文件读取）。"""
        if not task.log_file:
            return []
        try:
            path = resolve_within(task.log_file, self.cfg.task_log_dir)
        except ValueError as exc:
            self.logger.warning("拒绝读取任务日志目录之外的路径: %s", exc)
            return []
        if not path.exists():
            return []
        try:
            # pi-lens-ignore: python-path-traversal -- path 已由 resolve_within 限定在 task_log_dir 内
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read().splitlines()[-max(1, lines):]
        except OSError:
            return []

    @staticmethod
    def tail_path(path: str | os.PathLike[str], lines: int = 200) -> list[str]:
        """有界读取**用户自己在任务定义里声明的**日志文件尾部。

        与 tail_log 的区别: tail_log 读的是我们自己捕获的日志（限定在 task_log_dir 内）；
        这里读的是训练脚本自己重定向的输出（如 console.log），路径由用户在
        ``tasks.yaml`` 里通过 ``eta_logs`` 声明 —— 与 ``command`` 同一信任级别，
        因此不做目录包含校验，但仍然：
          · 只接受普通文件（拒绝目录/FIFO/设备）
          · **只读尾部若干行**（训练日志可能几百 MB，不能整个读进内存）
        """
        target = abs_path(path)
        try:
            info = target.stat()
        except OSError:
            return []
        if not stat_module.S_ISREG(info.st_mode):
            return []
        max_bytes = max(4096, int(lines) * 200)
        try:
            # pi-lens-ignore: python-path-traversal -- 路径由用户在 tasks.yaml 声明（与 command 同信任级别）
            with open(target, encoding="utf-8", errors="replace") as fh:
                offset = max(0, info.st_size - max_bytes)
                fh.seek(offset)
                if offset > 0:
                    # 只有从**文件中间**开始读时，首行才可能被截断 → 丢弃。
                    # 文件本身小于尾窗时不能丢首行，否则会把唯一一行也丢掉
                    # （小日志与 stage.status 这种几十字节的文件都靠这个）。
                    fh.readline()
                data = fh.read()
        except OSError:
            return []
        return data.splitlines()[-max(1, lines):]

    def owner_text(self, pid: int) -> str:
        uid = proc_uid(pid)
        if uid is None:
            return "unknown"
        return f"uid{uid}{'(self)' if uid == current_uid() else ''}"


__all__ = ["TaskRunner", "RunHandle", "RunnerError"]
