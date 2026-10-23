"""配置加载与校验。

所有阈值必须配置化，此模块是唯一读取 YAML 的入口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .utils import abs_path, parse_gpu_list


class ConfigError(ValueError):
    """配置错误 —— 启动时快速失败，绝不带着错误阈值跑调度。"""


# --------------------------------------------------------------------------- #
# 校验小工具
# --------------------------------------------------------------------------- #
def _require_mapping(raw: Any, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} 必须是映射（dict），实际是 {type(raw).__name__}")
    return raw


def _get_int(raw: dict[str, Any], key: str, default: int, where: str, *, minimum: int | None = None) -> int:
    if key not in raw or raw[key] is None:
        value = default
    else:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ConfigError(f"{where}.{key} 必须是整数，实际是 {value!r}")
        try:
            value = int(str(value).strip())
        except ValueError as exc:
            raise ConfigError(f"{where}.{key} 必须是整数，实际是 {value!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}.{key} 必须 >= {minimum}，实际是 {value}")
    return value


def _get_float(raw: dict[str, Any], key: str, default: float, where: str, *, minimum: float | None = None) -> float:
    if key not in raw or raw[key] is None:
        value = default
    else:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"{where}.{key} 必须是数字，实际是 {value!r}")
        try:
            value = float(str(value).strip())
        except ValueError as exc:
            raise ConfigError(f"{where}.{key} 必须是数字，实际是 {value!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}.{key} 必须 >= {minimum}，实际是 {value}")
    return value


def _get_bool(raw: dict[str, Any], key: str, default: bool, where: str) -> bool:
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1", "on"):
            return True
        if low in ("false", "no", "0", "off"):
            return False
    raise ConfigError(f"{where}.{key} 必须是布尔值，实际是 {value!r}")


def _get_str(raw: dict[str, Any], key: str, default: str, where: str) -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(f"{where}.{key} 必须是字符串，实际是 {type(value).__name__}")
    return value


# --------------------------------------------------------------------------- #
# 各配置段
# --------------------------------------------------------------------------- #
@dataclass
class ServerConfig:
    """服务器身份 —— 只与「通知内容/日志归属」有关，绝不用于 sudo 或系统级改动。"""

    name: str = ""
    user: str = ""

    def resolved_name(self) -> str:
        return self.name or os.uname().nodename

    def resolved_user(self) -> str:
        return self.user or os.environ.get("USER", "unknown")


@dataclass
class GpuConfig:
    """GPU 监控配置。

    - max_idle_utilization: 空闲判定利用率阈值（%）
    - max_idle_memory_mb:   空闲判定已用显存阈值（MiB）
    - require_no_compute_process: 必须没有 compute 进程才算空闲
    - check_interval_seconds: 每轮检查间隔
    - idle_confirm_checks: 连续多少次满足才算「确认空闲」
    - monitor_gpus: 参与观察（空闲提醒）的 GPU；"all" 或列表
    - scheduler_allowed_gpus: 允许自动调度占用的 GPU；"all" 或列表
    - reserved_gpus: 永远不参与自动调度（即使出现在 scheduler_allowed_gpus 中也无效）
    """

    max_idle_utilization: int = 5
    max_idle_memory_mb: int = 1000
    require_no_compute_process: bool = True
    check_interval_seconds: int = 30
    idle_confirm_checks: int = 3
    ignore_graphics_processes: bool = True
    monitor_gpus: str | list[Any] = "all"
    scheduler_allowed_gpus: str | list[Any] = "all"
    reserved_gpus: list[Any] = field(default_factory=list)
    #: 视为「自己进程」的用户名白名单，用于 compute 进程归属判断；
    #: 留空则默认只看当前用户（当前用户以外的任何 compute 进程都算被占用）
    own_users: list[str] = field(default_factory=list)

    # ---- 解析结果（运行时填充） ----
    monitor_gpu_ids: list[int] = field(default_factory=list)
    allowed_gpu_ids: list[int] = field(default_factory=list)
    reserved_gpu_ids: list[int] = field(default_factory=list)

    def resolve(self) -> None:
        if self.idle_confirm_checks < 1:
            raise ConfigError("gpu.idle_confirm_checks 必须 >= 1")
        if self.check_interval_seconds < 1:
            raise ConfigError("gpu.check_interval_seconds 必须 >= 1")
        if not (0 <= self.max_idle_utilization <= 100):
            raise ConfigError("gpu.max_idle_utilization 必须在 0..100 之间")
        self.monitor_gpu_ids = parse_gpu_list(self.monitor_gpus)
        self.allowed_gpu_ids = parse_gpu_list(self.scheduler_allowed_gpus)
        self.reserved_gpu_ids = parse_gpu_list(self.reserved_gpus)

    def monitor_set(self, all_gpu_ids: list[int]) -> list[int]:
        """monitor_gpus=all 时随实际卡数展开。"""
        if not self.monitor_gpu_ids:
            return sorted(all_gpu_ids)
        return sorted(g for g in self.monitor_gpu_ids if g in set(all_gpu_ids))

    def schedulable_set(self, all_gpu_ids: list[int]) -> list[int]:
        """可用于自动调度的 GPU = scheduler_allowed_gpus - reserved_gpus，且须真实存在。"""
        pool = self.allowed_gpu_ids if self.allowed_gpu_ids else list(all_gpu_ids)
        reserved = set(self.reserved_gpu_ids)
        existing = set(all_gpu_ids)
        return sorted(g for g in pool if g in existing and g not in reserved)

    def owns_username(self, username: str) -> bool:
        if not username:
            return False
        users = set(self.own_users) if self.own_users else set()
        return username in users


@dataclass
class SchedulerConfig:
    max_concurrent_tasks: int = 0
    strategy: str = "priority_first_fit"
    launch_second_check: bool = True
    grace_before_reuse_seconds: int = 20
    allow_cpu_tasks: bool = False
    waiting_notify_interval_seconds: int = 3600
    #: 队列长时间无法满足时的提醒，避免用户以为程序挂了
    queue_notify: bool = True

    def validate(self) -> None:
        if self.strategy not in ("priority_first_fit",):
            raise ConfigError(
                f"scheduler.strategy 目前仅支持 priority_first_fit，实际是 {self.strategy!r}"
            )
        if self.max_concurrent_tasks < 0:
            raise ConfigError("scheduler.max_concurrent_tasks 必须 >= 0（0 表示不限制）")


@dataclass
class NotifyConfig:
    on_gpu_idle: bool = True
    on_task_started: bool = True
    on_task_finished: bool = True
    on_task_failed: bool = True
    on_task_queued: bool = True
    on_error: bool = True
    idle_cooldown_seconds: int = 1800
    min_idle_gpus_to_notify: int = 1
    failure_cooldown_seconds: int = 0
    mention_all_on_idle: bool = False
    mention_all_on_failure: bool = True


@dataclass
class QueueConfig:
    """任务列表的**展示上限**与终态任务**记录保留**。

    两个概念要分清:
      · ``max_*_in_reply`` —— 只影响飞书「队列」回复的长度（交互查询仍然只读）
      · ``finished_retention_hours`` —— 超时的终态（completed/failed/cancelled）任务
        记录由 daemon **真正删除**（state.json 里不再有它，`list`/`show` 也查不到）；
        日志文件另有 ``storage`` 策略单独保留。0 = 永久保留记录。
    """

    #: 「队列」回复最多列多少个未完成任务（running 在前，pending 按优先级）
    max_unfinished_in_reply: int = 10
    #: 「队列」回复最多列多少个已结束任务（按结束时间倒序；cancelled 也计入，0 = 不列）
    max_finished_in_reply: int = 5
    #: 已结束任务记录的保留小时数；超时由 daemon 自动删除（0 = 永久保留）
    finished_retention_hours: int = 72

    def validate(self) -> None:
        if self.max_unfinished_in_reply < 1:
            raise ConfigError(
                "queue.max_unfinished_in_reply 必须 >= 1，"
                f"实际是 {self.max_unfinished_in_reply}"
            )
        if self.max_finished_in_reply < 0:
            raise ConfigError(
                "queue.max_finished_in_reply 必须 >= 0，"
                f"实际是 {self.max_finished_in_reply}"
            )
        if self.finished_retention_hours < 0:
            raise ConfigError(
                "queue.finished_retention_hours 必须 >= 0，"
                f"实际是 {self.finished_retention_hours}"
            )


@dataclass
class FeishuConfig:
    """飞书集成（**唯一**一段）：出站通知与入站交互查询共用同一个自建应用。

    为什么不再有 webhook（群机器人）：自定义群机器人只能「发」不能「收」，
    而自建应用 + 长连接既能发通知、又能在飞书里问答（且无需公网地址）。
    两件事共用一份 app 凭证，但开关**解耦**:
      · ``enabled``             —— 出站通知（空闲提醒 / 任务起止 / 看门狗告警）
      · ``interactive_enabled`` —— 入站查询（长连接监听，只读）

    缺少凭证不算致命错误：通知是旁路能力，绝不能因为没配好通知就让训练无法调度。
    这类情况由 credential_warnings() 提醒，并自动降级为「只写日志」。
    """

    # --- 自建应用凭证（通知与交互查询共用；开放平台 → 自建应用 → 凭证）---
    app_id: str = ""
    app_secret: str = ""

    # --- 出站通知 ---
    enabled: bool = True
    card: bool = True
    mention_all: bool = False
    receive_id_type: str = "open_id"
    #: 通知收件人；留空 = 自动用 allowed_users 的第一个（即与机器人的私聊）
    receive_id: str = ""
    timeout_seconds: int = 10
    verify_ssl: bool = True
    #: 出站 URL 主机白名单（防 SSRF：配置被误改/篡改时不至于把请求发到任意地址）
    allowed_hosts: list[str] = field(
        default_factory=lambda: ["open.feishu.cn", "open.larksuite.com"]
    )
    #: 显式允许任意主机（使用自建网关/代理时开启）
    allow_any_host: bool = False

    # --- 入站交互查询（自建应用 + 长连接；**只读**，不含任何写操作）---
    #: 长连接监听开关；与上面的 enabled（出站通知）解耦
    interactive_enabled: bool = False
    #: 允许查询的 open_id 白名单；空 = 拒绝所有人（fail-closed）
    allowed_users: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 10
    log_lines_default: int = 20

    # --- ETA（预计结束时间）---
    eta_enabled: bool = True
    eta_sample_interval_seconds: int = 60
    eta_history_fallback: bool = True
    eta_history_max_runs: int = 20
    #: 自定义进度正则；空 = 用 eta.DEFAULT_PATTERNS
    eta_patterns: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def resolved_receive_id(self) -> str:
        """通知收件人：显式 receive_id 优先，否则取白名单第一个人（与机器人私聊）。

        白名单里存的是 open_id，所以**只当 receive_id_type=open_id 时才用它兜底**；
        否则（例如想发到群的 chat_id）必须显式填 receive_id，避免把 ID 类型搞混。
        """
        explicit = (self.receive_id or "").strip()
        if explicit:
            return explicit
        if self.receive_id_type != "open_id":
            return ""
        return self.allowed_users[0].strip() if self.allowed_users else ""

    def validate(self) -> None:
        """只拦截真正非法的配置值。

        缺少凭证**不算致命错误**：通知是旁路能力，绝不能因为没配好通知就让长时间
        训练无法被调度。这类情况由 credential_warnings() 提醒，并降级为「只写日志」。
        """
        if self.receive_id_type not in ("open_id", "user_id", "chat_id", "email"):
            raise ConfigError(
                "feishu.receive_id_type 必须是 open_id / user_id / chat_id / email，"
                f"实际是 {self.receive_id_type!r}"
            )

    def notification_warnings(self) -> list[str]:
        """只与「出站通知」相关的降级原因（空 = 通知可用）。"""
        if not self.enabled:
            return []
        out: list[str] = []
        if not (self.app_id and self.app_secret):
            out.append("feishu.app_id/app_secret 未配置 → 通知只写日志，不会发到飞书")
        elif not self.resolved_receive_id():
            out.append(
                "feishu.receive_id 未配置 → 通知没有目标，只写日志"
                "（留空时默认用 allowed_users 的第一个 open_id；"
                "若 receive_id_type 不是 open_id 则必须显式填 receive_id）"
            )
        return out

    def interactive_warnings(self) -> list[str]:
        """只与「入站查询」相关的降级原因（空 = 查询可用）。"""
        if not self.interactive_enabled:
            return []
        if not (self.app_id and self.app_secret):
            return [
                "feishu.interactive_enabled=true 但 app_id/app_secret 未填 → 交互查询不会启动"
            ]
        if not self.allowed_users:
            return [
                "feishu.allowed_users 为空 → 按 fail-closed 策略将拒绝所有查询"
                "（先给机器人发一条消息，从 daemon 日志里取你的 open_id 填进去）"
            ]
        return []

    def credential_warnings(self) -> list[str]:
        """返回导致降级的原因（出站通知 + 入站查询；空列表 = 配置齐备）。

        仅供启动日志使用；构造通知器请用 notification_warnings()，
        否则「查询没配好」会连累通知降级。
        """
        return self.interactive_warnings() + self.notification_warnings()


@dataclass
class RunnerConfig:
    conda_bin: str = ""
    term_grace_seconds: int = 20
    kill_after_grace: bool = True
    max_log_mb: int = 64
    conda_default_env: str = ""
    #: 附加到每个训练进程的环境变量
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class PathsConfig:
    state_file: str = "data/state.json"
    log_file: str = "data/manager.log"
    event_log: str = "data/events.jsonl"
    task_log_dir: str = "logs"
    #: 看门狗诊断快照目录（单独放，避免干扰 logs/ 的"一次运行产物"整组清理）
    diagnostics_dir: str = "data/diagnostics"
    pid_file: str = "data/manager.pid"
    socket_file: str = "data/manager.sock"
    queue_file: str = "tasks.yaml"


@dataclass
class StorageConfig:
    """日志保留与磁盘占用上限（0 = 不限制）。

    有两级约束:
      · 每个任务最多保留最近 ``task_log_keep_files_per_task`` 次运行的日志
      · 整个 task_log_dir 总量不超过 ``task_log_dir_max_mb``，超出按最旧优先删
    另有三个天生不会轮转的文件（events / daemon stdout）按大小裁剪。
    """

    task_log_keep_files_per_task: int = 10
    task_log_dir_max_mb: int = 200
    prefer_keep_failed_logs: bool = True
    daemon_log_max_mb: int = 5
    daemon_log_backups: int = 2
    events_max_mb: int = 5
    daemon_out_max_mb: int = 5
    prune_interval_seconds: int = 300


@dataclass
class WatchdogConfig:
    """低频率看门狗：只在有训练任务时工作，持续无进展则告警 + 保留诊断。

    三类信号（任一有动静就算「有进展」）:
      · 进度记录: ``task.progress_updated_at``（eta 采样写入的新样本时间）
      · 日志时间: 捕获日志 / ``eta_logs`` / 进度来源文件的 mtime
      · 进程状态: ``/proc/<pid>`` 的存活、身份（starttime 防 PID 复用）与状态字符

    语义:
      · 无运行中任务 → 看门狗自动关闭（不检查、不占资源、不告警）
      · 连续 ``stall_threshold_seconds`` 无任何信号更新 → 告警（每隔
        ``repeat_alert_interval_seconds`` 重复提醒，恢复时可选通知）
      · 进程异常（消失 / 僵尸 Z / 停止 T / 不可中断 D / PID 复用）→ 连续
        ``anomaly_confirm_checks`` 轮确认后**立即**告警，不等停滞阈值

    安全: 全程只读 ``/proc``，绝不对任何进程发信号（与其他模块同一硬边界）。
    """

    enabled: bool = True
    #: 检查周期（低频；必须 <= stall_threshold_seconds）
    check_interval_seconds: int = 600
    #: 连续多久无任何进展算「停滞」
    stall_threshold_seconds: int = 3600
    #: 任务启动后的宽限期（编译/数据加载/首次 eval 等，期间不判停滞）
    startup_grace_seconds: int = 900
    #: 停滞期间重复提醒间隔（0 = 不重复提醒）
    repeat_alert_interval_seconds: int = 3600
    #: 恢复（有进展或进程恢复正常）时是否发一条通知
    notify_on_recovery: bool = True
    #: 进程异常时是否立即告警（False = 也走无进展阈值）
    instant_process_anomaly: bool = True
    #: 进程异常需要连续多少轮确认才告警（1 = 当轮即告警）
    anomaly_confirm_checks: int = 2
    #: 诊断快照里每个日志来源保留的尾部行数
    log_tail_lines: int = 200

    # --- 诊断快照保留 ---
    diagnostics_enabled: bool = True
    #: 每个任务最多保留多少份快照（0 = 不限制）
    diagnostics_keep_per_task: int = 5
    #: diagnostics 目录总量上限 MB（0 = 不限制）
    diagnostics_max_mb: int = 50
    #: 快照里是否包含该任务 GPU 的状态与占用进程
    include_gpu_snapshot: bool = True

    def validate(self) -> None:
        """只拦「说不通」的组合；不设拍脑袋的下限（阈值高低是使用者的选择）。"""
        if self.stall_threshold_seconds < self.check_interval_seconds:
            raise ConfigError(
                "watchdog.stall_threshold_seconds "
                f"({self.stall_threshold_seconds}) 小于 check_interval_seconds "
                f"({self.check_interval_seconds})——检查周期比判定阈值还长时阈值失去意义"
                "（每轮检查都必然已超阈值）"
            )

    def summary(self) -> str:
        """一行摘要，用于启动日志与 ``watchdog status``。"""
        if not self.enabled:
            return "已禁用"
        repeat = (
            f"每 {self.repeat_alert_interval_seconds}s 重复"
            if self.repeat_alert_interval_seconds
            else "不重复"
        )
        return (
            f"每 {self.check_interval_seconds}s 检查 / "
            f"{self.stall_threshold_seconds}s 无进展告警 / {repeat}"
        )


@dataclass
class TaskDefaults:
    """tasks.yaml 中每个任务未显式给出的字段默认值。"""

    gpu_num: int = 1
    priority: int = 0
    env: str = ""
    min_free_memory_mb: int = 0
    max_retries: int = 0
    retry_delay_seconds: int = 60
    timeout_seconds: int = 0


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    feishu: FeishuConfig = field(default_factory=FeishuConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    task_defaults: TaskDefaults = field(default_factory=TaskDefaults)
    #: 配置文件所在目录，相对路径均以此为基准
    base_dir: Path = field(default_factory=Path.cwd)
    source_path: Path | None = None

    # ------------------------------------------------------------------ #
    def path(self, attr: str) -> Path:
        raw = getattr(self.paths, attr)
        p = Path(os.path.expanduser(str(raw)))
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def state_file(self) -> Path:
        return self.path("state_file")

    @property
    def log_file(self) -> Path:
        return self.path("log_file")

    @property
    def event_log(self) -> Path:
        return self.path("event_log")

    @property
    def task_log_dir(self) -> Path:
        return self.path("task_log_dir")

    @property
    def diagnostics_dir(self) -> Path:
        return self.path("diagnostics_dir")

    @property
    def pid_file(self) -> Path:
        return self.path("pid_file")

    @property
    def socket_file(self) -> Path:
        return self.path("socket_file")

    @property
    def queue_file(self) -> Path:
        return self.path("queue_file")

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"base_dir": str(self.base_dir)}
        for f in fields(self):
            if f.name in ("base_dir", "source_path"):
                continue
            value = getattr(self, f.name)
            if hasattr(value, "__dataclass_fields__"):
                sub = {}
                for sf in fields(value):
                    sv = getattr(value, sf.name)
                    if sf.name.endswith("_ids") or sf.name in ("monitor_gpus", "scheduler_allowed_gpus", "reserved_gpus"):
                        continue
                    if hasattr(sv, "__dataclass_fields__"):
                        continue
                    sub[sf.name] = sv
                out[f.name] = sub
            else:
                out[f.name] = value
        return out


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _build(raw: dict[str, Any], base_dir: Path, source: Path | None) -> Config:
    known = {
        "server", "gpu", "scheduler", "notify", "queue", "feishu", "runner", "paths",
        "storage", "task_defaults", "tasks", "watchdog",
    }
    if "interactive" in raw:
        raise ConfigError(
            "配置中的 interactive 段已合并到 feishu 段（同时移除了 webhook 群机器人方式）。\n"
            "迁移方法:\n"
            "  · app_id / app_secret  → 移到 feishu 段（通知与查询共用同一份凭证）\n"
            "  · enabled              → feishu.interactive_enabled（入站查询开关，与通知开关解耦）\n"
            "  · allowed_users / rate_limit_per_minute / log_lines_default / eta_* → 原样搬到 feishu 段\n"
            "  · feishu.mode / webhook_url / sign_secret → 删除（通知改用自建应用发送）"
        )
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            "配置中存在未知的顶层字段: " + ", ".join(sorted(unknown)) + f"（允许: {', '.join(sorted(known))}）"
        )

    cfg = Config(base_dir=base_dir, source_path=source)

    # ---- server ----
    sec = _require_mapping(raw.get("server"), "server")
    cfg.server.name = _get_str(sec, "name", "", "server")
    cfg.server.user = _get_str(sec, "user", "", "server")

    # ---- gpu ----
    sec = _require_mapping(raw.get("gpu"), "gpu")
    gpu = GpuConfig()
    gpu.max_idle_utilization = _get_int(sec, "max_idle_utilization", gpu.max_idle_utilization, "gpu", minimum=0)
    gpu.max_idle_memory_mb = _get_int(sec, "max_idle_memory_mb", gpu.max_idle_memory_mb, "gpu", minimum=0)
    gpu.require_no_compute_process = _get_bool(sec, "require_no_compute_process", gpu.require_no_compute_process, "gpu")
    gpu.check_interval_seconds = _get_int(sec, "check_interval_seconds", gpu.check_interval_seconds, "gpu", minimum=1)
    gpu.idle_confirm_checks = _get_int(sec, "idle_confirm_checks", gpu.idle_confirm_checks, "gpu", minimum=1)
    gpu.ignore_graphics_processes = _get_bool(sec, "ignore_graphics_processes", gpu.ignore_graphics_processes, "gpu")
    if "monitor_gpus" in sec:
        gpu.monitor_gpus = sec["monitor_gpus"]
    if "scheduler_allowed_gpus" in sec:
        gpu.scheduler_allowed_gpus = sec["scheduler_allowed_gpus"]
    reserved = sec.get("reserved_gpus")
    gpu.reserved_gpus = list(reserved) if isinstance(reserved, (list, tuple)) else []
    own_users = sec.get("own_users")
    gpu.own_users = [str(u) for u in own_users] if isinstance(own_users, (list, tuple)) else []
    gpu.resolve()
    cfg.gpu = gpu

    # ---- scheduler ----
    sec = _require_mapping(raw.get("scheduler"), "scheduler")
    sch = SchedulerConfig()
    sch.max_concurrent_tasks = _get_int(sec, "max_concurrent_tasks", sch.max_concurrent_tasks, "scheduler", minimum=0)
    sch.strategy = _get_str(sec, "strategy", sch.strategy, "scheduler")
    sch.launch_second_check = _get_bool(sec, "launch_second_check", sch.launch_second_check, "scheduler")
    sch.grace_before_reuse_seconds = _get_int(sec, "grace_before_reuse_seconds", sch.grace_before_reuse_seconds, "scheduler", minimum=0)
    sch.allow_cpu_tasks = _get_bool(sec, "allow_cpu_tasks", sch.allow_cpu_tasks, "scheduler")
    sch.waiting_notify_interval_seconds = _get_int(sec, "waiting_notify_interval_seconds", sch.waiting_notify_interval_seconds, "scheduler", minimum=0)
    sch.queue_notify = _get_bool(sec, "queue_notify", sch.queue_notify, "scheduler")
    sch.validate()
    cfg.scheduler = sch

    # ---- notify ----
    sec = _require_mapping(raw.get("notify"), "notify")
    no = NotifyConfig()
    for key in (
        "on_gpu_idle", "on_task_started", "on_task_finished", "on_task_failed",
        "on_task_queued", "on_error", "mention_all_on_idle", "mention_all_on_failure",
    ):
        setattr(no, key, _get_bool(sec, key, getattr(no, key), "notify"))
    no.idle_cooldown_seconds = _get_int(sec, "idle_cooldown_seconds", no.idle_cooldown_seconds, "notify", minimum=0)
    no.min_idle_gpus_to_notify = _get_int(sec, "min_idle_gpus_to_notify", no.min_idle_gpus_to_notify, "notify", minimum=1)
    no.failure_cooldown_seconds = _get_int(sec, "failure_cooldown_seconds", no.failure_cooldown_seconds, "notify", minimum=0)
    cfg.notify = no

    # ---- queue（任务列表展示上限 + 终态记录保留）----
    sec = _require_mapping(raw.get("queue"), "queue")
    qu = QueueConfig()
    qu.max_unfinished_in_reply = _get_int(
        sec, "max_unfinished_in_reply", qu.max_unfinished_in_reply, "queue", minimum=1
    )
    qu.max_finished_in_reply = _get_int(
        sec, "max_finished_in_reply", qu.max_finished_in_reply, "queue", minimum=0
    )
    qu.finished_retention_hours = _get_int(
        sec, "finished_retention_hours", qu.finished_retention_hours, "queue", minimum=0
    )
    qu.validate()
    cfg.queue = qu

    # ---- feishu ----
    sec = _require_mapping(raw.get("feishu"), "feishu")
    fs = FeishuConfig()
    removed = sorted(set(sec) & {"mode", "webhook_url", "sign_secret"})
    if removed:
        raise ConfigError(
            f"feishu 段中已移除的字段: {', '.join(removed)}。\n"
            "群机器人 webhook 方式已取消——通知现在通过自建应用发送:\n"
            "  · 填 feishu.app_id / feishu.app_secret（开放平台 → 自建应用 → 凭证）\n"
            "  · feishu.receive_id 指定收件人（留空则自动用 allowed_users 的第一个，即与机器人的私聊）\n"
            "  · 原 interactive 段也已并入 feishu（详见 README「飞书」一节）"
        )
    fs.enabled = _get_bool(sec, "enabled", fs.enabled, "feishu")
    fs.card = _get_bool(sec, "card", fs.card, "feishu")
    fs.mention_all = _get_bool(sec, "mention_all", fs.mention_all, "feishu")
    fs.app_id = _get_str(sec, "app_id", fs.app_id, "feishu")
    fs.app_secret = _get_str(sec, "app_secret", fs.app_secret, "feishu")
    fs.receive_id_type = _get_str(sec, "receive_id_type", fs.receive_id_type, "feishu")
    fs.receive_id = _get_str(sec, "receive_id", fs.receive_id, "feishu")
    fs.timeout_seconds = _get_int(sec, "timeout_seconds", fs.timeout_seconds, "feishu", minimum=1)
    fs.verify_ssl = _get_bool(sec, "verify_ssl", fs.verify_ssl, "feishu")
    fs.allow_any_host = _get_bool(sec, "allow_any_host", fs.allow_any_host, "feishu")
    hosts_raw = sec.get("allowed_hosts")
    if hosts_raw is None:
        pass
    elif isinstance(hosts_raw, (list, tuple)):
        fs.allowed_hosts = [str(h).strip().lower() for h in hosts_raw if str(h).strip()]
    else:
        raise ConfigError("feishu.allowed_hosts 必须是主机名列表，例如 ['open.feishu.cn']")
    # 入站交互查询 + ETA（原 interactive 段，现并入本段）
    fs.interactive_enabled = _get_bool(
        sec, "interactive_enabled", fs.interactive_enabled, "feishu"
    )
    fs.rate_limit_per_minute = _get_int(
        sec, "rate_limit_per_minute", fs.rate_limit_per_minute, "feishu", minimum=1
    )
    fs.log_lines_default = _get_int(
        sec, "log_lines_default", fs.log_lines_default, "feishu", minimum=1
    )
    users = sec.get("allowed_users")
    if users is None:
        fs.allowed_users = []
    elif isinstance(users, (list, tuple)):
        fs.allowed_users = [str(u).strip() for u in users if str(u).strip()]
    else:
        raise ConfigError("feishu.allowed_users 必须是 open_id 列表，例如 ['ou_xxxx']")
    fs.eta_enabled = _get_bool(sec, "eta_enabled", fs.eta_enabled, "feishu")
    fs.eta_sample_interval_seconds = _get_int(
        sec, "eta_sample_interval_seconds", fs.eta_sample_interval_seconds, "feishu", minimum=5
    )
    fs.eta_history_fallback = _get_bool(
        sec, "eta_history_fallback", fs.eta_history_fallback, "feishu"
    )
    fs.eta_history_max_runs = _get_int(
        sec, "eta_history_max_runs", fs.eta_history_max_runs, "feishu", minimum=1
    )
    patterns_raw = sec.get("eta_patterns")
    if patterns_raw is None:
        fs.eta_patterns = []
    elif isinstance(patterns_raw, (list, tuple)):
        fs.eta_patterns = [str(p) for p in patterns_raw if str(p).strip()]
    else:
        raise ConfigError("feishu.eta_patterns 必须是正则字符串列表")
    fs.validate()
    cfg.feishu = fs

    # ---- runner ----
    sec = _require_mapping(raw.get("runner"), "runner")
    ru = RunnerConfig()
    ru.conda_bin = _get_str(sec, "conda_bin", ru.conda_bin, "runner")
    ru.term_grace_seconds = _get_int(sec, "term_grace_seconds", ru.term_grace_seconds, "runner", minimum=0)
    ru.kill_after_grace = _get_bool(sec, "kill_after_grace", ru.kill_after_grace, "runner")
    ru.max_log_mb = _get_int(sec, "max_log_mb", ru.max_log_mb, "runner", minimum=0)
    ru.conda_default_env = _get_str(sec, "conda_default_env", ru.conda_default_env, "runner")
    extra = sec.get("extra_env")
    ru.extra_env = {str(k): str(v) for k, v in extra.items()} if isinstance(extra, dict) else {}
    cfg.runner = ru

    # ---- paths ----
    sec = _require_mapping(raw.get("paths"), "paths")
    pa = PathsConfig()
    for key in (
        "state_file", "log_file", "event_log", "task_log_dir", "diagnostics_dir",
        "pid_file", "socket_file", "queue_file",
    ):
        setattr(pa, key, _get_str(sec, key, getattr(pa, key), "paths"))
    cfg.paths = pa

    # ---- storage ----
    sec = _require_mapping(raw.get("storage"), "storage")
    st = StorageConfig()
    st.task_log_keep_files_per_task = _get_int(
        sec, "task_log_keep_files_per_task", st.task_log_keep_files_per_task, "storage", minimum=0
    )
    st.task_log_dir_max_mb = _get_int(
        sec, "task_log_dir_max_mb", st.task_log_dir_max_mb, "storage", minimum=0
    )
    st.prefer_keep_failed_logs = _get_bool(
        sec, "prefer_keep_failed_logs", st.prefer_keep_failed_logs, "storage"
    )
    st.daemon_log_max_mb = _get_int(sec, "daemon_log_max_mb", st.daemon_log_max_mb, "storage", minimum=0)
    st.daemon_log_backups = _get_int(sec, "daemon_log_backups", st.daemon_log_backups, "storage", minimum=0)
    st.events_max_mb = _get_int(sec, "events_max_mb", st.events_max_mb, "storage", minimum=0)
    st.daemon_out_max_mb = _get_int(sec, "daemon_out_max_mb", st.daemon_out_max_mb, "storage", minimum=0)
    st.prune_interval_seconds = _get_int(
        sec, "prune_interval_seconds", st.prune_interval_seconds, "storage", minimum=0
    )
    cfg.storage = st

    # ---- watchdog ----
    sec = _require_mapping(raw.get("watchdog"), "watchdog")
    # 新段做键名校验：阈值拼错时静默退回默认值最难排查。
    # （既有各段暂不校验，以免历史配置里无害的多余键导致升级后启动失败。）
    wd_keys = {f.name for f in fields(WatchdogConfig)}
    unknown_wd = set(sec) - wd_keys
    if unknown_wd:
        raise ConfigError(
            "watchdog 段存在未知字段: " + ", ".join(sorted(unknown_wd))
            + "（允许: " + ", ".join(sorted(wd_keys)) + "）"
        )
    wd = WatchdogConfig()
    wd.enabled = _get_bool(sec, "enabled", wd.enabled, "watchdog")
    wd.check_interval_seconds = _get_int(
        sec, "check_interval_seconds", wd.check_interval_seconds, "watchdog", minimum=10
    )
    wd.stall_threshold_seconds = _get_int(
        sec, "stall_threshold_seconds", wd.stall_threshold_seconds, "watchdog", minimum=10
    )
    wd.startup_grace_seconds = _get_int(
        sec, "startup_grace_seconds", wd.startup_grace_seconds, "watchdog", minimum=0
    )
    wd.repeat_alert_interval_seconds = _get_int(
        sec, "repeat_alert_interval_seconds", wd.repeat_alert_interval_seconds, "watchdog", minimum=0
    )
    wd.notify_on_recovery = _get_bool(sec, "notify_on_recovery", wd.notify_on_recovery, "watchdog")
    wd.instant_process_anomaly = _get_bool(
        sec, "instant_process_anomaly", wd.instant_process_anomaly, "watchdog"
    )
    wd.anomaly_confirm_checks = _get_int(
        sec, "anomaly_confirm_checks", wd.anomaly_confirm_checks, "watchdog", minimum=1
    )
    wd.log_tail_lines = _get_int(sec, "log_tail_lines", wd.log_tail_lines, "watchdog", minimum=1)
    wd.diagnostics_enabled = _get_bool(
        sec, "diagnostics_enabled", wd.diagnostics_enabled, "watchdog"
    )
    wd.diagnostics_keep_per_task = _get_int(
        sec, "diagnostics_keep_per_task", wd.diagnostics_keep_per_task, "watchdog", minimum=0
    )
    wd.diagnostics_max_mb = _get_int(
        sec, "diagnostics_max_mb", wd.diagnostics_max_mb, "watchdog", minimum=0
    )
    wd.include_gpu_snapshot = _get_bool(
        sec, "include_gpu_snapshot", wd.include_gpu_snapshot, "watchdog"
    )
    wd.validate()
    cfg.watchdog = wd

    # ---- task_defaults ----
    sec = _require_mapping(raw.get("task_defaults"), "task_defaults")
    td = TaskDefaults()
    td.gpu_num = _get_int(sec, "gpu_num", td.gpu_num, "task_defaults", minimum=0)
    td.priority = _get_int(sec, "priority", td.priority, "task_defaults")
    td.env = _get_str(sec, "env", td.env, "task_defaults")
    td.min_free_memory_mb = _get_int(sec, "min_free_memory_mb", td.min_free_memory_mb, "task_defaults", minimum=0)
    td.max_retries = _get_int(sec, "max_retries", td.max_retries, "task_defaults", minimum=0)
    td.retry_delay_seconds = _get_int(sec, "retry_delay_seconds", td.retry_delay_seconds, "task_defaults", minimum=0)
    td.timeout_seconds = _get_int(sec, "timeout_seconds", td.timeout_seconds, "task_defaults", minimum=0)
    cfg.task_defaults = td

    return cfg


#: 包所在项目根目录 —— 未显式指定 config 时用它找默认配置
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_config_candidates() -> list[Path]:
    """按优先级列出「默认配置」的候选路径。

    为什么需要这个: `gpu-manager` 装到 PATH 后会在**任意目录**下被调用
    （例如在 ~ 下敲命令），此时把 "config.yaml" 当当前目录的相对路径
    会直接报「配置文件不存在」——这正是用户碰到的报错。按以下顺序找:
      1. $GPU_MANAGER_CONFIG
      2. 当前目录 ./config.yaml（在项目里直接跑时最直观）
      3. 包所在项目根目录的 config.yaml（PATH 里调用时的正确默认值）
    """
    candidates: list[Path] = []
    env_path = os.environ.get("GPU_MANAGER_CONFIG")
    if env_path:
        candidates.append(abs_path(env_path))
    candidates.append(abs_path("config.yaml"))
    candidates.append(_PROJECT_ROOT / "config.yaml")
    return candidates


def load_config(path: str | os.PathLike[str] | None = None, *, must_exist: bool = True) -> Config:
    """加载并校验配置。相对路径以配置文件所在目录为基准。

    未显式给出 ``path`` 时按 :func:`default_config_candidates` 的顺序查找，
    因此从任何目录调用 ``gpu-manager`` 都能找到本项目的配置。
    """
    candidates = [abs_path(path)] if path is not None else default_config_candidates()
    for p in candidates:
        if not p.exists():
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"配置文件 YAML 解析失败: {p}\n{exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是映射: {p}")
        return _build(raw, p.parent, p)
    if must_exist:
        tried = "\n".join(f"    - {c}" for c in candidates)
        raise ConfigError(
            f"找不到配置文件。已尝试:\n{tried}\n  可用 `--config <路径>` 显式指定。"
        )
    return _build({}, candidates[-1].parent, None)
