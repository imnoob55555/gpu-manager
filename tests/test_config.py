"""配置加载与校验测试（阈值与白名单语义）。"""

from __future__ import annotations

import pytest
import yaml

from gpu_manager.config import ConfigError, load_config


def write(tmp_path, raw: dict):
    path = tmp_path / "config.yaml"
    # pi-lens-ignore: python-path-traversal -- 目标是 pytest tmp_path 内的临时文件
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def base(**overrides) -> dict:
    raw = {
        "gpu": {"monitor_gpus": "all", "scheduler_allowed_gpus": "all"},
        "feishu": {"enabled": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    return raw


# --------------------------------------------------------------------------- #
# 基本校验
# --------------------------------------------------------------------------- #
def test_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml", must_exist=False)
    assert cfg.gpu.max_idle_utilization == 5
    assert cfg.gpu.idle_confirm_checks == 3


def test_unknown_top_level_key_rejected(tmp_path):
    path = write(tmp_path, {**base(), "gpu2": {}})
    with pytest.raises(ConfigError, match="未知的顶层字段"):
        load_config(path)


def test_yaml_error_reported_with_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("gpu: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML 解析失败"):
        load_config(path)


@pytest.mark.parametrize(
    "gpu,patch",
    [
        ({"max_idle_utilization": 101}, "0..100"),
        ({"max_idle_utilization": -1}, ">="),
        ({"max_idle_memory_mb": -5}, ">="),
        ({"idle_confirm_checks": 0}, ">="),
        ({"check_interval_seconds": 0}, ">="),
    ],
)
def test_gpu_thresholds_validated(tmp_path, gpu, patch):
    path = write(tmp_path, base(gpu=gpu))
    with pytest.raises(ConfigError, match=patch):
        load_config(path)


def test_bool_and_int_are_strict(tmp_path):
    path = write(tmp_path, base(gpu={"idle_confirm_checks": "abc"}))
    with pytest.raises(ConfigError, match="整数"):
        load_config(path)


def test_bool_accepts_common_strings(tmp_path):
    path = write(tmp_path, base(gpu={"require_no_compute_process": "false"}))
    assert load_config(path).gpu.require_no_compute_process is False


def test_scheduler_strategy_validated(tmp_path):
    path = write(tmp_path, base(scheduler={"strategy": "magic"}))
    with pytest.raises(ConfigError, match="priority_first_fit"):
        load_config(path)


# --------------------------------------------------------------------------- #
# GPU 白名单语义
# --------------------------------------------------------------------------- #
def test_gpu_scope_all_means_every_visible_gpu(tmp_path):
    cfg = load_config(write(tmp_path, base(gpu={"monitor_gpus": "all", "scheduler_allowed_gpus": "all"})))
    all_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    assert cfg.gpu.monitor_set(all_ids) == all_ids
    assert cfg.gpu.schedulable_set(all_ids) == all_ids


def test_scheduler_allowed_limits_auto_scheduling(tmp_path):
    cfg = load_config(write(tmp_path, base(gpu={"scheduler_allowed_gpus": [4, 5, 6, 7]})))
    all_ids = list(range(8))
    assert cfg.gpu.monitor_set(all_ids) == all_ids, "monitor 仍是全部卡（空闲提醒照常）"
    assert cfg.gpu.schedulable_set(all_ids) == [4, 5, 6, 7]


def test_reserved_gpus_override_allowed(tmp_path):
    cfg = load_config(
        write(tmp_path, base(gpu={"scheduler_allowed_gpus": [4, 5, 6, 7], "reserved_gpus": [7]}))
    )
    assert cfg.gpu.schedulable_set(list(range(8))) == [4, 5, 6], "reserved 优先级最高"


def test_monitor_subset_excludes_unwatched_gpus(tmp_path):
    cfg = load_config(write(tmp_path, base(gpu={"monitor_gpus": [0, 1]})))
    assert cfg.gpu.monitor_set(list(range(8))) == [0, 1]
    assert cfg.gpu.monitor_set([3, 4]) == [], "配置里的卡若不存在则不应出现"


def test_range_syntax_and_dedup(tmp_path):
    cfg = load_config(write(tmp_path, base(gpu={"scheduler_allowed_gpus": "4-7", "reserved_gpus": [5, 5]})))
    assert cfg.gpu.schedulable_set(list(range(8))) == [4, 6, 7]


def test_own_users_whitelist(tmp_path):
    cfg = load_config(write(tmp_path, base(gpu={"own_users": ["alice", "lab"]})))
    assert cfg.gpu.owns_username("alice")
    assert not cfg.gpu.owns_username("other")


# --------------------------------------------------------------------------- #
# 飞书配置：缺少凭证只警告，不阻塞
# ---------------------------------------------------------------------------
def test_missing_credentials_is_warning_not_error(tmp_path):
    cfg = load_config(write(tmp_path, base(feishu={"enabled": True, "app_id": "", "app_secret": ""})))
    warnings = cfg.feishu.credential_warnings()
    assert warnings, "应给出降级原因"
    assert "app_id/app_secret" in " ".join(warnings)


def test_missing_target_is_warning_not_error(tmp_path):
    """凭证齐备但没任何收件人 → 也要告知降级。"""
    cfg = load_config(
        write(tmp_path, base(feishu={"enabled": True, "app_id": "cli_x", "app_secret": "s"}))
    )
    assert "receive_id" in " ".join(cfg.feishu.credential_warnings())


def test_removed_webhook_fields_rejected_with_migration_hint(tmp_path):
    """群机器人 webhook 已取消：配置里再出现这些字段要启动即报错并给替代写法。"""
    for legacy in ({"mode": "webhook"}, {"webhook_url": "https://x/y"}, {"sign_secret": "s"}):
        path = write(tmp_path, base(feishu=legacy))
        with pytest.raises(ConfigError, match="已移除的字段"):
            load_config(path)


def test_interactive_section_rejected_with_migration_hint(tmp_path):
    """原 interactive 段已合并到 feishu 段。"""
    path = write(tmp_path, base(interactive={"enabled": True, "app_id": "cli_x"}))
    with pytest.raises(ConfigError, match="已合并到 feishu"):
        load_config(path)


def test_allowed_hosts_extension_allows_custom_gateway(tmp_path):
    cfg = load_config(
        write(tmp_path, base(feishu={"enabled": True, "allowed_hosts": ["gw.corp.internal"]}))
    )
    assert cfg.feishu.allowed_hosts == ["gw.corp.internal"]


def test_allow_any_host_parsed(tmp_path):
    cfg = load_config(write(tmp_path, base(feishu={"allow_any_host": True})))
    assert cfg.feishu.allow_any_host is True


def test_allowed_hosts_must_be_list(tmp_path):
    path = write(tmp_path, base(feishu={"allowed_hosts": "open.feishu.cn"}))
    with pytest.raises(ConfigError, match="allowed_hosts"):
        load_config(path)


# --------------------------------------------------------------------------- #
# 飞书：通知目标解析（私聊为默认）
# --------------------------------------------------------------------------- #
def test_receive_id_falls_back_to_whitelist_first_user(tmp_path):
    """留空 receive_id 时用白名单第一人（与机器人私聊）—— 零额外配置就能收到通知。"""
    cfg = load_config(
        write(
            tmp_path,
            base(feishu={"allowed_users": ["ou_a", "ou_b"], "receive_id": ""}),
        )
    )
    assert cfg.feishu.resolved_receive_id() == "ou_a"


def test_receive_id_type_not_open_id_requires_explicit_target(tmp_path):
    """白名单里是 open_id：想让通知发到群（chat_id）就必须显式填 receive_id，
    不能拿 open_id 当 chat_id 用。"""
    cfg = load_config(
        write(
            tmp_path,
            base(
                feishu={
                    "enabled": True,
                    "app_id": "cli_x",
                    "app_secret": "s",
                    "allowed_users": ["ou_a"],
                    "receive_id_type": "chat_id",
                    "receive_id": "",
                }
            ),
        )
    )
    assert cfg.feishu.resolved_receive_id() == ""
    assert "receive_id" in " ".join(cfg.feishu.credential_warnings())


def test_receive_id_type_validated(tmp_path):
    path = write(tmp_path, base(feishu={"receive_id_type": "群"}))
    with pytest.raises(ConfigError, match="receive_id_type"):
        load_config(path)


def test_interactive_toggle_is_independent_of_notifications(tmp_path):
    """入站/出站解耦：关掉长连接（interactive_enabled=false）不影响通知的可用性。"""
    cfg = load_config(
        write(
            tmp_path,
            base(
                feishu={
                    "enabled": True,
                    "interactive_enabled": False,
                    "app_id": "cli_x",
                    "app_secret": "s",
                    "allowed_users": ["ou_a"],
                    "receive_id": "",
                }
            ),
        )
    )
    assert cfg.feishu.credential_warnings() == []
    assert cfg.feishu.resolved_receive_id() == "ou_a"


# --------------------------------------------------------------------------- #
# 路径与导出
# --------------------------------------------------------------------------- #
def test_relative_paths_resolved_against_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, base(paths={"state_file": "data/state.json"})))
    assert cfg.state_file == tmp_path / "data/state.json"
    assert cfg.task_log_dir == tmp_path / "logs"


def test_absolute_paths_kept(tmp_path):
    cfg = load_config(write(tmp_path, base(paths={"task_log_dir": "/data/project/logs"})))
    assert str(cfg.task_log_dir) == "/data/project/logs"


def test_as_dict_hides_nothing_but_reports_shapes(tmp_path):
    cfg = load_config(write(tmp_path, base(server={"name": "labhost"})))
    data = cfg.as_dict()
    assert data["server"]["name"] == "labhost"
    assert "gpu" in data and "scheduler" in data
    assert "watchdog" in data, "新增配置段也应出现在生效配置里"


# --------------------------------------------------------------------------- #
# 看门狗
# --------------------------------------------------------------------------- #
def test_watchdog_defaults(tmp_path):
    cfg = load_config(write(tmp_path, base()))
    w = cfg.watchdog
    assert w.enabled is True
    assert (w.check_interval_seconds, w.stall_threshold_seconds) == (600, 3600)
    assert w.startup_grace_seconds == 900
    assert w.repeat_alert_interval_seconds == 3600
    assert w.notify_on_recovery is True and w.instant_process_anomaly is True
    assert w.anomaly_confirm_checks == 2
    assert (w.diagnostics_keep_per_task, w.diagnostics_max_mb) == (5, 50)
    assert cfg.diagnostics_dir == tmp_path / "data/diagnostics"


def test_watchdog_overrides_parsed(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            base(
                watchdog={
                    "enabled": False,
                    "check_interval_seconds": 120,
                    "stall_threshold_seconds": 600,
                    "repeat_alert_interval_seconds": 0,
                    "anomaly_confirm_checks": 1,
                    "diagnostics_keep_per_task": 1,
                    "diagnostics_max_mb": 10,
                },
                paths={"diagnostics_dir": "diag"},
            ),
        )
    )
    assert cfg.watchdog.enabled is False
    assert cfg.watchdog.summary() == "已禁用"
    assert cfg.watchdog.repeat_alert_interval_seconds == 0
    assert cfg.diagnostics_dir == tmp_path / "diag"


def test_watchdog_threshold_below_check_interval_rejected(tmp_path):
    """检查周期比阈值还长时阈值失去意义（每轮都必然超阈值）。"""
    path = write(
        tmp_path,
        base(watchdog={"check_interval_seconds": 600, "stall_threshold_seconds": 300}),
    )
    with pytest.raises(ConfigError, match="stall_threshold_seconds"):
        load_config(path)


def test_watchdog_repeat_interval_bounds(tmp_path):
    """0 = 不重复（放行）；很小的正数也没问题（每个检查周期最多提醒一次）。"""
    cfg = load_config(write(tmp_path, base(watchdog={"repeat_alert_interval_seconds": 0})))
    assert cfg.watchdog.repeat_alert_interval_seconds == 0
    cfg = load_config(write(tmp_path, base(watchdog={"repeat_alert_interval_seconds": 30})))
    assert cfg.watchdog.repeat_alert_interval_seconds == 30
    with pytest.raises(ConfigError, match="repeat_alert_interval_seconds"):
        load_config(write(tmp_path, base(watchdog={"repeat_alert_interval_seconds": -1})))


def test_watchdog_small_thresholds_allowed_when_consistent(tmp_path):
    """阈值高低是使用者的选择：只要「阈值 >= 检查周期」就应放行（沙箱/快速任务要用）。"""
    cfg = load_config(
        write(
            tmp_path,
            base(
                watchdog={
                    "check_interval_seconds": 10,
                    "stall_threshold_seconds": 10,
                    "startup_grace_seconds": 0,
                }
            ),
        )
    )
    assert cfg.watchdog.stall_threshold_seconds == 10


def test_watchdog_unknown_key_rejected(tmp_path):
    """阈值拼错必须报错，否则会静默退回默认值（最难排查）。"""
    path = write(tmp_path, base(watchdog={"stall_threshold_second": 900}))
    with pytest.raises(ConfigError, match="watchdog 段存在未知字段"):
        load_config(path)
