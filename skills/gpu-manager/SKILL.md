---
name: gpu-manager
description: >-
  操作本机（实验室共享服务器）的个人 GPU 管理器：查看 GPU 是否空闲/被谁占用、提交与排队训练任务、看任务进度与日志、任务卡住或无进展时的看门狗诊断、磁盘与日志清理、飞书机器人查询。Use when the user asks 看 GPU / 有没有空闲卡 / 提交训练 / 任务跑到哪了 / 还要多久 / 任务没动静或卡住了 / 清理日志与磁盘 / gpu-manager 怎么用或怎么排查, or asks to run any gpu-manager command.
metadata:
  short-description: 管理本机 GPU 与训练任务队列
compatibility: 需要本机已安装 gpu-manager（clone 后 `pip install -e .`，或把 bin/ 加入 PATH）
---

# GPU Manager（本机 GPU 与训练任务队列）

`gpu-manager` 是这台共享服务器上**只服务当前用户** 的个人工具：空闲 GPU 提醒 +
训练任务自动排队调度 + 日志/磁盘治理 + 看门狗 + 飞书查询。项目在 `~/gpu-manager/`，
命令已在 PATH（等价于 `~/gpu-manager/bin/gpu-manager`）。

## 何时用

- 有没有空闲卡 / 现在能不能跑 / 卡上是谁 → 状态类
- 提交训练 / 排队 / 调优先级 / 取消 → 任务类
- 跑到哪了 / 还要多久 / 怎么没输出 / 是不是卡住了 / 为什么失败 → 进度与排障
- 日志太大 / 磁盘占用 / 清理 → 占用类
- 怎么没收到提醒 / 机器人怎么用 → 通知类

## 铁律（先读，别越界）

1. **对他人只读**：绝不对其他用户的进程发信号、不 kill、不清显存、不 reset GPU、不用 sudo。
2. **只操作本工具管理的任务**：`cancel` 只对**你自己启动的进程组**发 TERM→KILL。
3. **权威状态在 `data/state.json`**；`tasks.yaml` 只是提交入口，daemon 只读它。
4. **影响在跑训练的动作先问用户**：`cancel`、`stop`/`restart` daemon、改 `config.yaml` 阈值。
   （`restart` 本身不杀训练任务，但会短暂停调度。）
5. 改配置 → `gpu-manager reload` 即可；**改了代码/新增模块才需要 `restart`**。

## 状态与查询

| 需求 | 命令 |
| --- | --- |
| 总览（GPU / 模式 / 队列 / 看门狗告警） | `gpu-manager status` |
| 持续刷新 | `gpu-manager status --watch 5` |
| 机器可读（含 `progress` 原始样本与 `watchdog` 段） | `gpu-manager status --json` |
| 任务列表（活跃在前，可过滤） | `gpu-manager list` / `--status running\|pending\|failed` |
| 单任务详情（cwd / 命令 / GPU / 时间 / 退出码） | `gpu-manager show <任务>` |
| 任务日志（可跟） | `gpu-manager logs <任务> -n 100 -f` |
| daemon 日志 / 事件流 / 生效配置 / 自检 | `gpu-manager logs -n 50` · `events -n 30` · `config` · `doctor` |

**进度与预计结束时间（ETA）**：CLI 的 `list`/`show` **不显示**进度和 ETA。要进度就用：
`gpu-manager logs <任务> -n 50`（最可靠）；或 `gpu-manager status --json` 取
`tasks[].progress`、`progress_updated_at`、`stage`；估算好的 ETA 只在**飞书「预计」/「队列」**
和**看门狗诊断快照**里。ETA 依赖任务声明 `eta_logs`（DETR 类还需 `total_epochs`，多阶段需
`stage_file`），而这些字段**只能写在 `tasks.yaml`**（CLI `add` 没有对应参数）。

## 提交任务

简单任务（不需要 ETA）：

```bash
gpu-manager add --name exp1 --cmd "python main.py --epochs 100" \
  --cwd /data/project --env myenv --gpu-num 8 --priority 30 --dry-run
# 去掉 --dry-run 才真正入队；--replace 覆盖同名终态任务
```

需要 ETA 或更复杂（推荐写文件，可复查、可版本化）——编辑 `~/gpu-manager/tasks.yaml`：

```yaml
tasks:
  - name: exp500_srsdd
    cwd: /data/project   # 绝对路径
    command: >-
      python main.py --epochs 500 --batch_size 12
    env: myenv
    gpu_num: 8
    priority: 40
    eta_logs:                                          # 脚本自己重定向的日志
      - /data/project/exps/exp500/srsdd/console.log
    total_epochs: 500                                  # DETR 类日志必需
    stage_file: /data/project/exps/exp500/srsdd/stage.status
    notes: 500 epochs
```

`tasks.yaml` 语义：daemon **只读且幂等**（内容变 = 一次新提交、重新入队；正在跑的任务不会被打断）；
`status: disabled` 可临时不跑，`status: cancelled` 可取消。字段全表与更多示例见
[references/tasks-yaml.md](references/tasks-yaml.md)。

## 取消 / 重排

```bash
gpu-manager cancel <任务>    # 运行中：优雅停止自己的进程组；未运行：直接置 cancelled
gpu-manager requeue <任务>   # 终态任务重新排队
gpu-manager remove <任务>    # 从队列移除终态任务（不影响日志文件）
```

## 看门狗（任务卡住自动发现）

- 只在有 running 任务时工作；没有任务时显示「已启用（当前无任务，自动关闭）」是**正常**的
- 默认：每 600s 检查 / 3600s 无进展告警 / 启动宽限 900s / 每小时重复提醒 / 恢复时通知
- 三类信号任一有动静即算有进展：进度记录 · 日志 mtime · `/proc` 进程状态
- 进程异常（消失 / 僵尸 Z / 停止 T / 不可中断 D / PID 复用）连续 2 轮确认后**立即**告警
- 告警 = 飞书通知 + 诊断快照 + `events.jsonl`（`watchdog_stall` / `watchdog_anomaly` / `watchdog_recovered`）

```bash
gpu-manager watchdog                              # 在监视谁、每任务最后进展、是否告警
gpu-manager watchdog diagnostics                  # 列出诊断快照
gpu-manager watchdog diagnostics <任务> --show    # 现场摘要（信号时间线/进程/日志尾部/GPU/ETA）
```

排查「是不是卡了」的顺序：`watchdog` → `logs <任务> -n 50` → `watchdog diagnostics <任务> --show`。
看门狗**只告警、不处置**（绝不对进程发信号）；要不要干预由人决定。

## 日志与磁盘

上限在 `config.yaml` 的 `storage` / `watchdog` 段（`0` = 不限制）：每任务留最近 10 次运行、
`logs/` 200MB、单文件 64MB 轮转、`events.jsonl` 5MB、诊断快照每任务 5 份 / 50MB
（**每任务最新一份永不删**）。**正在运行任务的日志绝不删**。

```bash
gpu-manager prune --usage     # 看占用与生效上限
gpu-manager prune --dry-run   # 预览会删什么
gpu-manager prune             # 立即清理
```

## 通知与飞书

- 推送：空闲提醒、任务启动/完成/失败、看门狗告警（`config.yaml` 的 `notify` / `feishu`）
- 交互查询（**只读**，白名单 fail-closed）：发「状态」「队列」「预计」「日志 <任务> [行数]」「帮助」
- 没收到提醒先查：`gpu-manager doctor`；再看 `gpu-manager logs -n 50 | grep -E "通知|未授权"`

## 排错（都是踩过的坑）

| 现象 | 原因 / 处理 |
| --- | --- |
| 一提交就 `failed`，日志 `FileNotFoundError` | `tasks.yaml` 里**相对 `cwd` 是按 daemon 进程 cwd（`~/gpu-manager`）解析**，不是配置文件目录 → 写绝对路径 |
| 日志里看不到 epoch/step | 脚本把输出重定向到别处 → 在 `tasks.yaml` 填 `eta_logs`（绝对路径）+ `total_epochs` |
| ETA 明显不对（跨阶段乱） | 配 `stage_file`；阶段变化时旧进度会被丢弃 |
| 改配置不生效 | `gpu-manager reload`；改代码 / 新增模块才需要 `restart` |
| `list` 状态长期不动 | daemon 没在跑：`gpu-manager status` 第一行；`gpu-manager start` |
| 机器人不回消息 | 白名单未配（`logs` 里会打印你的 open_id）、SDK 缺失、事件订阅没选长连接 |
| 看门狗告警太频繁 | 放宽 `watchdog.stall_threshold_seconds` 或加大 `startup_grace_seconds` |

## 验证提交/操作是否生效

```bash
gpu-manager list --status pending   # 新任务应出现（或已直接进入 running）
gpu-manager status                  # 模式应切到 QUEUE / RUNNING
gpu-manager logs <任务> -n 30       # 被调度后应看到脚本输出
```
