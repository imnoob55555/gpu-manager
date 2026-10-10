# GPU Manager

实验室共享 GPU 服务器上的**个人** GPU 管理工具：没有待跑任务时监视 GPU 并在出现空闲卡
时通过飞书提醒；有待跑任务时自动找空闲卡、自动启动训练、跑完自动接着调度下一个。

- 常驻运行，不依赖 SSH 会话（`nohup`/`setsid` 等价方式后台化）
- 只服务当前用户，**不对其他用户做任何干预**（见下方安全边界）
- 单主进程 + 单主循环，状态机自动切换，无需人工干预

```text
┌─ GPU Manager（单进程常驻）────────────────────────────────────────┐
│                                                                    │
│  IDLE   无 pending、无 running                                     │
│    │    → 监视 GPU → 确认空闲 → 飞书通知                            │
│    │    （出现 pending 任务时自动转入 QUEUE）                       │
│    ↓                                                               │
│  QUEUE  有 pending、但资源不足或无 running                          │
│    │    → 每轮寻找满足条件的空闲 GPU                                │
│    ↓                                                               │
│  RUNNING 至少有一个自己启动的任务在跑                                │
│    │    → 监控进程/退出码 → 完成后继续调度下一个                     │
│    └──── 队列清空后自动回到 IDLE                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 1. 硬性安全边界

程序在代码层面保证（有回归测试锁定）：

| 保证 | 实现方式 |
| --- | --- |
| 不 kill / pause / signal 其他用户的进程 | 发信号前双重校验：`/proc/<pid>` 的 **owner 必须是当前用户**，且 **starttime 必须与启动时记录一致**（防 PID 复用）。任一不满足即拒绝发信号 |
| 不抢占已被他人占用的 GPU | 判定空闲时必须「无任何 compute 进程」，且启动前再做一次 `nvidia-smi` 复核 |
| 不清理他人显存、不 reset GPU、不使用 sudo | 代码中不存在这类调用；所有 GPU 相关操作都是只读查询 |
| 只停止「用户明确要求停止」的自己的任务 | 取消走 `SIGTERM` → grace(默认 20s) → `SIGKILL`，且只作用于**自己的进程组** |
| 不修改系统级 CUDA 配置 | 仅通过子进程环境变量 `CUDA_VISIBLE_DEVICES` 限制可见 GPU |

另外，`gpu-manager stop` 只杀 daemon 自己，**不会**杀掉正在运行的训练任务。

---

## 2. 快速开始

```bash
git clone git@github.com:imnoob55555/gpu-manager.git ~/gpu-manager
cd ~/gpu-manager

# 1) 从模板生成本地配置（config.yaml 已被 .gitignore 忽略，凭证不会进仓库）
cp config.example.yaml config.yaml
#    填飞书自建应用凭证（开放平台 → 自建应用 → 凭证与基础信息；见 §10）
#    通知与「在飞书里问状态」共用同一个应用，只需维护这一份凭证
vim config.yaml          # 填 feishu.app_id / feishu.app_secret

# 2) 自检（会检查 nvidia-smi / conda / 目录权限 / 任务定义，可选实测发一条飞书）
./bin/gpu-manager doctor
./bin/gpu-manager doctor --notify-test

# 3) 后台启动（SSH 断开、关笔记本都不影响）
./bin/gpu-manager start

# 4) 查看状态 / 日志
./bin/gpu-manager status
./bin/gpu-manager logs -f
```

想让命令全局可用：

```bash
export PATH="$HOME/gpu-manager/bin:$PATH"     # 追加到 ~/.bashrc 即可
# 或者正式安装（生成真正的 gpu-manager 命令）:
pip install -e ~/gpu-manager
```

### 提交任务

方式 A：写进 `tasks.yaml`（适合提前批量排实验，daemon 每轮自动同步，幂等）

```yaml
tasks:
  - name: dino_baseline
    cwd: /data/project
    env: myenv
    command: >
      python main.py --config_file configs/baseline.py
      --output_dir exps/baseline
    gpu_num: 2
    priority: 10
```

方式 B：命令行直接加（适合临时插一个）

```bash
./bin/gpu-manager add --name quick_ablation \
    --cmd "python tools/ablate.py --seed 0" \
    --cwd /data/project --env myenv \
    --gpu-num 1 --priority 1
```

资源满足后会自动启动，启动/完成/失败都会发飞书。

---

## 3. 命令参考

| 命令 | 说明 |
| --- | --- |
| `start [--foreground] [--dry-run] [--no-launch]` | 后台常驻启动；`--dry-run` 不发飞书（只写日志），`--no-launch` 只做调度决策不启动任务 |
| `stop [--force]` | 停止 daemon（**不杀训练任务**） |
| `restart` / `reload` | 重启 / 热重载配置与 `tasks.yaml`（也可 `kill -HUP <pid>`） |
| `status [--watch] [--json]` | GPU 利用率/显存/空闲确认进度/可调度性/占用者 + 模式 + 队列 + 看门狗告警 |
| `list [--status S] [--json]` | 任务列表（活跃在前） |
| `show NAME` | 单任务详情（含实际执行的等效命令） |
| `add ...` | 提交任务（`--dry-run` 只校验） |
| `cancel NAME` | 取消任务：运行中则优雅停止其进程组；未运行直接置 cancelled |
| `remove NAME` / `requeue NAME` | 移除终态任务 / 重新入队 |
| `logs [NAME] [-n N] [-f]` | daemon 日志（不带 NAME）或任务日志（带 NAME）；`-f` 实时跟踪 |
| `events [-n N]` | 事件流（启动、模式切换、任务状态变化、启动被放弃的原因） |
| `config [--json] [--raw]` | 生效配置（密钥已打码） |
| `doctor [--notify-test]` | 环境自检 |
| `prune [--dry-run] [--usage]` | 清理过期日志/查看磁盘占用（上限见 `storage` 段） |
| `watchdog [status\|diagnostics [任务]] [--show]` | 看门狗状态 / 诊断快照（只读；无任务时看门狗自动关闭） |

调试用：`./bin/gpu-manager start --foreground --dry-run --no-launch`
（前台跑、不发消息、只决策不启动，可安全验证逻辑）。

> **给 AI agent 用**：操作方法与排错经验已做成 skill，源文件在仓库 `skills/gpu-manager/`。
> 装法：把该目录链接或复制到你的 skill 目录即可 —— pi 用 `~/.pi/agent/skills/`
> （调用 `/skill:gpu-manager`），codex 用 `~/.codex/skills/`（调用 `$gpu-manager`）；
> 两者在任务匹配时也会自动加载。

---

## 4. 关键行为说明

### 连续空闲确认

`util<=5%` 且 `已用显存<=1000MiB` 且 `无 compute 进程` 只算「本次看起来空闲」，
必须**连续 `idle_confirm_checks`(默认 3) 次**成立才确认为可调度（≈90 秒）。
计数会持久化；若 daemon 停了很久，重启后计数清零重新确认（避免用陈旧计数误占卡）。

### 启动前二次检查

确认为空闲后、真正启动前，再跑一次 `nvidia-smi`：若发现已被别人占用 →
**放弃本次分配并继续等待**，绝不抢占。

### GPU 使用范围

```yaml
gpu:
  monitor_gpus: all              # 参与「空闲提醒」观察的卡
  scheduler_allowed_gpus: all    # 允许自动调度占用的卡；例如 [4,5,6,7]
  reserved_gpus: []              # 永不参与自动调度；例如 [7]
```

- `monitor_gpus` 之外的卡连提醒都不会出现
- `scheduler_allowed_gpus` 之外的卡可以提醒你（手动用），但不会被自动占用
- `reserved_gpus` 优先级最高：即使出现在 `scheduler_allowed_gpus` 里也无效

### 调度策略

`priority` 降序 → 同优先级 FIFO → 每个任务取**编号最小的 N 张**可用卡。
大任务资源不够时**跳过它继续看后面的任务**，不阻塞队列：A 需要 4 卡而只有 3 卡空闲时，
后面的 B(2 卡)、C(1 卡) 会先跑（有单元测试锁定）。

`max_concurrent_tasks`(默认 0=不限) 可限制同时运行的任务数。

### 通知去重与降级

- 空闲提醒按「空闲卡集合签名 + 冷却时间(`idle_cooldown_seconds`)」去重，避免 checkpoint
  导致 util 波动时刷屏
- IDLE 模式：上报所有确认空闲的卡；QUEUE/RUNNING 模式：只上报「调度器不会用、但你能手动用」
  的卡（避免与调度动作互相刷屏）
- 排队超过 `waiting_notify_interval_seconds` 会周期性提醒一次等待原因
- **通知没配好不影响调度**：凭证或收件人缺失时自动降级为只写日志，`doctor` 会提示
  （`feishu.app_id`/`app_secret`；收件人 `feishu.receive_id` 留空则自动用 `allowed_users` 第一人）
- **出站地址受白名单约束**（`feishu.allowed_hosts`，默认只允许飞书官方域名）：
  配置被误改/篡改时，通知内容不会被发到任意地址；用自建网关时把主机加进白名单，
  或设 `allow_any_host: true`
- **瞬时失败的重试交给 HTTP 库**（urllib3 Retry，含指数退避与 429/5xx 重试），
  而不是手写「重试一次 + 固定 sleep」

### 磁盘占用与日志保留

日志天然带硬上限，不需要手动清理（全部在 `storage` 段可调，`0` = 不限制）：

| 项目 | 默认上限 | 说明 |
| --- | --- | --- |
| 每个任务的日志 | 最近 10 次运行 | 每次运行一个文件，超出删最旧 |
| `logs/` 总量 | 200MB | 超出按最旧优先删 |
| 单个日志文件 | 64MB | 超过则轮转为 `.log.1` |
| `data/manager.log` | 5MB × 2 | daemon 自身日志 |
| `data/events.jsonl` | 5MB | 只保留尾部（最新事件） |
| `data/manager.out.log` | 5MB | 异常堆栈 |

- 清理以「一次运行的产物」为单位：`.log` / `.log.1` / `.run.sh` / `.exit` 一起删，
  不会留下孤立的 run.sh
- **正在运行任务的日志绝不会被删**
- 总量超限时**先删成功任务的日志**，失败任务的留久一点
  （`prefer_keep_failed_logs: true`）
- daemon 每 `storage.prune_interval_seconds`（默认 300 秒）自动检查一次；
  也可手动：`gpu-manager prune`（`--dry-run` 预览、`--usage` 只看占用）

### 看门狗（训练“卡住了”自动发现）

**只在有运行中任务时工作**；任务全部结束就自动关闭（不检查、不占资源），有新任务再自动开启。
每 `watchdog.check_interval_seconds`（默认 600s）检查一次，三类信号**任一有动静就算有进展**：

| 信号 | 来源 |
| --- | --- |
| 进度记录 | `task.progress_updated_at`（ETA 采样写入新样本的时间） |
| 日志时间 | 捕获日志 / `eta_logs` / 进度来源文件的 mtime |
| 进程状态 | `/proc/<pid>` 的存活、身份（starttime 防 PID 复用）、状态字符 |

两类告警：

- **停滞**：连续 `stall_threshold_seconds`（默认 3600s）无任何信号更新
  （任务启动后 `startup_grace_seconds` 默认 900s 宽限期内不判停滞，`编译/加载数据集` 不会误报）
- **进程异常**：进程消失 / 僵尸 Z / 停止 T / 不可中断 D / PID 复用 → 连续
  `anomaly_confirm_checks`（默认 2 轮）确认后**立即**告警，不等停滞阈值

告警行为：

- 首次立即告警，持续停滞时每 `repeat_alert_interval_seconds`（默认 3600s）重复提醒，
  恢复正常发一条「已恢复」（都可在 `watchdog` 段关闭）
- 同时把现场写到 `data/diagnostics/<任务>/<时间戳>/`：`summary.txt`（结论+信号时间线+ETA）、
  `process.txt`（/proc 详情）、`log_tail.txt`（日志尾部）、`progress.json`、`gpu.txt`（相关卡状态）
- 快照有独立上限：每任务 `diagnostics_keep_per_task`（5 份）+ 目录总量
  `diagnostics_max_mb`（50MB），**每个任务最新的那份永不删**；也可用 `gpu-manager prune` 统一清理
- 事件流里会记 `watchdog_stall` / `watchdog_anomaly` / `watchdog_recovered`，
  可用 `gpu-manager events` 或飞书交互查询追溯

查看：

```bash
gpu-manager watchdog                      # 是否在监视、每任务「最后进展」、告警情况
gpu-manager watchdog diagnostics          # 列出诊断快照
gpu-manager watchdog diagnostics exp1 --show   # 打印某任务最新一份摘要
```

**安全**：看门狗全程只读 `/proc` 与文件，**绝不对任何进程发信号**（有回归测试用
`os.kill` 守卫锁定：只允许 `kill(pid, 0)` 这种存在性探测）。它只负责“发现并告诉你”，
不会自行杀任务——是否需要干预由你决定。

### 飞书交互式查询（在飞书里问状态 / 预计结束时间）

配好自建应用后（见 §10），在飞书里给机器人发消息即可查询：

| 你发 | 回复 |
| --- | --- |
| `状态` | 每张卡的 util/显存/占用者/可调度性 + 当前模式 |
| `队列` | 任务列表（进度、预计结束、等待原因） |
| `预计` | 运行中任务的预计结束 + 各卡预计空出时间 |
| `日志 <任务名> [行数]` | 该任务日志尾部（默认 20 行） |
| `帮助` | 命令表 |

- **只读**：没有取消任务等写操作，发错命令也不会影响训练
- **fail-closed 白名单**：`feishu.allowed_users` 为空时拒绝所有人；
  未授权消息不回内容，只在日志里打印对方 open_id（方便你加入白名单）
- 预计结束时间由日志进度（tqdm / `Epoch x/y` / `step x/y`）两点采样外推；
  没有进度输出时退化为「历史同类任务中位耗时」，并在回复里标明依据与置信度
- 别人占用的卡无法估算（那不是本工具启动的任务），只会报「已运行多久」

---

## 5. 目录结构

```text
gpu-manager/
├── config.yaml            全部配置（阈值/白名单/通知/路径）
├── tasks.yaml             任务提交文件（只读导入，手改不会与程序冲突）
├── bin/gpu-manager        免安装启动脚本
├── gpu_manager/
│   ├── cli.py             命令行入口
│   ├── daemon.py          主循环与状态机
│   ├── scheduler.py       调度决策（纯函数，可单测）
│   ├── gpu_monitor.py     nvidia-smi 采集与空闲判定
│   ├── idle_tracker.py    连续空闲确认
│   ├── runner.py          启动/停止训练进程（含安全校验）
│   ├── task_queue.py      任务模型/校验/队列
│   ├── notifier.py        飞书通知（自建应用；与交互查询共用凭证）
│   ├── watchdog.py        看门狗（无进展/进程异常 → 告警 + 诊断快照）
│   ├── retention.py       日志与诊断快照的保留上限
│   ├── eta.py             日志进度解析与预计结束时间
│   ├── bot.py             飞书交互式查询（只读）
│   ├── state_store.py     状态与文件锁
│   ├── ipc.py             CLI ↔ daemon 的 UNIX socket
│   ├── config.py          配置加载与校验
│   └── utils.py           进程/路径/日志工具
├── data/                  运行时：state.json、manager.log、events.jsonl、manager.pid/sock
├── data/diagnostics/      看门狗告警时的现场快照（有独立上限）
├── logs/<任务名>/          每个任务的 stdout/stderr、run.sh、退出码
└── tests/                 506 个测试（含全部安全边界回归）
```

---

## 6. 功能一览

| 功能 | 实现位置 | 说明 |
| --- | --- | --- |
| 单进程主循环 + 模式自动切换 | `daemon.py` | IDLE / QUEUE / RUNNING 自动转换 |
| GPU 采集（index/util/显存/进程） | `gpu_monitor.collect` | 一次采集即快照；采集失败时**不判空闲、不调度**（保守） |
| 阈值全部配置化 | `config.yaml → gpu.*` | util / 显存 / 是否要求无 compute 进程 |
| 连续空闲确认 | `idle_tracker.IdleTracker` | 计数持久化 + 陈旧自动清零 |
| 启动前二次检查 | `daemon._launch_pending` + `gpu_monitor.still_free` | 失败则取消本次分配并继续等待 |
| GPU 三级范围语义 | `config.GpuConfig.schedulable_set` | monitor / allowed / reserved |
| 多用户安全边界 | `utils.safe_terminate` + `runner.verify_identity` | owner + starttime 双重校验，其余一律拒绝 |
| 任务定义与校验 | `task_queue.build_task` | name / cwd / command / env / gpu_num / priority 等 |
| 任务状态机 | `task_queue.Task.status` | pending / running / completed / failed / cancelled，无模糊态 |
| 子进程 GPU 隔离 | `runner.build_env` | 只改 `CUDA_VISIBLE_DEVICES`，不动系统配置 |
| 调度策略 | `scheduler.plan_assignments` | priority + first fit，大任务不阻塞队列 |
| 看门狗 | `watchdog.py` | 无进展 / 进程异常 → 告警 + 诊断快照 |
| 日志与磁盘治理 | `retention.py` | 两级上限；**绝不删运行中任务的日志** |
| ETA 估算 | `eta.py` | 日志进度采样 + 历史中位耗时兜底 |
| 飞书通知 / 交互查询 | `notifier.py` / `bot.py` | 同一个自建应用：出站推送 + 入站只读问答 |

---

## 7. 排错

```bash
./bin/gpu-manager doctor          # 先看这个：GPU/conda/权限/飞书/任务定义
./bin/gpu-manager status          # 每张卡为什么不可用、空闲确认进度
./bin/gpu-manager events -n 60    # 最近发生了什么（含启动被放弃的原因）
./bin/gpu-manager logs -n 200     # daemon 日志
```

常见情况：

| 现象 | 原因与处理 |
| --- | --- |
| 任务一直 pending | `status` 看「等待原因」列与「空闲确认」列；可能是阈值太严、卡不在白名单、或在等连续确认 |
| 想立刻用某张卡 | 用 `add` 提交任务即可；若该卡不在 `scheduler_allowed_gpus` 内则不会被自动占用（设计如此） |
| 飞书没收到 | `doctor --notify-test` 实测；检查 `feishu.app_id`/`app_secret` 与收件人（`receive_id` 留空则用 `allowed_users` 第一人） |
| 任务显示 `failed` 且退出码未知 | 进程消失但没有退出码记录（例如被 SIGKILL，或 daemon 曾在任务运行期间重启）。属保守判定，请看任务日志确认 |
| 修改了 config 想生效 | `./bin/gpu-manager reload`（改 `log_file` 需重启） |
| 想换 GPU 白名单 | 改 `config.yaml` 的 `gpu.*` 后 `reload` |

---

## 8. 设计取舍与已知保守行为

1. **tasks.yaml 只读**：daemon 从不改写它，因此你手改文件不会与程序冲突；权威运行状态在
   `data/state.json`（`list` 展示的就是它）。文件里的 `status` 只是提交指令
   （`pending` / `cancelled` / `disabled`），不是运行状态。
2. **README 第 7 节最后一行**：任务失败但退出码未知时判为 `failed` 而非 `completed`，
   宁可让你看到异常，也不把可能失败的任务当成功。
3. **daemon 退出不杀任务**：训练进程在独立会话中，daemon 停止/重启后仍继续跑，
   重启后自动接管监控（退出码用 `run.sh` 的 EXIT trap 记录在 `logs/<任务>/`，重启也不丢）。
4. **进程组清理的保守点**：若任务主进程已消失（无法再校验身份），程序不会再向该进程组发信号，
   避免误杀。此时残存子进程需你手动确认。
5. **采集失败即暂停决策**：`nvidia-smi` 读取失败的一轮不做任何调度（不会盲目占卡）。
6. **路径一律做包含校验**：日志/脚本/退出码文件的路径都由「任务名」拼出，
   先经 `safe_name` 字符白名单，再经 `resolve_within`（realpath + 目录包含）校验；
   读日志也只允许读取 `task_log_dir` 内的文件，软链逃逸会被拒绝。
7. **不实现复杂调度**：不做抢占、不做公平份额、不做跨机调度（第一版刻意保持简单可解释）。

---

## 9. 测试

```bash
python3 -m pytest tests/ -q
```

覆盖重点：GPU 输出解析（含「首行被当表头丢掉」导致误判空闲的回归）、空闲阈值与
「自己的任务也算占用」、连续确认与陈旧计数清零、二次检查拒绝他人占用、
first-fit 不阻塞大队列、`tasks.yaml` 幂等导入与「绝不打断运行中任务」、
PID 复用拒绝发信号、不向其他用户进程发信号、子进程组一并停止、
`set -e`/`exit N` 场景退出码不丢失、真实启动/停止/日志/退出码端到端、
通知主机白名单与降级行为、配置阈值/白名单语义、路径包含校验与软链逃逸防护、
看门狗（无进展/进程异常/恢复）与日志上限策略。

测试文件分工：`test_gpu_monitor` / `test_idle_tracker` / `test_scheduler` /
`test_task_queue` / `test_runner_safety`（安全边界）/ `test_utils` /
`test_config` / `test_notifier` / `test_cli`（含后台启停端到端）/ `test_daemon` /
`test_retention`（磁盘占用）/ `test_eta`（进度外推）/ `test_bot`（交互查询与白名单）。

静态检查：项目根有 `pyrightconfig.json`；少数位置带行内 lint 抑制注释
（每处均写明原因，例如「路径已由 `resolve_within` 校验」「目标是 pytest `tmp_path` 内的临时文件」）。

---

## 10. 飞书配置（通知 + 交互式查询）

**只用一个自建应用**：通知（出站）与「在飞书里问状态」（入站）共用同一份凭证，
`config.yaml` 里也只有 `feishu` 一段。

为什么不用自定义群机器人 Webhook：它只能「发」**不能「收」**（接收你发的消息必须用
自建应用 + 长连接 WebSocket），而且还得另外维护一份凭证。既然查询必须用应用，
通知也用同一个应用发——少一份配置、少一处会失效的地方。

两个开关**解耦**：`feishu.enabled` 控制通知，`feishu.interactive_enabled` 控制入站查询；
关掉查询不影响通知（反之亦然）。长连接由服务器主动外连飞书，因此**不需要公网地址或域名**，
NAT 后的实验室服务器可用。

### 10.1 开放平台配置（一次性，约 5 分钟）

1. 打开 <https://open.feishu.cn/app> → **创建企业自建应用**（名字例如 GPU Manager Bot）
2. 「添加应用能力」→ 开启 **机器人**
3. 「权限管理」添加权限：`im:message`（获取与发送单聊、群组消息）
   与 `im:message:send_as_bot`（以应用身份发消息）
4. 「事件与回调」→ 订阅方式选 **使用长连接接收事件** →
   添加事件 `im.message.receive_v1`（接收消息）
5. 「版本管理与发布」→ 创建版本并发布（企业内应用可能需要管理员审批）
6. 「凭证与基础信息」复制 App ID / App Secret 填入 `config.yaml`：

   ```yaml
   feishu:
     app_id: "cli_xxxxxx"
     app_secret: "xxxxxxxx"
     enabled: true              # 发送通知
     interactive_enabled: true  # 接收查询（长连接）
   ```

7. 安装官方 SDK（唯一的可选依赖）并重启：

   ```bash
   python3 -m pip install lark-oapi
   gpu-manager restart
   ```

8. **把你自己加入白名单**：先直接给机器人发一条「状态」——它不会回复
   （fail-closed），但日志里会打印你的 open_id：

   ```bash
   gpu-manager logs -n 100 | grep open_id
   ```

   把 `ou_xxxxxxxx` 填进 `feishu.allowed_users`，然后 `gpu-manager reload`：

   ```yaml
   feishu:
     allowed_users: ["ou_xxxxxxxx"]   # 同时也是通知的默认收件人（私聊）
   ```

   > 群里使用时需要 **@机器人**（飞书只把被 @ 的消息推给应用）；也可以直接与机器人私聊。

9. 自检：`gpu-manager doctor` 会检查 SDK、凭证、白名单是否齐备。

### 10.2 ETA（预计结束时间）怎么估的

按优先级依次尝试，**并在回复里标明依据**：

| 优先级 | 依据 | 适用 |
| --- | --- | --- |
| 1 | tqdm 自带的剩余时间 `[01:23<02:05, 3.6it/s]` | 直接用，置信度最高 |
| 2 | 日志进度两点采样外推（`Epoch 12/500` / `step 1000/5000`） | 需要两次采样（默认间隔 60s） |
| 3 | 历史中位耗时：先找**同名任务**，没有就用**全部已完成任务** | 日志无可用进度时兑底 |

**训练脚本把输出重定向到自己的日志时（很常见），要在任务里声明这三项：**

```yaml
  - name: exp500
    cwd: /data/project/exps/exp500/source
    command: bash run.sh
    gpu_num: 8
    total_epochs: 500           # 把「Epoch: [141] [10/63]」换算成整轮进度
    eta_logs:                   # 脚本自己写的日志
      - /data/project/exps/exp500/console.log   # 相对 cwd 或绝对路径
    stage_file: /data/project/exps/exp500/stage.status
```

> 实用提醒：`eta_logs` / `stage_file` 的相对路径是相对于任务的 `cwd`。
> 很多实验脚本的 `cwd` 是 `.../source/`，而日志写在它的**兄弟目录** `.../<experiment>/`，
> 这时必须用**绝对路径**，否则找不到文件（管理器会诚实报「无法估算」，不会乱猜）。

为什么需要 `total_epochs`：DETR 系日志形如
`Epoch: [141] [10/63] eta: 0:00:53 …`，其中行内的 `eta` 是**本 epoch 的剩余时间**，
`[10/63]` 是轮内迭代进度，而**总轮数不在日志里**。不声明 `total_epochs` 时
管理器只会报「第几轮」，**不会**把 53 秒当成整轮剩余时间。

其他保证：

- 只接受**单调递增**的进度；进度回退（换 epoch / 新阶段）会重置样本，
  绝不用旧数据外推
- 日志文件没增长时**不记新样本**（否则会在 eval 阶段用训练速度外推）
- `stage_file` 变化时丢弃旧进度（多阶段流程：先训 8 卡、再单卡 eval）
- 进度停滞超过 5 分钟会降级置信度并标注「进度已 N 分钟未更新」
- 什么依据都没有时明确回答「无法估算」，不编造时间
- 自定义进度正则可写在 `feishu.eta_patterns`（命名组：`done` / `total` / `remain`）

### 10.3 安全边界

- **只读**：命令集固定，没有任何写操作（不能取消任务、不能改配置）
- **fail-closed 白名单**：`allowed_users` 为空 = 拒绝所有人；
  未授权请求**不读取任何状态**、不回任何内容，只记审计日志
- **每用户限流**（默认 10 条/分钟）+ `message_id` 去重（飞书事件至少投递一次）
- **无 shell 透传**：只认命令表里的固定命令
- **入站全量审计**到 `data/events.jsonl`（谁、何时、要了什么、是否回复成功）
- 出站回复同样受 `feishu.allowed_hosts` 主机白名单约束（防 SSRF）
- 回复内容只包含**你自己的**任务信息；其他用户只显示 nvidia-smi 本就公开的 pid/uid

### 10.4 排错

| 现象 | 处理 |
| --- | --- |
| 发消息机器人不回 | `gpu-manager logs -n 50`：多半是白名单未配（会打印你的 open_id）、未装 SDK、或事件订阅没选「长连接」 |
| 日志出现「未安装 lark-oapi」 | `python3 -m pip install lark-oapi` 后 `gpu-manager restart` |
| 回了「请求太频繁」 | 调大 `feishu.rate_limit_per_minute` |
| ETA 显示「无法估算」 | 训练脚本没有进度输出；同类任务跑完一次后会自动用历史中位耗时兑底 |
| 群里 @了但无响应 | 确认应用已发布并通过审批、机器人已加入该群 |
| 没收到通知 | 检查 `feishu.enabled`、`app_id`/`app_secret` 与收件人（`receive_id` 留空则用 `allowed_users` 第一人）；`doctor --notify-test` 实测 |
| 想临时关掉 | `feishu.interactive_enabled: false` + `gpu-manager reload` |
