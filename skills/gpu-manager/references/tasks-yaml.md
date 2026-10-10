# tasks.yaml 字段参考

`~/gpu-manager/tasks.yaml` 是**任务提交入口**：daemon 周期性读取它（幂等），
权威运行状态在 `data/state.json`（用 `gpu-manager list` 看）。手改这个文件永远不会与程序冲突。

## 语义

- **幂等**：内容没变 → 不重复入队；内容变了 → 视为一次新提交并重新入队。
  某任务正在运行时不会被打断（要停就先 `gpu-manager cancel <name>`）。
- **`status`** 字段只是提交指令：`pending`（默认，入队等待调度）/ `cancelled`（取消，
  运行中则优雅停止）/ `disabled`（忽略该条目，用于「暂时不跑」）。
- 多行命令用 YAML 的 `|`（保留换行）或 `>-`（折叠成一行）。

## 字段全表

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | ✅ | 唯一任务名；字母/数字开头，只含字母数字 `._-`（≤64） |
| `cwd` | ✅ | 训练脚本的工作目录。**写绝对路径**（见下方坑） |
| `command` | ✅ | 实际命令，通过 bash 执行 |
| `env` | | conda 环境名，或环境目录绝对路径（如 `/opt/envs/myenv`） |
| `gpu_num` | | 需要几张卡（默认取 `config.yaml` 的 `task_defaults.gpu_num`） |
| `priority` | | 越大越先调度；同级按提交顺序 FIFO（默认 0） |
| `min_free_memory_mb` | | 单卡最低空闲显存要求 MiB，0 = 不看显存余量 |
| `max_retries` | | 失败自动重试次数（超时不重试） |
| `retry_delay_seconds` | | 重试等待秒数 |
| `timeout_seconds` | | 运行超时秒数，超时优雅停止（0 = 不限） |
| `env_vars` | | 附加环境变量（与自动注入的合并） |
| `notes` | | 备注，出现在通知里 |
| `eta_logs` | | **额外的进度日志文件列表**（脚本把输出重定向走时用）。相对 `cwd` 或绝对路径 |
| `total_epochs` | | 整轮总轮数，用于把 `Epoch: [141] [10/63]` 换算成整轮进度 |
| `stage_file` | | 阶段标记文件（内容如 `training` / `evaluating`）。相对 `cwd` 或绝对路径 |

`eta_logs` / `total_epochs` / `stage_file` **只能写在 tasks.yaml**：CLI `gpu-manager add`
没有这些参数（它适合不需要 ETA 的简单任务）。

## ETA 为什么需要这些字段

- 训练脚本常把输出重定向到自己的 `console.log`，管理器捕获的 stdout 里就没有进度 →
  用 `eta_logs` 指过去。**日志在 `cwd` 之外时必须用绝对路径**（这是最常见的情况：
  实验目录和代码目录不同层）。
- DETR 系日志形如 `Epoch: [141]  [10/63]  eta: 0:00:53`：`[10/63]` 是**轮内**进度、
  行内 `eta` 只是当前轮剩余时间。整轮总轮数不在日志里 → 不填 `total_epochs`
  就只能报「第几轮」，不会编造结束时间。
- 多阶段流程（先 8 卡训练、再单卡 eval）→ 配 `stage_file`，阶段一变旧进度会被丢弃，
  避免用训练速度外推 eval 阶段。
- 其他保证：进度只接受单调递增；日志没增长时不记新样本；进度停滞 >5 分钟会降级置信度
  并标注「进度已 N 分钟未更新」；什么依据都没有时明确回答「无法估算」。

## Manager 自动注入的环境变量

`CUDA_VISIBLE_DEVICES=<分配到的卡>`、`PYTHONUNBUFFERED=1`，
以及 `GPU_MANAGER_TASK` / `GPU_MANAGER_GPU_IDS` / `GPU_MANAGER_SERVER` / `GPU_MANAGER_LOG`。

## 示例

多卡训练 + ETA（推荐给真实实验）：

```yaml
tasks:
  - name: exp500_srsdd
    cwd: /data/project
    command: >-
      python main.py --epochs 500 --batch_size 12
      --output_dir exps/OWDETR_original_500e_20261010
    env: myenv
    gpu_num: 8
    priority: 40
    eta_logs:
      - /data/project/exps/exp500/srsdd/console.log
    total_epochs: 500
    stage_file: /data/project/exps/exp500/srsdd/stage.status
    notes: 500 epochs
```

单卡小实验（低优先级插空跑，带超时保护）：

```yaml
  - name: quick_ablation
    cwd: /data/project
    env: myenv
    command: python tools/ablate.py --seed 0
    gpu_num: 1
    priority: 1
    timeout_seconds: 21600
```

多行 shell 脚本（等价于 `command: |`）：

```yaml
  - name: train_then_eval
    cwd: /data/project
    env: /opt/envs/myenv
    command: |
      set -e
      python train.py --epochs 10
      python eval.py --ckpt exps/latest.pth
    gpu_num: 4
    priority: 5
```

临时不跑某个任务：给它加 `status: disabled`；重新启用改回 `pending`（内容变化即重新入队）。

## 已知坑

| 坑 | 说明 |
| --- | --- |
| 相对 `cwd` | **按 daemon 进程的 cwd（`~/gpu-manager`）解析**，不是按配置文件目录。写 `cwd: work` 会变成 `~/gpu-manager/work` → 任务一启动就 `FileNotFoundError`。**用绝对路径。** |
| `eta_logs` 路径不存在 | 不会报错，只是没有进度（ETA 退化为「无法估算」）。可用 `gpu-manager logs <任务>` 与 `gpu-manager watchdog diagnostics <任务> --show` 对照确认。 |
| 忘了 `total_epochs` | DETR 类日志只能报「第几轮」，不会给出结束时间。 |
| 想改正在运行任务的参数 | 先 `cancel`，改完再提交（daemon 不会为改文件而打断运行中的任务）。 |
| 同名任务 | 已被终态任务占用时报错；CLI 用 `gpu-manager add --replace` 覆盖，文件里则先删掉旧条目或改名。 |
