# Cosmos3 / Cosmos-Predict2.5 ActionFollowing baseline 交接

本文档交接 Cosmos3-Nano 与 Cosmos-Predict2.5-2B 在 ActionFollowingData 上的 baseline 适配、真实数据 smoke、代码路径、持久化产物和后续 40k 正式训练边界。结构参考 [Ctrl-World 单任务模型交接](https://github.com/Ricardo520nono/ctrl-world-train-wjx/blob/dev-csx-codex/code/scripts_daily/20260719/README_ctrlworld_single_task_handoff_20260719.md)。

## 当前结论

- 两个模型的真实 `mix4` 20 optimizer-step smoke 均已在 AIHC `cce-pmm1yohj/train22` 成功完成。
- 两条成功 smoke 均为 effective global batch 16，没有使用 batch-8 OOM fallback。
- 已验证真实 clean/mix4 数据、50 tasks、Rot6D20 `[32,20]`、规定的 mix4 比例、finite loss、checkpoint 与 latest marker。
- 当前成功 job 是 `mix4` smoke；其中 clean 与四类 enhanced 均做过真实 decode/audit，但没有另起 clean-only 20-step job。
- Cosmos3 使用三视角；Cosmos-Predict2.5 使用原生 action-conditioned 单视角 head。
- `clean` 与 `mix4` 的 40k 训练配置已经写好，但当前没有 Cosmos3/Cosmos-Predict2.5 40k AIHC job。
- 因此当前状态是“适配和 smoke gate 通过”，不是“完整 baseline 复现完成”。完整复现还需要 2 个模型 x 2 个协议，共 4 条 40k 正式训练。

### 状态矩阵

| 模型 | 协议 | 代码 | 真实数据检查 | 20-step smoke | 40k job | 当前结论 |
|---|---|---|---|---|---|---|
| Cosmos3-Nano | `clean` | 已实现 | clean root、50 tasks、Rot6D20 已验证 | 未单独启动 clean-only smoke | 未创建 | 配置可审查，尚未形成训练结果 |
| Cosmos3-Nano | `mix4` | 已实现 | 五类数据、counts、10000 次 sampler audit、三视角已验证 | `job-s4qigpgr3lky` 成功 | 未创建 | smoke gate 通过 |
| Cosmos-Predict2.5-2B | `clean` | 已实现 | clean root、50 tasks、Rot6D20 已验证 | 未单独启动 clean-only smoke | 未创建 | 配置可审查，尚未形成训练结果 |
| Cosmos-Predict2.5-2B | `mix4` | 已实现 | 五类数据、counts、10000 次 sampler audit、单视角已验证 | `job-c469z4urkofj` 成功 | 未创建 | smoke gate 通过 |

### 这里的“成功”分别是什么意思

- `scheduler Succeeded`：只表示容器命令以 0 退出，不能单独作为训练成功证据。
- `smoke passed`：必须同时满足真实数据、50 tasks、Rot6D20、视角、20 optimizer steps、global batch 16、finite loss、持久化 checkpoint、latest marker 和 `SMOKE_RESULT.txt`。
- `baseline reproduced`：四条 `2 models x {clean,mix4}` 训练都真实完成 40,000 optimizer steps并通过最终产物审计。当前尚未达到这一状态。

## 固定实验约定

| 项目 | 约定 |
|---|---|
| 数据协议 | `clean`、`mix4` |
| task 数 | 50 |
| Action | chunk 32，Rot6D20，shape `[32,20]` |
| Rot6D | `concat(R[:,0], R[:,1])`，每臂 XYZ 3D + Rot6D 6D + gripper 1D |
| mix4 目标比例 | clean 50%；perturbed / random feasible / counterfactual replay / exploration 各 12.5% |
| Batch | effective global batch 16；只有真实 CUDA OOM 才允许降到 8 |
| 正式 steps | 40,000 optimizer steps |
| Cosmos3 视角 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| Cosmos2.5 视角 | `cam_high` |
| AIHC | pool `cce-pmm1yohj`，queue `train22`，8x A800 80GB |

full50 chunk32 有效 train counts：

```text
clean                 475122
perturbed             250000
random_feasible      1350000
counterfactual_replay 474645
exploration           121071
```

采样必须在 chunk-sample level 按权重实现，禁止 family-first。当前目标为：

```text
clean : perturbed : random_feasible : counterfactual_replay : exploration
  50% :      12.5% :            12.5% :                  12.5% :       12.5%
```

## Git 与目录状态

Cosmos3 与 Cosmos-Predict2.5 分属两个不同官方 upstream，GitHub 不能用一个 fork 同时保留两条 upstream lineage。因此公开交付使用两个配对 public forks，并统一使用同名分支：

```text
Cosmos3 public fork:
https://github.com/EasonAI-5589/cosmos-framework-actionfollowing

Cosmos2.5 public fork:
https://github.com/EasonAI-5589/cosmos-predict2.5-actionfollowing

delivery branch:
agent/actionfollowing-rot6d20-baseline
```

公开交付记录：

```text
Cosmos3 initial scoped commit:
dd51cbca4c8a79fef3b54ea684e2072242540c7a
draft review: https://github.com/EasonAI-5589/cosmos-framework-actionfollowing/pull/1

Cosmos2.5 initial scoped commit:
825520534173e91c9de426912ceea775b3dd70d8
draft review: https://github.com/EasonAI-5589/cosmos-predict2.5-actionfollowing/pull/2
```

两个 fork 只接收本文列出的 ActionFollowing-owned files、复现文档和对应 smoke bundle，不向 NVIDIA upstream 自动创建 PR。

### 获取公开交付代码

Cosmos3：

```bash
git clone https://github.com/EasonAI-5589/cosmos-framework-actionfollowing.git
cd cosmos-framework-actionfollowing
git switch agent/actionfollowing-rot6d20-baseline
git remote add upstream https://github.com/NVIDIA/cosmos-framework.git
```

Cosmos-Predict2.5 的官方仓库含较多 LFS 资产；只做代码审查时建议跳过 LFS smudge：

```bash
GIT_LFS_SKIP_SMUDGE=1 \
git clone https://github.com/EasonAI-5589/cosmos-predict2.5-actionfollowing.git
cd cosmos-predict2.5-actionfollowing
git switch agent/actionfollowing-rot6d20-baseline
git remote add upstream https://github.com/nvidia-cosmos/cosmos-predict2.5.git
```

公开分支的 upstream 基线与服务器成功 smoke 对齐：

```text
Cosmos3 upstream base:       26a50b8eb7b78fd8e0449918aa2d6e5b54fd9b8d
Cosmos-Predict2.5 base:      2650181ec50e15fbe5b3218544afddb214e1592b
delivery branch (both):      agent/actionfollowing-rot6d20-baseline
```

Cosmos-Predict2.5 基线 commit 不是任意更新到最新 upstream main 后得到的结果。升级 upstream 前必须重新跑配置 compose、loader unit test 和 20-step smoke，不能把旧成功证据直接继承到新 base。

运行环境与公开交付目录如下：

| 位置 | Git 状态 |
|---|---|
| 本地 `/Users/user/HumanoidX-DEV/ACWM` | 非 Git 仓库，保存 adapter 镜像、AIHC bundle 和本文档 |
| 服务器 `/mnt/gyc/cosmos-framework` | Git repo；`main`，HEAD `26a50b8eb7b7`，origin `NVIDIA/cosmos-framework`；ActionFollowing 适配尚未 commit |
| 服务器 `/mnt/gyc/cosmos-predict2.5` | Git repo；`dev-gyc-robotwin`，HEAD `2650181ec50e`，origin `nvidia-cosmos/cosmos-predict2.5`；工作树已有大量历史改动，ActionFollowing 适配尚未 commit |
| 服务器 `/mnt/gyc/Action-Following` | 非 Git 仓库，保存 AIHC job bundle |

不要直接把两个 dirty 服务器 worktree 全量提交。公开 forks 从已验证 upstream commit 创建干净分支，只迁移本文“ActionFollowing-owned files”列出的文件。

## ActionFollowing-owned files

### Cosmos3-Nano

本地镜像根：

```text
/Users/user/HumanoidX-DEV/ACWM/cosmos_adapters/cosmos3
```

服务器运行根：

```text
/mnt/gyc/cosmos-framework
```

核心文件：

```text
cosmos_framework/data/generator/action/datasets/actionfollowing_lerobot_dataset.py
cosmos_framework/configs/base/experiment/action/posttrain_config/action_forward_dynamics_actionfollowing_nano.py
examples/toml/sft_config/actionfollowing_full50_clean.toml
examples/toml/sft_config/actionfollowing_full50_mix4.toml
examples/launch_sft_actionfollowing_full50_clean.sh
examples/launch_sft_actionfollowing_full50_mix4.sh
tests/test_actionfollowing_lerobot_dataset.py
```

此外有两处注册/兼容修改：

```text
cosmos_framework/configs/base/config.py
cosmos_framework/data/generator/action/domain_utils.py
```

### Cosmos-Predict2.5

本地镜像根：

```text
/Users/user/HumanoidX-DEV/ACWM/cosmos_adapters/cosmos25
```

服务器运行根：

```text
/mnt/gyc/cosmos-predict2.5
```

核心文件：

```text
cosmos_predict2/_src/predict2/action/datasets/actionfollowing_rot6d20.py
cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_actionfollowing_rot6d20.py
cosmos_predict2/_src/reason1/tokenizer/processor.py
cosmos_predict2/_src/reason1/tokenizer/preprocessor_config.json
scripts/train_actionfollowing_full50_clean_8gpu.sh
scripts/train_actionfollowing_full50_mix4_8gpu.sh
tests/test_actionfollowing_rot6d20.py
```

### AIHC bundle

本地：

```text
/Users/user/HumanoidX-DEV/ACWM/aihc/cosmos_baseline_smoke_20260718
```

服务器：

```text
/mnt/gyc/Action-Following/jobs/20260718
```

核心文件：

```text
cosmos3_job.json
cosmos25_job.json
run_cosmos3_mix4_20step_smoke.sh
run_cosmos25_mix4_20step_smoke.sh
probe_actionfollowing_video_assets.sh
data_asset_probe_job.json
```

本地与服务器的两个核心 dataset adapter、两个成功 smoke 启动脚本 SHA256 已核对一致。

## 代码模块职责

### Cosmos3 模块

| 文件 | 职责 | 关键约束 |
|---|---|---|
| `actionfollowing_lerobot_dataset.py` | 枚举 full50 clean/mix4 sources、校验 metadata、展开 chunk index、按协议采样、解码三视角、生成 action spec | `chunk_length=32`、`action/state=20D`、`split=full`、`mode=forward_dynamics` |
| `action_forward_dynamics_actionfollowing_nano.py` | 注册 Cosmos3-Nano forward-dynamics experiment | 8-rank FSDP、per-rank 2 samples、global batch 16、33 visual frames、40k 默认配置 |
| `config.py` | 显式导入并注册新 experiment | 缺少该导入时 Hydra 找不到 experiment |
| `domain_utils.py` | 注册 `robotwin-actionfollowing` embodiment/domain | raw action dim 固定 20 |
| `actionfollowing_full50_{clean,mix4}.toml` | clean/mix4 40k 配置入口 | 使用前必须做 structured dryrun |
| `launch_sft_actionfollowing_full50_{clean,mix4}.sh` | 官方 repo 内的人工启动入口 | 只启动训练进程，不负责创建 AIHC job |
| `test_actionfollowing_lerobot_dataset.py` | source 数量、Rot6D20、mix4 audit、视频 fallback 回归测试 | 需要官方 Cosmos Python/CUDA dependencies |

Cosmos3 loader 的关键行为：

1. `clean` 枚举 50 个 task roots；`mix4` 枚举 350 个 roots：clean 50、perturbed 100、random feasible 100、counterfactual replay 50、exploration 50。
2. 每个 source 在构建索引时校验 `action` 和 `observation.state` 最后一维均为 20，并要求三视角 feature 全部存在。
3. `perturbed` 已是 chunk-level，每条 sample 只贡献一个 32-step prefix；其它 trajectory-level family 使用 stride-1 sliding windows。
4. sampler 在 flattened chunk index 上构建 deterministic weighted CDF，不先抽 family。
5. 每个样本输出 32 步 Rot6D20 action；视频使用 `cam_high/cam_left_wrist/cam_right_wrist`，32 帧解码后复制最后一帧形成 33-frame visual sequence。
6. 10 FPS enhanced 视频仍配 30 Hz action timeline，nearest-frame tolerance 为 `0.051s`；不得把 action 标签重采样到 10 Hz。

### Cosmos-Predict2.5 模块

| 文件 | 职责 | 关键约束 |
|---|---|---|
| `actionfollowing_rot6d20.py` | 独立 full50 dataset/dataloader、协议采样、单视角解码、LRU source/video cache | `[32,20]` action、33-frame head video、family-preserving bounded retry |
| `exp_actionfollowing_rot6d20.py` | 从官方 2B action-conditioned experiment 继承并覆盖数据/model/action 参数 | `action_dim=20`、`num_action_per_chunk=32`、batch 2 x 8 GPUs |
| `processor.py` | 允许 Reason1 processor 尊重显式 `cache_dir` | 有本地路径时禁止重新解析 S3/Hub |
| `preprocessor_config.json` | 本地 Qwen2.5-VL processor overlay | 与共享盘 Reason1 权重配套 |
| `train_actionfollowing_full50_{clean,mix4}_8gpu.sh` | 40k 人工启动入口 | 当前是 repo-level launcher，不是最终 AIHC submission bundle |
| `test_actionfollowing_rot6d20.py` | full50、Rot6D20、sampling、canonical kwargs、fallback/retry 测试 | 需要官方 CUDA extra 环境 |

Cosmos-Predict2.5 loader 的关键行为：

1. source 枚举、chunk 展开和 sampler 目标与 Cosmos3 完全一致。
2. 只读取原生 action-conditioned 路径支持的 `observation.images.cam_high`，不伪造多视角输入。
3. decord 输出先变成 contiguous Torch `TCHW`，再执行 torchvision resize，最后返回 `CTHW` uint8 tensor。
4. 真实 smoke 的单样本审计 shape 为 action `[32,20]`、head video `[3,33,256,320]`。
5. 缺失/坏视频触发的 bounded retry 必须留在原 sampled family 内，不能把 enhanced 样本静默替换成 clean。
6. 从父配置继承的 RoboTwin compatibility kwargs 只允许 canonical 值：`gripper_rescale_factor=1`、`num_action_per_chunk=32`、`fps_downsample_ratio=1`。
7. 父 joint-pretraining 配置残留的 `dataloaders` map 被专用 builder 丢弃；其它未知参数继续向下传递并显式报错。

## 端到端数据流

```text
ActionFollowingData_LeRobot_Rot6D/train
  -> 50 task names x clean/enhanced source roots
  -> metadata action/state/view validation
  -> effective chunk expansion
       perturbed: prefix-only, one chunk per sample
       other families: stride-1 sliding window
  -> deterministic chunk-level weighted sampler
  -> fixed-seed 10k sample audit
  -> video decode + Rot6D20 action assembly
  -> model-native batch format
       Cosmos3: three views, 33 visual frames, 32 actions
       Cosmos2.5: head view CTHW, 33 visual frames, 32 actions
  -> 8-GPU optimizer step
  -> persistent log/checkpoint/latest marker/SMOKE_RESULT
```

采样质量由代码动态根据 effective count 计算，不依赖写死的近似权重。对 family `f`：

```text
per_chunk_mass[f] = target_probability[f] / effective_chunk_count[f]
```

因此 sampled family probability 为：

```text
P(f) = N[f] * per_chunk_mass[f] / sum_i(N[i] * per_chunk_mass[i])
```

这也是为什么不能先均匀选择 family 再均匀选择 family 内样本：后者改变了 chunk-sample level 的定义，并会掩盖 subtype/task/trajectory/window 的实际质量差异。

## 数据与模型资产

canonical 数据：

```text
/mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train
```

AIHC 容器中公共盘挂载后，脚本会校验 `/mnt/public_ckp` alias。主要模型资产：

```text
Cosmos3 DCP:
/mnt/gyc_ckp/models/Cosmos3-Nano-DCP-411f42a8fdfb

Cosmos3 VAE:
/mnt/public_ckp/cosmos3-cache/wan22_vae/Wan2.2_VAE.pth

Cosmos3 tokenizer:
/mnt/public_ckp/cosmos3-cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/text_tokenizer

Cosmos2.5 action-conditioned checkpoint:
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt

Cosmos2.5 VAE tokenizer:
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/tokenizer.pth

Cosmos2.5 Reason1:
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Reason1-7B
```

## 已成功 smoke

| 模型 | AIHC job | 结果 | final loss | checkpoint |
|---|---|---|---:|---|
| Cosmos3-Nano | `job-s4qigpgr3lky` | `Succeeded`，20/20 steps，batch 16 | rank-0 `0.2027` | 137GB DCP，`iter_000000020` |
| Cosmos-Predict2.5-2B | `job-c469z4urkofj` | `Succeeded`，20/20 steps，batch 16 | `0.1634` | 20GB DCP，`iter_000000020` |

持久化输出：

```text
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12
```

两个目录均包含：

```text
SMOKE_RESULT.txt                  # status=passed, protocol=mix4, steps=20, effective_global_batch=16
.../checkpoints/latest_checkpoint.txt
.../checkpoints/iter_000000020/
```

注意：这些 checkpoint 只证明训练链路真实可用，不是 40k baseline 最终模型。

查询当前状态：

```bash
/root/.agents/skills/aihccli/scripts/aihc-agent.sh job get job-s4qigpgr3lky \
  -p cce-pmm1yohj -q train22 -s

/root/.agents/skills/aihccli/scripts/aihc-agent.sh job get job-c469z4urkofj \
  -p cce-pmm1yohj -q train22 -s
```

### Cosmos3 smoke 证据链

```text
AIHC name:
ACWM_cosmos3_full50_mix4_rot6d20_bs16_20step_smoke_train22_retry12_20260718

job ID:
job-s4qigpgr3lky

persistent root:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12

training log:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12/bs16/logs/actionfollowing_full50_mix4_sft.log
```

日志中的关键事实：

```text
effective counts:
clean=475122
counterfactual_replay=474645
exploration=121071
perturbed=250000
random_feasible=1350000

10k sampler audit:
clean=0.5003
perturbed=0.1250
random_feasible=0.1250
counterfactual_replay=0.1250
exploration=0.1247
max_abs_error=0.0003

iteration 20 rank-0 loss: 0.2027
all rank losses at iteration 20: finite
checkpoint: iter_000000020
checkpoint size: approximately 137 GB
terminal trainer message: Done with training.
```

`SMOKE_RESULT.txt`：

```text
status=passed
model=Cosmos3-Nano
protocol=mix4
steps=20
effective_global_batch=16
checkpoint=.../checkpoints/iter_000000020
```

### Cosmos-Predict2.5 smoke 证据链

```text
AIHC name:
ACWM_cosmos25_full50_mix4_rot6d20_bs16_20step_smoke_train22_retry12_20260718

job ID:
job-c469z4urkofj

persistent root:
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12

data/train log:
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12/bs16/train.log

trainer console:
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12/bs16/cosmos_predict2_action_conditioned/actionfollowing_full50/cosmos_predict25_afd_full50_mix4_rot6d20_a32_bs16_20step_smoke/console.log

trainer debug log:
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12/bs16/cosmos_predict2_action_conditioned/actionfollowing_full50/cosmos_predict25_afd_full50_mix4_rot6d20_a32_bs16_20step_smoke/debug.log
```

日志中的关键事实：

```text
effective counts: 与 Cosmos3 相同
10k sampler audit: 与 Cosmos3 相同，max_abs_error=0.0003
iteration 20 loss: 0.1634
checkpoint: iter_000000020
checkpoint size: approximately 20 GB
terminal trainer message: Done with training.
```

`SMOKE_RESULT.txt`：

```text
status=passed
model=Cosmos-Predict2.5-2B
protocol=mix4
steps=20
effective_global_batch=16
checkpoint=.../checkpoints/iter_000000020
```

### 复核持久化产物

```bash
for root in \
  /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12 \
  /mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12; do
  echo "===== $root ====="
  cat "$root/SMOKE_RESULT.txt"
  latest_file="$(find "$root" -name latest_checkpoint.txt -type f -print -quit)"
  test -n "$latest_file"
  latest_iter="$(cat "$latest_file")"
  echo "$latest_file=$latest_iter"
  test -d "$(dirname "$latest_file")/$latest_iter"
  du -sh "$(dirname "$latest_file")/$latest_iter"
done
```

### 本地/公开仓库静态验证

两套 public fork 已执行并通过：

- `git diff --check`；
- Python `py_compile`；
- shell `bash -n`；
- JSON parsing；
- Cosmos3 TOML parsing；
- secret-pattern scan；
- AIHC job bundle validator。

bundle validator 示例：

```bash
python3 docs/actionfollowing/tools/validate_job_bundle.py \
  --job-json docs/actionfollowing/aihc/cosmos3_job_20step_smoke.json \
  --script docs/actionfollowing/aihc/run_cosmos3_mix4_20step_smoke.sh \
  --model cosmos3 \
  --steps 20
```

```bash
python3 docs/actionfollowing/tools/validate_job_bundle.py \
  --job-json docs/actionfollowing/aihc/cosmos25_job_20step_smoke.json \
  --script docs/actionfollowing/aihc/run_cosmos25_mix4_20step_smoke.sh \
  --model cosmos25 \
  --steps 20
```

普通 Mac/login-node Python 环境不能作为 full pytest 结论：Cosmos3 测试依赖完整 framework packages，Cosmos-Predict2.5 import 会检查官方 CUDA extras。公开 PR 如实保留这一限制；真正的运行级验证来自完全相同 adapter/launcher 在 AIHC 容器中的成功 8-GPU smoke。升级依赖或 upstream base 后仍必须重新跑 pytest 与 smoke，不能复用旧结论。

## 40k 配置与启动边界

已有 40k 配置：

```text
Cosmos3 clean:
/mnt/gyc/cosmos-framework/examples/toml/sft_config/actionfollowing_full50_clean.toml

Cosmos3 mix4:
/mnt/gyc/cosmos-framework/examples/toml/sft_config/actionfollowing_full50_mix4.toml

Cosmos2.5 clean:
/mnt/gyc/cosmos-predict2.5/scripts/train_actionfollowing_full50_clean_8gpu.sh

Cosmos2.5 mix4:
/mnt/gyc/cosmos-predict2.5/scripts/train_actionfollowing_full50_mix4_8gpu.sh
```

当前没有为四条正式训练准备并提交最终 AIHC 40k job JSON。提交前必须：

1. 从成功 smoke bundle 派生四条独立 job JSON，名称显式包含模型、`clean/mix4`、`rot6d20`、`bs16`、`40k`。
2. 保持 8x A800、CPU 123、memory 970Gi、RDMA 1、shared memory 120Gi 和持久化 `pfs-Zx30ll` 挂载。
3. 重新跑 syntax、unit test、配置 compose、真实数据 audit 和 bundle validator。
4. 为每条任务使用独立持久化 output root，禁止覆盖 smoke 或其它协议结果。
5. 向用户报告四条 job 的精确名称、命令、镜像、资源、挂载和输出路径。
6. 只有获得单独明确的“提交 40k”授权后，才能执行 `job create`。

### 待生成的四条正式任务

下表是 handoff 约定，不代表任务已经创建：

| 模型 | 协议 | 建议 job name 模板 | 建议持久化 output root | 当前状态 |
|---|---|---|---|---|
| Cosmos3-Nano | clean | `ACWM_cosmos3_full50_clean_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos3/clean/train_40k_<date>` | 未生成 job JSON、未提交 |
| Cosmos3-Nano | mix4 | `ACWM_cosmos3_full50_mix4_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40k_<date>` | 未生成 job JSON、未提交 |
| Cosmos-Predict2.5 | clean | `ACWM_cosmos25_full50_clean_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/clean/train_40k_<date>` | 未生成 job JSON、未提交 |
| Cosmos-Predict2.5 | mix4 | `ACWM_cosmos25_full50_mix4_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/train_40k_<date>` | 未生成 job JSON、未提交 |

### 40k bundle 不能直接照抄 repo-level launcher

公开仓库中的 clean/mix4 launcher 表达模型配置意图，但最终 AIHC bundle 必须从已成功 retry12 smoke 脚本派生。原因：

- 登录节点路径与 AIHC 容器挂载路径不同；
- `/mnt/public_ckp` 在容器内需要由已挂载的 `/mnt/dataset/public_data` 建立并验证 alias；
- Cosmos3 必须继续使用持久化 DCP、本地 Wan2.2 VAE 和本地 Cosmos3-Nano tokenizer；
- Cosmos-Predict2.5 必须继续使用本地 action-conditioned checkpoint、VAE tokenizer、Reason1 与 processor overlay；
- repo-level Cosmos2.5 launcher 的通用 `IMAGINAIRE_OUTPUT_ROOT` 不是最终 40k PFS output root；
- final job JSON 必须显式保留 `pfs-Zx30ll` checkpoint/workspace/public data mounts；
- smoke 尾部的 `max_iter=20`、`logging_iter=1`、`save_iter=20`、scheduler cycle/warmup override 必须替换为审查后的 40k 值，而不是简单删除后假设父配置正确。

### 40k 提交前审计顺序

1. 在干净 public-fork checkout 上确认目标 commit，并记录 `git rev-parse HEAD`。
2. 将该 commit 的 ActionFollowing-owned files 同步到服务器运行 repo；比较 SHA256，禁止全目录覆盖 dirty worktree。
3. 对 `clean` 和 `mix4` 分别执行 loader metadata/decode probe；mix4 必须读取五类真实样本。
4. 固定 seed `20260717` 抽样至少 10000 次，确认各 family absolute error `<= 0.02`。
5. Cosmos3 执行 tokenizer preload、structured TOML dryrun 和 experiment registry 检查。
6. Cosmos-Predict2.5 执行 Hydra final-config compose，检查 dataloader target、canonical kwargs、local VAE/Reason1 路径。
7. 运行 `validate_job_bundle.py --steps 40000`，检查 job name、资源、挂载、命令、持久化输出和 checkpoint 配置。
8. 把四条任务的精确 JSON 摘要展示给用户并取得新的明确 launch 授权。
9. 执行 `job create` 后分别记录 job ID；Created/Pending 不等于 Running，不报告无依据 ETA。
10. 训练期间检查 optimizer step、finite loss 和 persistent log；不要只看控制台末尾。

### AIHC 资源合同

每条正式任务保持以下资源模板，除非重新审查：

```text
pool:             cce-pmm1yohj
queue:            train22
replicas:         1
GPU:              8 x baidu.com/a800_80g_cgpu
CPU:              123
memory:           970 GiB
RDMA:             1
shared memory:    120 GiB
effective batch:  16
checkpoint PFS:   pfs-Zx30ll
```

batch 8 不是排队紧张时的替代方案。只有 batch-16 运行日志出现真实 CUDA OOM，且原因不能通过明显配置错误修正时，才允许另建 batch-8 retry，并在 job name/handoff 中明确记录。

## 已修复的关键问题

- Cosmos3 Hugging Face checkpoint 已转换并持久化为 DCP；官方 Wan2.2 VAE 与本地 Qwen3-VL tokenizer 已固定。
- Cosmos3 注册了 ActionFollowing experiment，修复了并发 metadata loader 的 `tqdm` lock。
- enhanced split 中坏绝对软链通过受限 prefix remap 解析，不修改 canonical split。
- exploration 视频部分为 10 FPS，但 action label 保持 30 Hz；timestamp tolerance 固定为 `0.051s`，允许最近重复帧，不重定时 action。
- Cosmos2.5 loader 已把 NumPy HWC/THWC 转成 contiguous Torch TCHW。
- Cosmos2.5 固定使用本地官方 VAE、Reason1 权重与 processor overlay，避免运行时下载。
- Cosmos2.5 只接受 canonical `gripper_rescale_factor=1`、`num_action_per_chunk=32`、`fps_downsample_ratio=1`。
- mix4 断言使用真实 train effective counts，不得误用数据文档中的全量资产 counts。

### 失败定位表

| 现象 | 第一因果 | 最终处理 | 复验要求 |
|---|---|---|---|
| 容器找不到 repo/checkpoint | login node 路径没有对应到 AIHC PFS mount | 使用 job datasource mount，并在容器内建立受检 alias | `readlink -f`、文件大小、关键文件存在性 |
| Cosmos3 首次无法加载模型 | HF checkpoint 不是训练器要求的 DCP | 一次性 HF -> DCP 转换并写入持久化盘 | DCP 目录完整且后续 job 不重复转换 |
| Cosmos3 VAE fallback/download 失败 | VAE 资产不完整或路径不在容器内 | 固定官方 Wan2.2 VAE 本地文件 | 启动前 load probe |
| Cosmos3 experiment 不存在 | 新 experiment module 未显式 import | 在 base config 注册模块 | structured TOML dryrun 能 compose |
| Cosmos3 tokenizer vocab/path 异常 | offline model ID 命中不完整 cache | 指向 Cosmos3-Nano snapshot 内本地 `text_tokenizer` | processor/tokenizer 本地加载，vocab 151643 |
| 并发 metadata 初始化报 `tqdm._lock` | Hugging Face nested thread pool 竞态 | 外层线程池启动前调用 `tqdm.get_lock()` | 多线程 metadata discovery 完成 |
| enhanced video `FileNotFound` | split MP4 是指向旧绝对前缀的坏 symlink | 只对匹配旧 prefix 的 target 做 env remap | 真实 enhanced 视频成功打开；不修改 canonical split |
| Cosmos3 timestamp tolerance assertion | 10 FPS 视频配 30 Hz action timeline | 保持 30 Hz action，tolerance=`0.051s` | exploration decode probe 通过 |
| Cosmos2.5 torchvision 输入错误 | NumPy HWC/THWC 直接进入 Torch transform | contiguous NumPy -> Torch TCHW | `[3,33,256,320]` head video |
| Cosmos2.5 tokenizer 尝试远端下载 | VAE/Reason1 路径未完全本地化 | 固定本地 VAE、Reason1、processor overlay | 断网/无 Hub fallback 下能加载 |
| Hydra 报未知 dataset kwargs | 父 RoboTwin 参数残留 | 只接受并硬校验 canonical 三个值 | 错误值必须 fail loudly |
| Hydra 出现父 `dataloaders` map | joint-pretraining merge 残留 | 专用 builder 只删除该枚举 residue | final target/keys compose 检查 |
| counts 硬断言错误 | 文档全量资产 counts 被误当成 train effective counts | 使用真实 train chunk counts | 两个 loader 打印同一组五类 counts |
| rank0 后出现 NCCL abort/SIGTERM | 其它 rank 已先抛第一因果 traceback | 从完整持久化日志找最早 traceback | 不把 NCCL 尾声当 root cause |

诊断新失败时，必须先保存：job ID、pod、完整持久化 log、第一条 traceback、目标 commit、job JSON SHA、run-script SHA。没有这些信息时不要直接改多个层级，也不要删除失败证据。

## AIHC 监控与诊断命令

状态与 pod：

```bash
AIHC=/root/.agents/skills/aihccli/scripts/aihc-agent.sh

$AIHC job get <job-id> -p cce-pmm1yohj -q train22 -s
$AIHC job get <job-id> -p cce-pmm1yohj -q train22 --pods
$AIHC pod list <job-id> -p cce-pmm1yohj -q train22
```

日志：

```bash
$AIHC job logs <job-id> -p cce-pmm1yohj -q train22
```

控制面日志可能截断；一旦 job 已创建 persistent output root，应优先读对应 `train.log`、`console.log`、`debug.log` 或 Cosmos3 SFT log。状态口径：

| 状态 | 解释 | 操作 |
|---|---|---|
| `Created` | 控制面已接收，可能尚无 pod | 记录真实状态，继续检查；不编造 ETA |
| `Scheduled/Starting` | 已调度或容器启动中 | 检查 pod/event；不停止 |
| `Running` | 容器在运行 | 同时检查 optimizer step 和 persistent log |
| `Succeeded` | 命令 0 退出 | 继续核验 data audit、steps、loss、checkpoint、marker |
| `Failed` | 自然失败 | 定位第一 traceback；在授权范围内修复后新建 retry |
| `ManualTermination` | 被人工停止 | 不当成自然训练结论 |

未经新的明确授权，不停止或删除任何 `Created/Pending/Scheduled/Starting/Running` job。清理历史 job 时也必须逐个实时确认仍为 `Failed`，并保留成功 job 与持久化产物。

## Git 维护与后续开发

两个 public forks 的默认分支均为：

```text
agent/actionfollowing-rot6d20-baseline
```

`main` 保留官方 upstream 基线，draft PR 只开在自己的 fork 内：

```text
Cosmos3:  https://github.com/EasonAI-5589/cosmos-framework-actionfollowing/pull/1
Cosmos2.5:https://github.com/EasonAI-5589/cosmos-predict2.5-actionfollowing/pull/2
```

提交新修改的建议流程：

```bash
git status -sb
git diff --check
git diff -- <owned-files>

# 只添加本次负责文件，禁止 git add -A 污染 mixed worktree。
git add -- <exact-owned-files>
git diff --cached --check
git diff --cached --stat
git commit -m "Describe the scoped ActionFollowing change"
git push
```

从 official upstream 更新时，不在服务器 dirty worktree 直接 rebase/reset。应在 public fork 的干净 clone 中：

1. `git fetch upstream`；
2. 从目标 upstream commit 建新 review branch；
3. cherry-pick/重新应用 ActionFollowing-owned commits；
4. 解决冲突后跑静态检查、official-env pytest、配置 compose；
5. 重新跑 20-step smoke；
6. 只有新 smoke 通过后才更新 handoff 的 known-good base。

public fork、服务器运行 repo 和本地 adapter 镜像是三个不同角色：

| 位置 | 角色 | 是否直接训练 |
|---|---|---|
| public fork | 审查、版本控制、handoff | 否 |
| `/mnt/gyc/cosmos-framework`、`/mnt/gyc/cosmos-predict2.5` | 服务器运行源码 | 是，但同步必须按文件白名单 |
| `/Users/user/HumanoidX-DEV/ACWM/cosmos_adapters/*` | 本地已验证文件镜像 | 否 |

任何“代码已在 GitHub”都不自动等于“服务器已同步”，任何“服务器已同步”也不自动等于“AIHC 训练已提交”。三种状态必须分别报告。

## 接手检查清单

- [ ] 确认工作目录与目标 Git/fork，不要把两个 dirty 官方 repo 的无关改动带入提交。
- [ ] 重新核对本地 adapter 与服务器运行文件 SHA256。
- [ ] 读取 `action_following_data_assets.md`，保持 50 tasks、Rot6D20 与 mix4 chunk-sample 采样协议。
- [ ] 查询两条 smoke job 和持久化 marker，不能只看 scheduler `Succeeded`。
- [ ] 为 `Cosmos3 clean`、`Cosmos3 mix4`、`Cosmos2.5 clean`、`Cosmos2.5 mix4` 各自生成 40k bundle。
- [ ] 提交前展示精确资源与输出路径，并重新取得 40k launch 授权。
- [ ] 40k 运行后分别核验 finite loss、step 40000、latest marker、checkpoint 和最终数据 audit。

## 最终 40k 完成验收

四条正式任务分别满足以下条件后，才能把本 handoff 的总状态改为“baseline reproduced”：

- [ ] AIHC job 与 pod terminal `Succeeded`；
- [ ] persistent log 明确到达 optimizer step 40000；
- [ ] 全程没有未处理的 NaN/Inf 或连续 skip；
- [ ] effective global batch 与 job name/config 一致；
- [ ] clean run 只含 clean；mix4 run 的 sampler audit 保持目标比例；
- [ ] 50 tasks、Rot6D20 `[32,20]`、视角合同没有漂移；
- [ ] `latest_checkpoint.txt` 指向真实存在的 step-40000 checkpoint；
- [ ] checkpoint 位于持久化 PFS，而不是容器临时盘；
- [ ] 保存最终 config、job JSON、run script、commit SHA、数据 root 与关键模型资产 provenance；
- [ ] 四条任务分别形成可读的结果摘要，不能用一条模型的成功替代另一条。

## 本地运行手册

public fork 内的自包含副本：

```text
docs/actionfollowing/README.md
docs/actionfollowing/AIHC_RUNBOOK.md
docs/actionfollowing/action_following_data_assets.md
docs/actionfollowing/tools/validate_job_bundle.py
```

本机 Codex Skill 原始路径：

```text
/Users/user/.codex/skills/run-cosmos-actionfollowing/SKILL.md
/Users/user/.codex/skills/run-cosmos-actionfollowing/references/runbook.md
/Users/user/.codex/skills/run-cosmos-actionfollowing/references/action_following_data_assets.md
```
