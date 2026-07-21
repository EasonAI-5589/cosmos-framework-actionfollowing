# Cosmos3 / Cosmos-Predict2.5 ActionFollowing baseline 交接

本文档交接 Cosmos3-Nano 与 Cosmos-Predict2.5-2B 在 ActionFollowingData 上的 baseline 适配、真实数据 smoke、代码路径、持久化产物和后续 40k 正式训练边界。结构参考 [Ctrl-World 单任务模型交接](https://github.com/Ricardo520nono/ctrl-world-train-wjx/blob/dev-csx-codex/code/scripts_daily/20260719/README_ctrlworld_single_task_handoff_20260719.md)。

## 当前结论

- Cosmos-Predict2.5 的真实 `mix4` 20 optimizer-step smoke 已在 AIHC `cce-pmm1yohj/train22` 完成并仍然有效。
- Cosmos3 修复版真实 `mix4` 20 optimizer-step smoke `job-12x2lgpx0a6p` 已于 2026-07-21 在 AIHC `cce-pmm1yohj/train` 完成并通过 canonical gate：global batch 16、20/20 steps、finite rank-0 final loss `0.1316`、137G DCP `iter_000000020`、latest marker、`DATA_AUDIT.log`、`TRAIN_AUDIT.json` 和 `SMOKE_RESULT.txt` 均已核验。
- Cosmos3 loader 已改为读取真实 `O[t]..O[t+32]`，并使用 LeRobot task metadata 中的 RoboTwin `full_description`；成功 smoke 直接读取 Motus mix41111 使用的 symlink-free Rot6D20 train root。真实审计证明五类各覆盖 50 tasks、action `[32,20]`、三视角 33 帧，并对五类各完成 decode probe；100k sampler 的观测比例为 clean `0.49995`、四类 enhanced 约 `0.125`，最大绝对误差 `5e-5`。
- Cosmos3 历史 smoke `job-s4qigpgr3lky` 虽然完成 20 steps、finite loss 和 checkpoint，但复查发现它把第 32 个 observation 复制成尾帧且 text prompt 为空；该结果仍只能作为历史 bring-up 证据，不能作为 canonical smoke gate 或修复版续训起点。
- 首次 `train` 提交 `job-ploabmfspj7o` 已自然失败。第一因果来自 node log：`[FATAL] RoboTwin task_instruction root missing`；它只运行到 bootstrap，没有进入数据 decode 或 optimizer step。修复版不再依赖未挂载的个人 RoboTwin 工作区，而是使用仓库内、固定到 RoboTwin 官方 commit `c3ddfa8b97d5519efa828b075999bd0006778e5e` 的 50-task `full_description` manifest。
- 已逐项比较 manifest 与官方 50 个 JSON，并在 Motus symlink-free train root 上检查全部 350 个 source：clean / perturbed / random feasible / counterfactual replay / exploration 均覆盖 50 tasks，文本全部一致。真实 LeRobot v2 `tasks.parquet` 把文本保存在 `__index_level_0__`，loader 会恢复为 task 文本；smoke 现已显式审计并记录该列。
- manifest/bootstrap retry2 `job-3bxehdg8flpj` 因共享 Hugging Face datasets cache 只读而自然失败；writable-cache retry3 `job-12x2lgpx0a6p` 已完成全部真实数据审计和 20-step 训练，成为当前 Cosmos3 canonical smoke gate。
- Cosmos3 使用三视角；Cosmos-Predict2.5 使用原生 action-conditioned 单视角 head。
- 用户在 smoke gate 通过后已单独授权正式训练；Cosmos3 `mix4` 40k `job-3cmb7l4p44jw` 已提交到 `cce-pmm1yohj/train21`，并于 2026-07-21 06:26:58 +08 进入 `Running`。复核 Motus 的 clean/mix4 成对基线后，Cosmos3 clean 40k bundle 也已补齐，使用相同 Motus clean 数据协议并把 checkpoint 周期改为 5000 steps；提交状态以本节状态矩阵和运行记录为准。
- 因此当前状态是“Cosmos3 与 Cosmos2.5 的 `mix4` 20-step smoke gate 均通过，Cosmos3 `mix4` 40k 已开始运行，Cosmos3 clean 40k 已形成受检 bundle”，不是“完整 baseline 复现完成”。只有正式 job 真实到达 step 40000 并通过最终产物审计后，才能报告对应 40k 结果成功。

### 状态矩阵

| 模型 | 协议 | 代码 | 真实数据检查 | 20-step smoke | 40k job | 当前结论 |
|---|---|---|---|---|---|---|
| Cosmos3-Nano | `clean` | 修复已实现，40k bundle validator 通过 | 真实 clean decode、50 tasks、Rot6D20 已由 canonical smoke/bundle audit 覆盖 | 未单独启动 clean-only smoke | 5K checkpoint bundle 已准备 | 待 PR 与 AIHC 运行级预检后提交 |
| Cosmos3-Nano | `mix4` | 修复已实现，12 unit tests/ruff/bundle validator 通过 | 五类真实 decode、50 tasks、`[32,20]`、33 帧三视角、官方 full_description 与 100k sampler audit 通过 | `job-12x2lgpx0a6p` 成功，20/20 steps，batch 16，loss `0.1316` | `job-3cmb7l4p44jw`，`train21`，Running | 正式训练中，尚未完成 |
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
| AIHC | pool `cce-pmm1yohj`；本轮修复 smoke 按用户要求使用 queue `train`，历史 smoke 在 `train22`；8x A800 80GB |

基础 32-action chunk counts（Cosmos-Predict2.5 保持此口径）：

```text
clean                 475122
perturbed             250000
random_feasible      1350000
counterfactual_replay 474645
exploration           121071
```

Cosmos3 forward dynamics 必须为 32 个 action 读取 33 个真实 observation，因此每条 trajectory 排除最后一个没有 `O[t+32]` 的起点；perturbed chunk-level prefix 不变。修复后的 model-specific counts 为：

```text
clean                 472622
perturbed             250000
random_feasible      1345000
counterfactual_replay 472145
exploration           120821
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

公开分支的 upstream 基线与历史服务器 bring-up 对齐；Cosmos3 本轮修复仍需重新同步和 smoke：

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
docs/actionfollowing/assets/robotwin_50_full_descriptions.json
docs/actionfollowing/aihc/run_cosmos3_mix4_20step_smoke.sh
docs/actionfollowing/tools/validate_job_bundle.py
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

历史 smoke 时本地与服务器 SHA256 曾核对一致；本轮 Cosmos3 `future32_prompt` 修复在重新同步后必须生成新的 SHA256 证据，旧记录不可复用。

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
3. `perturbed` 已是 chunk-level，每条 sample 只贡献一个 32-step prefix；其它 trajectory-level family 使用 stride-1 sliding windows，并排除没有真实 `O[t+32]` 的 terminal start。
4. sampler 在 flattened chunk index 上构建 deterministic weighted CDF，不先抽 family。
5. 每个样本输出 32 步 Rot6D20 action；三视角分别查询 33 个时间点并解码真实 `O[t]..O[t+32]`，禁止复制尾帧。
6. `ai_caption` 必须来自 LeRobot `sample["task"]`，其 metadata 已与仓库内固定版本的 RoboTwin `full_description` manifest 对齐；manifest 记录官方 repository、commit 和原始 JSON path，空 prompt 或任一 source 不一致必须 fail loudly。
7. 10 FPS enhanced 视频仍配 30 Hz action timeline，nearest-frame tolerance 为 `0.051s`；不得把 action 标签重采样到 10 Hz。

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

本轮 Cosmos3 与 Motus 数据端对齐使用同一份 symlink-free train root：

```text
AIHC container:
/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train

login node equivalent:
/mnt/public_ckp/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train
```

“同一份数据”只共享数据资产、50 tasks、Rot6D20 和 mix41111 目标比例。Cosmos3 仍使用自己的 forward-dynamics 口径：`A[t]..A[t+31]` 对应 33 个真实 observation，并使用 Cosmos3-specific effective counts，不能照搬 Motus 的 48-action loader。

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

## Smoke 状态

| 模型 | AIHC job | 结果 | final loss | checkpoint |
|---|---|---|---:|---|
| Cosmos3-Nano | `job-12x2lgpx0a6p` | `Succeeded`，canonical `current1+future32` + full-description smoke，20/20 steps，batch 16 | `0.1316` | 137G DCP，`iter_000000020` |
| Cosmos3-Nano | `job-s4qigpgr3lky` | 历史 bring-up 完成，但 timeline/prompt 语义错误，canonical gate 无效 | rank-0 `0.2027` | 历史 137GB DCP，禁止作为修复版起点 |
| Cosmos-Predict2.5-2B | `job-c469z4urkofj` | `Succeeded`，20/20 steps，batch 16 | `0.1634` | 20GB DCP，`iter_000000020` |

持久化输出：

```text
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12
/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12
```

canonical 成功目录包含以下最终 marker；旧 Cosmos3 目录中的同名 marker 仅保留为历史记录：

```text
SMOKE_RESULT.txt                  # status=passed, protocol=mix4, steps=20, effective_global_batch=16
.../checkpoints/latest_checkpoint.txt
.../checkpoints/iter_000000020/
```

注意：Cosmos3 历史 checkpoint 使用复制尾帧和空 prompt，既不是 40k baseline，也不是修复版可续训 checkpoint。Cosmos-Predict2.5 checkpoint 仍是有效的 20-step smoke 产物。

查询当前状态：

```bash
/root/.agents/skills/aihccli/scripts/aihc-agent.sh job get job-12x2lgpx0a6p \
  -p cce-pmm1yohj -q train -s

/root/.agents/skills/aihccli/scripts/aihc-agent.sh job get job-c469z4urkofj \
  -p cce-pmm1yohj -q train22 -s
```

### Cosmos3 历史 smoke 证据链（已失效）

下列记录只用于追踪旧训练，不得再表述为 canonical smoke passed：

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

这里的 `status=passed` 是旧脚本当时写入的原始记录；发现 timeline/prompt bug 后，handoff 判定已覆盖该旧状态。

修复版 20-step smoke bundle：

```text
AIHC name: ACWM_cosmos3_full50_mix41111_motusdata_rot6d20_future32_prompt_bs16_20step_smoke_train_20260721
script: docs/actionfollowing/aihc/run_cosmos3_mix4_20step_smoke.sh
job JSON: docs/actionfollowing/aihc/cosmos3_job_20step_smoke.json
first job ID: job-ploabmfspj7o
queue/final status: cce-pmm1yohj/train; Failed (2026-07-21 02:51:25 +08)
expected output (not created): /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-ploabmfspj7o
console: https://console.bce.baidu.com/aihc/#/job/detail/job-ploabmfspj7o?poolId=cce-pmm1yohj
```

该 job 的容器从 `Scheduled/Starting` 进入 `Failed`，pod `job-ploabmfspj7o-master-0` 的 node log 第一条也是唯一因果为：

```text
[FATAL] RoboTwin task_instruction root missing
```

失败发生在输出目录创建和数据审计之前，因此没有 loss/checkpoint，不能解释为训练失败或数据错误。修复后 bundle 会从最早期 bootstrap 起把 stdout/stderr 持久化到：

```text
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/bootstrap_logs/<job-id>.log
```

manifest/bootstrap 修复版（retry2）记录：

```text
AIHC name: ACWM_cosmos3_full50_mix41111_motusdata_rot6d20_future32_prompt_bs16_20step_smoke_retry2_20260721
job ID: job-3bxehdg8flpj
queue/final status: cce-pmm1yohj/train; Failed (2026-07-21 05:34:46 +08)
expected output: /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-3bxehdg8flpj
bootstrap log: /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/bootstrap_logs/job-3bxehdg8flpj.log
console: https://console.bce.baidu.com/aihc/#/job/detail/job-3bxehdg8flpj?poolId=cce-pmm1yohj
```

retry2 已加载本地 Qwen3-VL tokenizer 并进入真实 `clean`/`mix4` 数据审计；第一条因果 traceback 是 Hugging Face `datasets` 尝试在只读的共享 `HF_HOME` 下创建 parquet cache lock：

```text
OSError: [Errno 30] Read-only file system:
/mnt/dataset/public_data/cosmos3-cache/huggingface/datasets/...parquet....lock
```

这不是数据比例、timeline、prompt 或训练 loss 错误。retry3 保持共享 `HF_HOME` 只读用于离线模型/tokenizer，同时把 `HF_DATASETS_CACHE`、`XDG_CACHE_HOME` 和 `TMPDIR` 定向到每个 job 独立的持久化可写目录：

```text
<OUT_BASE>/runtime_cache/
```

提交 retry3 前已在同一服务器、同一真实 Motus-aligned source 上用该 cache 成功构造 `LeRobotDatasetMetadata`（30 FPS、50 episodes），并通过远端 ruff、12/12 loader tests、shell/JSON、bundle validator、structured TOML dryrun 与 staging SHA256 校验。对应 PR commit 为 `7290246`。

writable-cache 修复版（retry3）重提记录：

```text
AIHC name: ACWM_cosmos3_full50_mix41111_motusdata_rot6d20_future32_prompt_bs16_20step_smoke_retry3_20260721
job ID: job-12x2lgpx0a6p
queue/status: cce-pmm1yohj/train; initially Created / pod Pending (2026-07-21 05:40:58 +08), finally Succeeded (2026-07-21 06:11:47 +08)
expected output: /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p
bootstrap log: /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/bootstrap_logs/job-12x2lgpx0a6p.log
console: https://console.bce.baidu.com/aihc/#/job/detail/job-12x2lgpx0a6p?poolId=cce-pmm1yohj
```

训练前会强制检查 32 action timestamps、33 camera timestamps、五类新 counts、五类各 50 tasks、全部 350 个 LeRobot task metadata 与固定版本的 RoboTwin 官方 `full_description` 一致，并记录命中的 parquet prompt 列，再对每个 family 做真实 decode probe。训练结束后还会硬检查 rank-0 optimizer steps 恰好为 1..20、loss 全部 finite、`iter_000000020` DCP 与 latest marker，之后才写 `SMOKE_RESULT.txt`。

retry3 最终状态与持久化证据（2026-07-21 06:11 +08 复核）：

```text
AIHC/pod final status: Succeeded
queue: cce-pmm1yohj/train
data root: /mnt/dataset/public_data/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train

Cosmos3 current1+future32 effective counts:
clean=472622
perturbed=250000
random_feasible=1345000
counterfactual_replay=472145
exploration=120821

family task counts: all five families = 50
decoded action shape: [32,20]
decoded video shape: [3,33,720,640]
views: cam_high, cam_left_wrist, cam_right_wrist
prompt source: RoboTwin full_description
prompt source commit: c3ddfa8b97d5519efa828b075999bd0006778e5e
timeline: current1+future32

100k observed sampler:
clean=0.49995
perturbed=0.12500
random_feasible=0.12500
counterfactual_replay=0.12503
exploration=0.12502
max_abs_error=0.00005

optimizer steps: exactly 1..20
effective global batch: 16 (per-rank 2 x 8 GPUs)
rank-0 final loss: 0.1316 (finite)
trainer terminal message: Done with training.
checkpoint size: 137G, 36 DCP files
latest_checkpoint.txt: iter_000000020
SMOKE_RESULT.txt: status=passed
```

本次成功产物：

```text
DATA_AUDIT.log:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p/DATA_AUDIT.log

TRAIN_AUDIT.json:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p/TRAIN_AUDIT.json

SMOKE_RESULT.txt:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p/SMOKE_RESULT.txt

checkpoint:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p/bs16/cosmos3_actionfollowing/forward_dynamics_mix4/cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_20step_smoke/checkpoints/iter_000000020
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
effective counts: clean=475122, perturbed=250000, random_feasible=1350000,
                  counterfactual_replay=474645, exploration=121071
10k sampler audit: max_abs_error=0.0003
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
  /mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_job-12x2lgpx0a6p \
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

Cosmos3 `mix4` 的最终 AIHC 40k bundle 已由 canonical retry3 smoke 派生、验证并提交；Cosmos3 `clean` bundle 已按同一数据、future32、prompt 和 batch 合同补齐，并使用新的 5000-step checkpoint 周期。已经运行的 `job-3cmb7l4p44jw` 启动时读取的是 10000-step 周期，不能热修改，也不会为改变保存频率而重启。提交任一新任务前必须：

1. 从各模型最新有效 smoke bundle 派生四条独立 job JSON，名称显式包含模型、`clean/mix4`、`rot6d20`、`bs16`、`40k`；Cosmos3 必须先通过 `future32_prompt` 修复版 smoke。
2. 保持目标队列已验证的 8x A800 资源模板：`train/train22` 使用 CPU 123、memory 975Gi；`train21` 使用 CPU 122、memory 1960Gi；均保留 RDMA 1、shared memory 0Gi 与持久化 `pfs-Zx30ll` 挂载。
3. 重新跑 syntax、unit test、配置 compose、真实数据 audit 和 bundle validator。
4. 为每条任务使用独立持久化 output root，禁止覆盖 smoke 或其它协议结果。
5. 向用户报告四条 job 的精确名称、命令、镜像、资源、挂载和输出路径。
6. 只有获得单独明确的“提交 40k”授权后，才能执行 `job create`。

### 四条正式任务

下表同时区分已创建任务和待准备任务：

| 模型 | 协议 | 建议 job name 模板 | 建议持久化 output root | 当前状态 |
|---|---|---|---|---|
| Cosmos3-Nano | clean | `ACWM_cosmos3_full50_clean_motusdata_rot6d20_future32_prompt_bs16_40000step_ckpt5k_20260721` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos3/clean/train_40000_20260721_motusdata_future32_prompt_<job-id>` | bundle 已生成，待提交 |
| Cosmos3-Nano | mix4 | `ACWM_cosmos3_full50_mix41111_motusdata_rot6d20_future32_prompt_bs16_40000step_20260721` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw` | `job-3cmb7l4p44jw`，`train21`，Running |
| Cosmos-Predict2.5 | clean | `ACWM_cosmos25_full50_clean_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/clean/train_40k_<date>` | 未生成 job JSON、未提交 |
| Cosmos-Predict2.5 | mix4 | `ACWM_cosmos25_full50_mix4_rot6d20_bs16_40k_train22_<date>` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/train_40k_<date>` | 未生成 job JSON、未提交 |

### 40k bundle 不能直接照抄 repo-level launcher

公开仓库中的 clean/mix4 launcher 表达模型配置意图，但最终 AIHC bundle 必须从同模型最新有效 smoke 脚本派生。Cosmos3 不得从语义已失效的 retry12 派生，必须以修复版 `future32_prompt` smoke 成功后的 bundle 为准。原因：

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
7. 运行 `validate_job_bundle.py --steps 40000`，clean 5K bundle 额外传入 `--checkpoint-save-iter 5000`，检查 job name、资源、挂载、命令、持久化输出和 checkpoint 配置。
8. 把四条任务的精确 JSON 摘要展示给用户并取得新的明确 launch 授权。
9. 执行 `job create` 后分别记录 job ID；Created/Pending 不等于 Running，不报告无依据 ETA。
10. 训练期间检查 optimizer step、finite loss 和 persistent log；不要只看控制台末尾。

### AIHC 资源合同

每条正式任务使用目标队列的真实机器模板；不能把 `train` 的 1TB 数值直接提交到 `train21` 的 2TB 节点：

```text
pool:                    cce-pmm1yohj
queue train/train22:     CPU 123, memory 975 GiB
queue train21:           CPU 122, memory 1960 GiB
replicas:                1
GPU:                     8 x baidu.com/a800_80g_cgpu
RDMA:                    1
shared memory:           0 GiB
effective batch:         16
checkpoint PFS:          pfs-Zx30ll
new 40k save interval:   5000 optimizer steps
```

首次使用 1TB 模板提交 `train21` 时，控制面返回 `ResourceTemplateMismatch`，且没有创建 job。随后从该队列真实成功的 8-GPU job 读取 2TB 模板、更新 validator 并完整复验后，才创建 `job-3cmb7l4p44jw`。

### Cosmos3 mix4 40k 运行记录

```text
PR commit containing the launch bundle: 73e7fe7
job ID: job-3cmb7l4p44jw
queue: cce-pmm1yohj/train21
AIHC created: 2026-07-21 06:26:41 +08
AIHC running: 2026-07-21 06:26:58 +08
pod: job-3cmb7l4p44jw-master-0
node: 10.40.0.202
pod IP at startup: 172.17.161.181
```

启动 bundle：

```text
docs/actionfollowing/aihc/run_cosmos3_mix4_40000.sh
docs/actionfollowing/aihc/cosmos3_job_40000_mix4.json
```

持久化输出与 bootstrap 日志：

```text
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/bootstrap_logs/job-3cmb7l4p44jw.log
```

启动时已核验 8 GPUs、本地 Qwen3-VL tokenizer `vocab_size=151643`、独立 writable datasets cache，以及 `optimizer_steps=40000 / checkpoint_save_iter=10000 / scheduler_cycle=40000 / warmup_steps=1000`。真实 clean/mix4 审计与五类 decode 均已再次通过，训练已进入 real optimizer steps；step 10000 的 137G DCP 与 latest marker 已成功落盘。该运行中的保存周期保持 10000，不因后续仓库将新任务改为 5000 而重启。`Running` 不等于 40k 已成功。

batch 8 不是排队紧张时的替代方案。只有 batch-16 运行日志出现真实 CUDA OOM，且原因不能通过明显配置错误修正时，才允许另建 batch-8 retry，并在 job name/handoff 中明确记录。

## 已修复的关键问题

- Cosmos3 Hugging Face checkpoint 已转换并持久化为 DCP；官方 Wan2.2 VAE 与本地 Qwen3-VL tokenizer 已固定。
- Cosmos3 注册了 ActionFollowing experiment，修复了并发 metadata loader 的 `tqdm` lock。
- Cosmos3 action/observation timeline 已改为 `A[t:t+32]` 对应真实 `O[t:t+33]`，删除复制尾帧，并排除每条 trajectory 的无后继 terminal window。
- Cosmos3 text conditioning 已从空字符串改为 RoboTwin 每任务 `full_description`，且 smoke 会与 canonical JSON 逐样本比对。
- Cosmos3 prompt 审计不再依赖未挂载的个人 RoboTwin checkout；50-task manifest 固定到 RoboTwin 官方 commit，并兼容真实 LeRobot v2 `tasks.parquet` 的 `__index_level_0__` 文本列。
- Cosmos3 bootstrap 从首条命令开始写 PFS 日志，挂载、入口或 prompt manifest 等早期错误不再只依赖 AIHC 聚合日志。
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
| 最后一个 action 监督静止尾帧 | camera delta 只有 32 项，loader 复制第 32 帧 | camera delta 改为 33 项并删除尾帧复制；trajectory counts 每 episode 减 1 | 真实 sample 有 `O[t+32]`，最后两帧来自不同时间戳 |
| Cosmos3 text prompt 为空 | adapter 固定 `ai_caption=""` | 读取 LeRobot `sample["task"]`，并要求其等于 RoboTwin `full_description` | clean/mix4 五类 decode probe 均逐条比对 canonical JSON |
| Cosmos3 bootstrap 立即退出 | job 容器没有挂载个人 `RoboTwin/description/task_instruction` 目录 | 把官方 50-task `full_description` 固定成仓库内 manifest；从 bootstrap 起持久化 stderr/stdout | 官方 50 JSON、真实 350 source prompt、bundle validator、PFS bootstrap log |
| Cosmos2.5 torchvision 输入错误 | NumPy HWC/THWC 直接进入 Torch transform | contiguous NumPy -> Torch TCHW | `[3,33,256,320]` head video |
| Cosmos2.5 tokenizer 尝试远端下载 | VAE/Reason1 路径未完全本地化 | 固定本地 VAE、Reason1、processor overlay | 断网/无 Hub fallback 下能加载 |
| Hydra 报未知 dataset kwargs | 父 RoboTwin 参数残留 | 只接受并硬校验 canonical 三个值 | 错误值必须 fail loudly |
| Hydra 出现父 `dataloaders` map | joint-pretraining merge 残留 | 专用 builder 只删除该枚举 residue | final target/keys compose 检查 |
| counts 硬断言错误 | 文档全量资产 counts 被误当成 train effective counts，或把 action-chunk counts 误用于需要 33 observation 的 Cosmos3 | 按模型使用 train counts；Cosmos3 再排除 terminal start | Cosmos3 与 Cosmos2.5 分别打印各自 model-specific counts |
| rank0 后出现 NCCL abort/SIGTERM | 其它 rank 已先抛第一因果 traceback | 从完整持久化日志找最早 traceback | 不把 NCCL 尾声当 root cause |

诊断新失败时，必须先保存：job ID、pod、完整持久化 log、第一条 traceback、目标 commit、job JSON SHA、run-script SHA。没有这些信息时不要直接改多个层级，也不要删除失败证据。

## AIHC 监控与诊断命令

状态与 pod：

```bash
AIHC=/root/.agents/skills/aihccli/scripts/aihc-agent.sh
QUEUE=train

$AIHC job get <job-id> -p cce-pmm1yohj -q "$QUEUE" -s
$AIHC job get <job-id> -p cce-pmm1yohj -q "$QUEUE" --pods
$AIHC pod list <job-id> -p cce-pmm1yohj -q "$QUEUE"
```

日志：

```bash
$AIHC job logs <job-id> -p cce-pmm1yohj -q "$QUEUE"
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
- [x] 查询两条 mix4 smoke job 和持久化 marker，不能只看 scheduler `Succeeded`。
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
