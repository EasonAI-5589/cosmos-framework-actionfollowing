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

两个 fork 只接收本文列出的 ActionFollowing-owned files、复现文档和对应 smoke bundle，不向 NVIDIA upstream 自动创建 PR。

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

## 已修复的关键问题

- Cosmos3 Hugging Face checkpoint 已转换并持久化为 DCP；官方 Wan2.2 VAE 与本地 Qwen3-VL tokenizer 已固定。
- Cosmos3 注册了 ActionFollowing experiment，修复了并发 metadata loader 的 `tqdm` lock。
- enhanced split 中坏绝对软链通过受限 prefix remap 解析，不修改 canonical split。
- exploration 视频部分为 10 FPS，但 action label 保持 30 Hz；timestamp tolerance 固定为 `0.051s`，允许最近重复帧，不重定时 action。
- Cosmos2.5 loader 已把 NumPy HWC/THWC 转成 contiguous Torch TCHW。
- Cosmos2.5 固定使用本地官方 VAE、Reason1 权重与 processor overlay，避免运行时下载。
- Cosmos2.5 只接受 canonical `gripper_rescale_factor=1`、`num_action_per_chunk=32`、`fps_downsample_ratio=1`。
- mix4 断言使用真实 train effective counts，不得误用数据文档中的全量资产 counts。

## 接手检查清单

- [ ] 确认工作目录与目标 Git/fork，不要把两个 dirty 官方 repo 的无关改动带入提交。
- [ ] 重新核对本地 adapter 与服务器运行文件 SHA256。
- [ ] 读取 `action_following_data_assets.md`，保持 50 tasks、Rot6D20 与 mix4 chunk-sample 采样协议。
- [ ] 查询两条 smoke job 和持久化 marker，不能只看 scheduler `Succeeded`。
- [ ] 为 `Cosmos3 clean`、`Cosmos3 mix4`、`Cosmos2.5 clean`、`Cosmos2.5 mix4` 各自生成 40k bundle。
- [ ] 提交前展示精确资源与输出路径，并重新取得 40k launch 授权。
- [ ] 40k 运行后分别核验 finite loss、step 40000、latest marker、checkpoint 和最终数据 audit。

## 本地运行手册

```text
/Users/user/.codex/skills/run-cosmos-actionfollowing/SKILL.md
/Users/user/.codex/skills/run-cosmos-actionfollowing/references/runbook.md
/Users/user/.codex/skills/run-cosmos-actionfollowing/references/action_following_data_assets.md
```
