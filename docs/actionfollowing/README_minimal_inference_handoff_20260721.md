# Cosmos3 ActionFollowing 最小推理交接（2026-07-21）

本文提供 Cosmos3-Nano ActionFollowing 模型的最小、可审计 forward-dynamics 推理。结构参考 [Ctrl-World 单任务模型交接](https://github.com/Ricardo520nono/ctrl-world-train-wjx/blob/dev-csx-codex/code/scripts_daily/20260719/README_ctrlworld_single_task_handoff_20260719.md)，重点固定 checkpoint、训练配置、action space、归一化、相机排布、输入样本、运行命令和输出证据。

## 当前证明状态

- `mix4` 训练 job `job-3cmb7l4p44jw` 已执行到 40,000 optimizer steps，final rank0 loss `0.0915`（finite），`iter_000040000` 与 `latest_checkpoint.txt=iter_000040000` 均存在。AIHC 状态为 `Failed`，原因是训练后严格审计只解析到 39,999 条 rank0 step 日志、唯一缺少 step 51；因此没有生成 `TRAIN_AUDIT.json`/`TRAIN_RESULT.txt`。这是“训练和 checkpoint 完成、训练后审计失败”，不能写成 job 正式验收通过。
- `clean` 训练 job `job-ogwrcxzuaokw` 同样已执行到 40,000 optimizer steps，final rank0 loss `0.0567`（finite），5k 到 40k 的 checkpoint、`iter_000040000` 和 latest marker 均存在。AIHC 状态同样为 `Failed`，训练后审计也只缺 step 51 的日志行，未生成最终审计文件。
- 最小推理使用 `mix4/iter_000030000`，retry3 job `job-9i1cciznft0a` 已 `Succeeded` 并完成独立验收：DCP→HF 导出成功，输入/输出均为 finite physical Rot6D20 `[32,20]`，视频可解码，`sample_outputs.json` 为 success，`INFERENCE_RESULT.txt` 为 passed。
- `iter_000030000` 的推理结果仍明确属于 30k 中间 checkpoint；最终 40k checkpoint 已存在，但本文没有把 30k 推理冒充 40k 推理。
- AIHC `train` 是 8×A800 整机模板，但脚本只运行一个 inference 进程；这不表示模型推理本身要求 8 卡。

## Checkpoint 与训练配置

### mix4 checkpoint（最小推理默认使用 30k，训练最终为 40k）

```text
run root:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw/bs16/cosmos3_actionfollowing/forward_dynamics_mix4/cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_40000step

DCP:
.../checkpoints/iter_000030000

final DCP:
.../checkpoints/iter_000040000

frozen training config:
.../config.yaml
```

训练 job 为 `job-3cmb7l4p44jw`，AIHC `cce-pmm1yohj/train21`，global batch 16，目标 40,000 optimizer steps，checkpoint 每 10,000 steps。

### clean 最终 checkpoint

```text
run root:
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/clean/train_40000_20260721_motusdata_future32_prompt_job-ogwrcxzuaokw/bs16/cosmos3_actionfollowing/forward_dynamics_clean/cosmos3_nano_afd_full50_clean_rot6d20_a32_future32_prompt_bs16_40000step

DCP:
.../checkpoints/iter_000040000

frozen training config:
.../config.yaml
```

训练 job 为 `job-ogwrcxzuaokw`，AIHC `cce-pmm1yohj/train21`，global batch 16，目标 40,000 optimizer steps，checkpoint 每 5,000 steps。

## 最小推理的输入输出

固定输入为 ActionFollowingData clean 的 `place_burger_fries` 样本：

```text
current observation: 1 frame, 3 RGB cameras
expert action:        [32,20]
text:                 RoboTwin full_description
ground truth:         33 observations, O[t]..O[t+32]
model output:         32 future frames conditioned on O[t] and A[t:t+32]
```

输入资产由 `examples/actionfollowing/prepare_minimal_inference.py` 一次生成：

```text
current_tshape.png
observation.images.cam_high.png
observation.images.cam_left_wrist.png
observation.images.cam_right_wrist.png
actions_physical_rot6d20.json
full_description.txt
ground_truth_tshape_33frames.mp4
cosmos3_forward_dynamics.json
input_metadata.json
```

### 相机排布

原始相机读取顺序固定为：

```text
cam_high, cam_left_wrist, cam_right_wrist
```

Cosmos3 loader 将三视角拼成一个 T-shape：

```text
+---------------------------+
|          cam_high         |
+-------------+-------------+
| left_wrist  | right_wrist |
+-------------+-------------+
```

head 保持原分辨率；两个 wrist 各缩到 head 的一半高、一半宽。推理必须读取 `current_tshape.png`，不能把三个相机文件当成三个独立 Cosmos vision item。

## Action space：为什么不能照搬 policy normalized action

ActionFollowing 的 canonical action 是 robot base frame 下的 physical delta-EE Rot6D20：

```text
[left_dx, left_dy, left_dz,
 left_r00, left_r10, left_r20, left_r01, left_r11, left_r21,
 left_gripper,
 right_dx, right_dy, right_dz,
 right_r00, right_r10, right_r20, right_r01, right_r11, right_r21,
 right_gripper]
```

```text
rot6d = concat(R[:,0], R[:,1])
       = [r00,r10,r20,r01,r11,r21]
```

坐标轴为 `+X robot forward / +Y robot left / +Z robot up`。这里的 `R` 表示 delta rotation 对应的旋转矩阵，不是 Euler、quaternion、camera-frame pose 或 joint action。

Cosmos3 这条 adapter 的 `action_normalization=None`：推理 JSON 中的 `[32,20]` action 是 physical Rot6D20，loader 只把 20D 零填充到模型内部 `max_action_dim`，不做 q01/q99 或 policy stats 归一化。因此：

- 可以直接使用同一 canonical ActionFollowing LeRobot 样本的 physical action；
- 不可以直接输入 QwenOFT/StarVLA 的 normalized action；必须先用 policy stats 反归一化回 physical Rot6D20；
- 不可以套用 Ctrl-World 或 LingBot 的 q01/q99；Cosmos 的训练入口没有使用它们；
- 不能把 14D Euler action 仅靠 padding 伪装成 20D。

## 运行命令

```bash
cd /mnt/gyc/cosmos-framework

export DCP_RUN_ROOT=/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw/bs16/cosmos3_actionfollowing/forward_dynamics_mix4/cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_40000step
export DCP_ITER=iter_000030000

bash examples/actionfollowing/run_cosmos3_minimal_inference.sh
```

脚本顺序：

1. 从 canonical symlink-free Rot6D20 root 解码一条真实 clean 样本；
2. 核验 video `[3,33,H,W]`、action `[32,20]`、finite action、非空 full_description；
3. 使用该 run 的 `config.yaml` 把 DCP 导出为独立 HF safetensors；
4. 运行 Cosmos 官方 `forward_dynamics` inference，10 denoising steps，seed `20260721`；
5. 检查输出 MP4 和 `sample_outputs.json`，最后写 `INFERENCE_RESULT.txt`。

## 输出目录和验收

默认输出在 checkpoint run 内，避免丢失 checkpoint provenance：

```text
<DCP_RUN_ROOT>/handoff_inference/<DCP_ITER>_place_burger_fries_clean0_steps10/
  hf_model/
  input/
  output/actionfollowing_place_burger_fries_clean0/
  minimal_inference.log
  INFERENCE_RESULT.txt
```

本次 AIHC job 的固定输出根为：

```text
/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw/bs16/cosmos3_actionfollowing/forward_dynamics_mix4/cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_40000step/handoff_inference/iter_000030000_place_burger_fries_clean0_steps10_aihc_train_20260721_job-9i1cciznft0a
```

本次 `job-9i1cciznft0a` 的验收结果：

- `hf_model/checkpoint.json` 和 safetensors 已由 `iter_000030000` DCP 导出；
- 输入为 `place_burger_fries/clean/episode_0`，physical Rot6D20 action `[32,20]`、finite、无归一化；
- 输出 action 同为 finite `[32,20]`，`sample_outputs.json` 状态为 success；
- `vision.mp4` 为 H.264、224×256、33 帧、503,316 bytes，可解码；
- 相机排布为 head-top、wrists-bottom 的 T-shape；
- `INFERENCE_RESULT.txt` 为 `status=passed`，源码 commit 为 `26dd6adecbeb9f5cb20a1661f0ddb681916b108d`；
- 该结果仅证明 30k checkpoint 的最小推理链路，不等价于 40k checkpoint 的推理验收。

## 代码位置

```text
examples/actionfollowing/prepare_minimal_inference.py
examples/actionfollowing/run_cosmos3_minimal_inference.sh
tests/test_actionfollowing_minimal_inference.py
docs/actionfollowing/README_minimal_inference_handoff_20260721.md
```

公开仓库与分支：

```text
https://github.com/EasonAI-5589/cosmos-framework-actionfollowing
branch: agent/actionfollowing-rot6d20-baseline
PR: https://github.com/EasonAI-5589/cosmos-framework-actionfollowing/pull/1
```
