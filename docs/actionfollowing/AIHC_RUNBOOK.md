# Cosmos ActionFollowing AIHC runbook

## Contents

1. Fixed experiment contract
2. Paths and assets
3. Launch sequence
4. Model-specific configuration
5. Verified failure map
6. Artifact verification
7. Cleanup sequence
8. Known-good smoke reference

## 1. Fixed experiment contract

| Item | Value |
|---|---|
| Data protocols | `clean`, `mix4` |
| Tasks | 50 RoboTwin tasks |
| Action | 32-step chunk, Rot6D20 (`[32,20]`) |
| mix4 | clean 50%; four enhanced families 12.5% each |
| Steps | 20 optimizer-step smoke; 40,000 only after approval |
| Batch | global 16; global 8 only on detected CUDA OOM |
| Cosmos3 views | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Predict2.5 views | `cam_high` only |
| Pool / queue | `cce-pmm1yohj` / `train22` |
| Hardware | 8x A800 80GB, CPU 123, memory 970 Gi, RDMA 1, shm 120 Gi |

Effective train counts are `clean=475122`, `perturbed=250000`, `random_feasible=1350000`, `counterfactual_replay=474645`, `exploration=121071`.

## 2. Paths and assets

| Purpose | Path |
|---|---|
| Local workspace | `/Users/user/HumanoidX-DEV/ACWM` |
| Local launch bundle | `aihc/cosmos_baseline_smoke_20260718` |
| Remote job bundle | `/mnt/gyc/Action-Following/jobs/20260718` |
| Cosmos3 repo | `/mnt/gyc/cosmos-framework` |
| Predict2.5 repo | `/mnt/gyc/cosmos-predict2.5` |
| Canonical data | `/mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train` |
| Persistent output | `/mnt/gyc_ckp/Action-Following/outputs` |
| Cosmos3 DCP | `/mnt/gyc_ckp/models/Cosmos3-Nano-DCP-411f42a8fdfb` |
| Cosmos3 VAE | `/mnt/public_ckp/cosmos3-cache/wan22_vae/Wan2.2_VAE.pth` |
| Cosmos3 tokenizer | `/mnt/public_ckp/cosmos3-cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/text_tokenizer` |
| Predict2.5 checkpoint | `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt` |
| Predict2.5 VAE tokenizer | `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Predict2.5-2B/tokenizer.pth` |
| Predict2.5 Reason1 | `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/models/Cosmos-Reason1-7B` |

AIHC containers mount public storage at `/mnt/dataset/public_data`; startup scripts create and verify `/mnt/public_ckp` aliases. Do not hard-code a path that is absent inside the container.

## 3. Launch sequence

1. Run `bash -n`, JSON parsing, lint, and relevant unit tests locally and remotely.
2. Cosmos3: instantiate the local Qwen tokenizer and run the exact structured TOML `--dryrun` with all production environment variables.
3. Predict2.5: compose the Hydra config and assert the final dataloader target/keys, canonical compatibility values, local VAE path, and local Reason1 processor path.
4. Run `scripts/validate_job_bundle.py`.
5. Query queue pressure; `free=0` means submit may remain `Created`, not that submission failed.
6. Create jobs with:

   ```bash
   /root/.agents/skills/aihccli/scripts/aihc-agent.sh job create \
     -p cce-pmm1yohj -q train22 -f <job.json>
   ```

7. Inspect status/pods/logs without stopping prior jobs.

## 4. Model-specific configuration

### Cosmos3-Nano

- Use native three-view video and `Cosmos3-Nano` DCP.
- Register `action_forward_dynamics_actionfollowing_nano` explicitly from `cosmos_framework/configs/base/config.py`.
- Pre-initialize the Hugging Face `tqdm` lock before concurrent metadata discovery.
- Resolve the VLM tokenizer to the Cosmos3-Nano snapshot's local `text_tokenizer`; do not rely on offline resolution of `Qwen/Qwen3-VL-8B-Instruct`.
- Override the real launch tail with `trainer.max_iter=20`, `trainer.logging_iter=1`, `checkpoint.save_iter=20`, per-rank batch 2, scheduler cycle 20, and warmup 1.

### Cosmos-Predict2.5

- Use the single head view because the inherited action-conditioned path is single-view.
- Convert decoded video from NumPy HWC/T HWC into contiguous Torch TCHW before torchvision preprocessing.
- Use the shared local Wan VAE tokenizer and local Reason1 weights/processor overlay.
- Accept only canonical inherited RoboTwin compatibility values: `gripper_rescale_factor=1`, `num_action_per_chunk=32`, `fps_downsample_ratio=1`.
- Use the dedicated ActionFollowing dataloader builder to discard only the verified stale parent `dataloaders` map. Forward every other key to PyTorch so new incompatibilities fail loudly.

## 5. Verified failure map

| Symptom | Cause | Required fix / proof |
|---|---|---|
| Missing repo or checkpoint path | AIHC PFS mount differs from login node | Verify aliases and datasource mounts inside the pod |
| Missing `/mnt/public_ckp` | Alias not created in container | Create only after target mount exists; compare `readlink -f` |
| VAE download/fallback failure | Wrong or partial tokenizer asset | Pin official local VAE/tokenizer and preload it |
| Incorrect mix4 counts | Full asset counts confused with train split | Assert the five effective train counts above |
| `tqdm._lock` missing | Nested concurrent metadata loaders race on lazy lock | Call `tqdm.get_lock()` before the outer thread pool |
| Hydra experiment missing | New Cosmos3 experiment module not imported | Add explicit import and run structured TOML dryrun |
| Qwen vocab path is `None` | Offline model ID resolves to incomplete cache | Point to Cosmos3-Nano local `text_tokenizer`; assert vocab 151643 |
| Enhanced video is `FileNotFound` although the split entry exists | Train-split MP4 is an absolute symlink to the absent `/mnt/dataset/csx_workspace/Ideas/data` prefix | Resolve only broken symlink targets through `AFD_VIDEO_SYMLINK_PREFIX_REMAP=/mnt/dataset/csx_workspace/Ideas/data=/mnt/dataset/sixiangchen_workspace/Ideas/data`; preserve the split parquet/actions/counts |
| Cosmos3 timestamp tolerance assertion on exploration video | Some enhanced MP4s are 10 FPS while the action timeline remains 30 Hz | Keep actions at 30 Hz and use `VIDEO_TIMESTAMP_TOLERANCE_S=0.051` so the nearest repeated 10 FPS frame is accepted; do not retime labels |
| NumPy/torch video error | HWC NumPy passed into Torch transforms | Convert to contiguous TCHW tensor first |
| Unexpected `gripper_rescale_factor` | Parent RoboTwin dataset keys survive Hydra merge | Accept and hard-check only canonical values |
| Unexpected `dataloaders` | Parent joint-pretraining map survives top-level merge | Dedicated builder discards this one enumerated residue |
| Reason1 download failure | Text encoder tries remote Hub | Pin local Reason1 weights and processor overlay |

Always use the first causal traceback. NCCL abort/SIGTERM lines after one rank fails are consequences, not the root cause.

## 6. Artifact verification

For each successful model, verify:

1. AIHC job and pod report terminal success.
2. Logs contain 20 optimizer-step records and at least one finite loss.
3. No batch-8 fallback occurred unless the batch-16 log contains a genuine CUDA OOM signature.
4. Data audit records the five expected family counts and target/observed ratios.
5. Shapes and view lists match the fixed contract.
6. `latest_checkpoint.txt` points to an existing checkpoint on `/mnt/gyc_ckp`.
7. `SMOKE_RESULT.txt` states `status=passed`, model, protocol, steps, effective global batch, and checkpoint.

## 7. Cleanup sequence

After both latest jobs pass and cleanup is authorized:

1. List all matching Cosmos ActionFollowing smoke jobs in `train22`.
2. Select only terminal `Failed` jobs.
3. Re-query each ID and show the deletion manifest.
4. Delete one exact ID at a time:

   ```bash
   /root/.agents/skills/aihccli/scripts/aihc-agent.sh job delete <job-id> \
     -p cce-pmm1yohj -q train22
   ```

5. Re-list to prove failed jobs are gone and successful jobs remain.

Do not delete persistent outputs until separately authorized.

## 8. Known-good smoke reference

Verified on 2026-07-18 in `cce-pmm1yohj/train22`:

| Model | Successful job | Persistent output | Final proof |
|---|---|---|---|
| Cosmos3-Nano | `job-s4qigpgr3lky` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260718_retry12` | batch 16, 20 steps, rank-0 final loss 0.2027, 137 GB DCP |
| Cosmos-Predict2.5-2B | `job-c469z4urkofj` | `/mnt/gyc_ckp/Action-Following/outputs/cosmos_predict25/mix4/smoke_20260718_retry12` | batch 16, 20 steps, final loss 0.1634, 20 GB DCP |

Both roots contain `SMOKE_RESULT.txt`; each latest marker points to `iter_000000020`. Treat these as known-good references, not as substitutes for live verification on a new run.
