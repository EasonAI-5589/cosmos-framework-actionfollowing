#!/usr/bin/env bash
set -Eeuo pipefail

RUN_TAG="${AIHC_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
BOOTSTRAP_LOG_DIR=/mnt/dataset/csx_ckp/Action-Following/outputs/cosmos3/mix4/bootstrap_logs
mkdir -p "$BOOTSTRAP_LOG_DIR"
BOOTSTRAP_LOG="$BOOTSTRAP_LOG_DIR/${RUN_TAG}.log"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
trap 'rc=$?; printf "[BOOTSTRAP_ERROR] rc=%s line=%s command=%q\n" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND"; exit "$rc"' ERR
echo "[BOOTSTRAP] job_id=$RUN_TAG log=$BOOTSTRAP_LOG"

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

ensure_mount_alias() {
  local alias_path="$1"
  local target_path="$2"
  [[ -d "$target_path" ]] || die "missing AIHC mount: $target_path"
  if [[ ! -e "$alias_path" ]]; then
    ln -s "$target_path" "$alias_path"
  fi
  [[ "$(readlink -f "$alias_path")" == "$(readlink -f "$target_path")" ]] || \
    die "$alias_path does not resolve to $target_path"
}

ensure_mount_alias /mnt/gyc /mnt/dataset/csx_workspace
ensure_mount_alias /mnt/gyc_ckp /mnt/dataset/csx_ckp
ensure_mount_alias /mnt/public_ckp /mnt/dataset/public_data

REPO=/mnt/gyc/cosmos-framework
# Use the exact symlink-free LeRobot Rot6D train root used by the Motus
# ACWM-Motus_mix41111_40000 runs.  Cosmos keeps its own current1+future32
# indexing on top of these shared assets.
AFD_ROOT=/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train
WAN_VAE_PATH=/mnt/dataset/public_data/cosmos3-cache/wan22_vae/Wan2.2_VAE.pth
BASE_CHECKPOINT_PATH=/mnt/gyc_ckp/models/Cosmos3-Nano-DCP-411f42a8fdfb
HF_HOME=/mnt/dataset/public_data/cosmos3-cache/huggingface
COSMOS3_TOKENIZER_PATH="$HF_HOME/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa/text_tokenizer"
OUT_BASE="/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/smoke_20260721_motusdata_future32_prompt_${RUN_TAG}"
ROBOTWIN_FULL_DESCRIPTION_MANIFEST="$REPO/docs/actionfollowing/assets/robotwin_50_full_descriptions.json"

[[ -x "$REPO/.venv/bin/python" ]] || die "Cosmos3 venv missing"
[[ -f "$WAN_VAE_PATH" ]] || die "Wan2.2 VAE missing: $WAN_VAE_PATH"
[[ -f "$COSMOS3_TOKENIZER_PATH/vocab.json" ]] || die "Cosmos3 tokenizer vocab missing"
[[ -f "$COSMOS3_TOKENIZER_PATH/merges.txt" ]] || die "Cosmos3 tokenizer merges missing"
[[ -f "$ROBOTWIN_FULL_DESCRIPTION_MANIFEST" ]] || \
  die "RoboTwin full_description manifest missing: $ROBOTWIN_FULL_DESCRIPTION_MANIFEST"
[[ -f "$AFD_ROOT/demo_clean_zed2i_visible/turn_switch/meta/info.json" ]] || die "clean data missing"
[[ -f "$AFD_ROOT/exploration/tasks/turn_switch/meta/info.json" ]] || die "mix4 exploration data missing"

export AFD_ROOT WAN_VAE_PATH BASE_CHECKPOINT_PATH HF_HOME COSMOS3_TOKENIZER_PATH OUT_BASE
export ROBOTWIN_FULL_DESCRIPTION_MANIFEST
unset AFD_VIDEO_FALLBACK_ROOTS AFD_VIDEO_SYMLINK_PREFIX_REMAP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export OMP_NUM_THREADS=8

if [[ -e "$OUT_BASE" ]] && [[ -n "$(find "$OUT_BASE" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "refusing to reuse non-empty smoke output: $OUT_BASE"
fi
mkdir -p "$OUT_BASE"

echo "[PRECHECK] job=${AIHC_JOB_NAME:-UNKNOWN} gpus=${TRAINING_CARD_SIZE:-UNKNOWN}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$REPO/.venv/bin/python" - <<'PY'
import os

from cosmos_framework.model.generator.tokenizers.tokenization_qwen2 import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained(
    os.environ["COSMOS3_TOKENIZER_PATH"],
    local_files_only=True,
)
assert tokenizer.vocab_size == 151643, tokenizer.vocab_size
print(f"[TOKENIZER] local Qwen3-VL tokenizer vocab_size={tokenizer.vocab_size}")
PY

if [[ -e "$BASE_CHECKPOINT_PATH" && ! -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]]; then
  die "incomplete Cosmos3 DCP exists: $BASE_CHECKPOINT_PATH"
fi

if [[ ! -f "$BASE_CHECKPOINT_PATH/checkpoint.json" ]]; then
  mkdir -p "$(dirname "$BASE_CHECKPOINT_PATH")"
  DCP_TMP="${BASE_CHECKPOINT_PATH}.tmp.${AIHC_JOB_ID:-manual}.$$"
  echo "[DCP] converting cached nvidia/Cosmos3-Nano -> $DCP_TMP"
  cd "$REPO"
  "$REPO/.venv/bin/python" -m cosmos_framework.scripts.convert_model_to_dcp \
    -o "$DCP_TMP" \
    --checkpoint-path Cosmos3-Nano
  [[ -f "$DCP_TMP/checkpoint.json" ]] || die "DCP conversion did not write checkpoint.json"
  [[ -f "$DCP_TMP/model/.metadata" ]] || die "DCP conversion did not write model/.metadata"
  mv "$DCP_TMP" "$BASE_CHECKPOINT_PATH"
fi

echo "[DATA] auditing clean and mix4 on the real AIHC mount"
cd "$REPO"
"$REPO/.venv/bin/python" - <<'PY' | tee "$OUT_BASE/DATA_AUDIT.log"
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from cosmos_framework.data.generator.action.datasets.actionfollowing_lerobot_dataset import (
    ACTION_FEATURE,
    CAMERA_FEATURES,
    ROBOTWIN_50_TASKS,
    ActionFollowingLeRobotDataset,
)

expected = {
    "clean": 472_622,
    "perturbed": 250_000,
    "random_feasible": 1_345_000,
    "counterfactual_replay": 472_145,
    "exploration": 120_821,
}

prompt_manifest = json.loads(Path(os.environ["ROBOTWIN_FULL_DESCRIPTION_MANIFEST"]).read_text())
assert prompt_manifest["schema_version"] == 1, prompt_manifest
assert prompt_manifest["source_repository"] == "https://github.com/RoboTwin-Platform/RoboTwin"
assert prompt_manifest["source_commit"] == "c3ddfa8b97d5519efa828b075999bd0006778e5e"
canonical_prompts = prompt_manifest["full_descriptions"]
assert set(canonical_prompts) == set(ROBOTWIN_50_TASKS), (
    set(canonical_prompts),
    set(ROBOTWIN_50_TASKS),
)


def canonical_full_description(task_name: str) -> str:
    prompt = canonical_prompts[task_name]
    assert isinstance(prompt, str) and prompt.strip(), (task_name, prompt)
    return prompt.strip()


def metadata_full_description(row: dict) -> tuple[str, str]:
    for column in ("task", "__index_level_0__"):
        prompt = row.get(column)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip(), column
    raise AssertionError(f"tasks.parquet row has no full_description text column: {row}")


for protocol in ("clean", "mix4"):
    dataset = ActionFollowingLeRobotDataset(
        root=os.environ["AFD_ROOT"],
        protocol=protocol,
        audit_num_samples=100_000,
        audit_max_abs_error=0.02,
    )
    wanted = {"clean": expected["clean"]} if protocol == "clean" else expected
    assert dataset.family_effective_counts == wanted, (dataset.family_effective_counts, wanted)
    assert len(set(dataset._task_by_source)) == 50
    family_tasks = defaultdict(set)
    metadata_prompt_count = 0
    metadata_prompt_columns = set()
    for source_root, family, task_name in dataset._source_specs:
        family_tasks[family].add(task_name)
        rows = pq.read_table(Path(source_root) / "meta" / "tasks.parquet").to_pylist()
        assert len(rows) == 1, (source_root, len(rows))
        metadata_prompt, metadata_prompt_column = metadata_full_description(rows[0])
        metadata_prompt_columns.add(metadata_prompt_column)
        assert metadata_prompt == canonical_full_description(task_name), (
            source_root,
            metadata_prompt,
            canonical_full_description(task_name),
        )
        metadata_prompt_count += 1
    expected_tasks = set(ROBOTWIN_50_TASKS)
    assert all(tasks == expected_tasks for tasks in family_tasks.values()), family_tasks
    assert set(family_tasks) == set(wanted), family_tasks
    delta_timestamps = dataset._dataset_build_args[0]["delta_timestamps"]
    assert len(delta_timestamps[ACTION_FEATURE]) == 32
    assert all(len(delta_timestamps[feature]) == 33 for feature in CAMERA_FEATURES)
    assert delta_timestamps[ACTION_FEATURE][-1] < delta_timestamps[CAMERA_FEATURES[0]][-1]
    audit = dataset.audit_sampling(num_samples=100_000, seed=20260717)
    assert audit["max_abs_error"] <= 0.02, audit
    item = dataset[0]
    assert tuple(item["action"].shape) == (32, 20), item["action"].shape
    assert item["video"].shape[1] == 33, item["video"].shape
    assert item["ai_caption"] == canonical_full_description(item["task_name"])
    assert torch.isfinite(item["action"]).all()
    assert torch.isfinite(item["video"].float()).all()
    print(json.dumps({
        "protocol": protocol,
        "tasks": len(set(dataset._task_by_source)),
        "family_task_counts": {family: len(tasks) for family, tasks in sorted(family_tasks.items())},
        "metadata_prompt_count": metadata_prompt_count,
        "metadata_prompt_columns": sorted(metadata_prompt_columns),
        "effective_counts": dataset.family_effective_counts,
        "audit": audit,
        "action_shape": list(item["action"].shape),
        "video_shape": list(item["video"].shape),
        "timeline": "current1+future32",
        "prompt_source": "RoboTwin full_description",
        "prompt_manifest_source_commit": prompt_manifest["source_commit"],
        "prompt": item["ai_caption"],
        "views": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    }, sort_keys=True))
    family_probe_indices = {}
    for sample_index in range(min(len(dataset), 10_000)):
        record_index, _ = dataset._weighted_record_offset(sample_index)
        family = dataset._record_families[record_index]
        family_probe_indices.setdefault(family, sample_index)
        if set(family_probe_indices) == set(wanted):
            break
    assert set(family_probe_indices) == set(wanted), family_probe_indices
    for family, sample_index in sorted(family_probe_indices.items()):
        family_item = dataset[sample_index]
        assert family_item["family"] == family, (family_item["family"], family)
        assert tuple(family_item["action"].shape) == (32, 20)
        assert family_item["video"].shape[1] == 33
        assert family_item["ai_caption"] == canonical_full_description(family_item["task_name"])
        print(json.dumps({
            "decode_probe_family": family,
            "sample_index": sample_index,
            "action_shape": list(family_item["action"].shape),
            "video_shape": list(family_item["video"].shape),
            "timeline": "current1+future32",
            "prompt_source": "RoboTwin full_description",
            "prompt_manifest_source_commit": prompt_manifest["source_commit"],
            "prompt": family_item["ai_caption"],
        }, sort_keys=True))
        del family_item
    del dataset, item
    gc.collect()
PY

echo "[DRYRUN] validating Cosmos3 structured TOML"
DRYRUN_JOB_NAME="cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_20step_dryrun"
AFD_PROTOCOL=mix4 \
BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
WAN_VAE_PATH="$WAN_VAE_PATH" \
"$REPO/.venv/bin/python" -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/actionfollowing_full50_mix4.toml \
  --dryrun \
  -- \
  trainer.max_iter=20 \
  trainer.logging_iter=1 \
  checkpoint.save_iter=20 \
  dataloader_train.max_samples_per_batch=2 \
  scheduler.cycle_lengths='[20]' \
  scheduler.warm_up_steps='[1]' \
  job.name="$DRYRUN_JOB_NAME"

run_train() {
  local per_rank_batch="$1"
  local label="$2"
  local output_root="$OUT_BASE/$label"
  local run_name="cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_${label}_20step_smoke"
  mkdir -p "$output_root"
  echo "[TRAIN] label=$label per_rank_batch=$per_rank_batch global_batch=$((per_rank_batch * 8))"
  AFD_PROTOCOL=mix4 \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  OUTPUT_ROOT="$output_root" \
  IMAGINAIRE_OUTPUT_ROOT="$output_root" \
  EXTRA_TAIL_OVERRIDES="trainer.max_iter=20 trainer.logging_iter=1 checkpoint.save_iter=20 dataloader_train.max_samples_per_batch=$per_rank_batch scheduler.cycle_lengths=[20] scheduler.warm_up_steps=[1] job.name=$run_name" \
  MASTER_PORT=29541 \
  NPROC_PER_NODE=8 \
  bash "$REPO/examples/launch_sft_actionfollowing_full50_mix4.sh"
}

if run_train 2 bs16; then
  EFFECTIVE_BATCH=16
  TRAIN_OUT="$OUT_BASE/bs16"
else
  rc=$?
  if grep -RqiE "CUDA out of memory|CUDA error: out of memory|OutOfMemoryError" "$OUT_BASE/bs16"; then
    echo "[OOM] global batch 16 failed; retrying the allowed global batch 8 fallback"
    run_train 1 bs8
    EFFECTIVE_BATCH=8
    TRAIN_OUT="$OUT_BASE/bs8"
  else
    exit "$rc"
  fi
fi

LATEST_FILE="$(find "$TRAIN_OUT" -name latest_checkpoint.txt -type f -print -quit)"
[[ -n "$LATEST_FILE" ]] || die "20-step run completed without latest_checkpoint.txt"
LATEST_ITER="$(cat "$LATEST_FILE")"
[[ "$LATEST_ITER" == "iter_000000020" ]] || die "expected latest checkpoint iter_000000020, got $LATEST_ITER"
[[ -d "$(dirname "$LATEST_FILE")/$LATEST_ITER" ]] || die "latest checkpoint directory missing"
[[ -f "$(dirname "$LATEST_FILE")/$LATEST_ITER/model/.metadata" ]] || die "model DCP metadata missing"
[[ -f "$(dirname "$LATEST_FILE")/$LATEST_ITER/trainer/.metadata" ]] || die "trainer DCP metadata missing"

TRAIN_LOG="$TRAIN_OUT/logs/actionfollowing_full50_mix4_sft.log"
[[ -f "$TRAIN_LOG" ]] || die "persistent training log missing: $TRAIN_LOG"
TRAIN_AUDIT="$OUT_BASE/TRAIN_AUDIT.json"
"$REPO/.venv/bin/python" - "$TRAIN_LOG" "$TRAIN_AUDIT" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
audit_path = Path(sys.argv[2])
pattern = re.compile(r"\[RANK 0\] Iteration (\d+):.*?\| Loss: ([^ |]+)")
loss_by_step = {}
for step_text, loss_text in pattern.findall(log_path.read_text(errors="replace")):
    loss_by_step[int(step_text)] = float(loss_text)

expected_steps = list(range(1, 21))
if sorted(loss_by_step) != expected_steps:
    raise SystemExit(f"expected exactly rank-0 optimizer steps 1..20, got {sorted(loss_by_step)}")
if not all(math.isfinite(value) for value in loss_by_step.values()):
    raise SystemExit(f"non-finite rank-0 loss: {loss_by_step}")

audit = {
    "status": "passed",
    "optimizer_steps": expected_steps,
    "finite_losses": True,
    "final_loss": loss_by_step[20],
    "training_log": str(log_path),
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, sort_keys=True))
PY
FINAL_LOSS="$("$REPO/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_loss"])' "$TRAIN_AUDIT")"

printf 'status=passed\nmodel=Cosmos3-Nano\nprotocol=mix4\ndata_root=%s\ntimeline=current1+future32\noptimizer_steps=20\neffective_global_batch=%s\nfinal_loss=%s\ncheckpoint=%s\n' \
  "$AFD_ROOT" "$EFFECTIVE_BATCH" "$FINAL_LOSS" "$(dirname "$LATEST_FILE")/$LATEST_ITER" | \
  tee "$OUT_BASE/SMOKE_RESULT.txt"
