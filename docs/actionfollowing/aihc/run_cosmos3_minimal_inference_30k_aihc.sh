#!/usr/bin/env bash
# Cosmos3 ActionFollowing iter_000030000 minimal-inference AIHC bootstrap.
set -Eeuo pipefail

die() { echo "[FATAL] $*" >&2; exit 1; }
ensure_mount_alias() {
  local alias_path="$1" target_path="$2"
  [[ -d "$target_path" ]] || die "missing AIHC mount: $target_path"
  if [[ ! -e "$alias_path" ]]; then ln -s "$target_path" "$alias_path"; fi
  [[ "$(readlink -f "$alias_path")" == "$(readlink -f "$target_path")" ]] || \
    die "$alias_path does not resolve to $target_path"
}

ensure_mount_alias /mnt/gyc        /mnt/dataset/csx_workspace
ensure_mount_alias /mnt/gyc_ckp    /mnt/dataset/csx_ckp
ensure_mount_alias /mnt/public_ckp /mnt/dataset/public_data

REPO=/mnt/gyc/cosmos-framework-actionfollowing-inference-fixed
expected_commit="${COSMOS_EXPECTED_COMMIT:?COSMOS_EXPECTED_COMMIT is required}"
actual_commit="$(git -C "$REPO" rev-parse HEAD)"
[[ "$actual_commit" == "$expected_commit" ]] || \
  die "repo commit mismatch: expected=$expected_commit actual=$actual_commit"

export REPO_ROOT="$REPO"
export PYTHON=/mnt/gyc/cosmos-framework/.venv/bin/python
[[ -x "$PYTHON" ]] || die "Cosmos runtime python missing: $PYTHON"
export DCP_RUN_ROOT=/mnt/gyc_ckp/Action-Following/outputs/cosmos3/mix4/train_40000_20260721_motusdata_future32_prompt_job-3cmb7l4p44jw/bs16/cosmos3_actionfollowing/forward_dynamics_mix4/cosmos3_nano_afd_full50_mix4_rot6d20_a32_future32_prompt_bs16_40000step
export DCP_ITER=iter_000030000
export PREPARE_INPUT=0
export INPUT_DIR=/mnt/gyc_ckp/Action-Following/outputs/handoff_inputs/place_burger_fries_clean0_20260721
run_tag="${AIHC_JOB_ID:-${AIHC_JOB_NAME:-manual_$(date +%Y%m%d_%H%M%S)}}"
export HANDOFF_ROOT="${DCP_RUN_ROOT}/handoff_inference/${DCP_ITER}_place_burger_fries_clean0_steps10_aihc_train_20260721_${run_tag}"

[[ ! -e "$HANDOFF_ROOT" ]] || die "refusing to reuse handoff root: $HANDOFF_ROOT"
mkdir -p "$HANDOFF_ROOT"
exec > >(tee -a "$HANDOFF_ROOT/bootstrap.log") 2>&1
trap 'rc=$?; printf "[BOOTSTRAP_ERROR] rc=%s line=%s command=%q\n" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND"; exit "$rc"' ERR

echo "[INFERENCE_CONTRACT] model=cosmos3 checkpoint=$DCP_ITER action_shape=[32,20] normalization=none input=$INPUT_DIR"
echo "[PROVENANCE] commit=$actual_commit repo=$REPO output=$HANDOFF_ROOT"
[[ -f "$DCP_RUN_ROOT/config.yaml" ]] || die "training config missing"
[[ -d "$DCP_RUN_ROOT/checkpoints/$DCP_ITER/model" ]] || die "DCP checkpoint missing"
for file in current_tshape.png actions_physical_rot6d20.json full_description.txt cosmos3_forward_dynamics.json input_metadata.json; do
  [[ -s "$INPUT_DIR/$file" ]] || die "input missing: $INPUT_DIR/$file"
done

bash "$REPO/examples/actionfollowing/run_cosmos3_minimal_inference.sh"
