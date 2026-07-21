#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/gyc/cosmos-framework}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
AFD_ROOT="${AFD_ROOT:-/mnt/public_ckp/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train}"
DCP_RUN_ROOT="${DCP_RUN_ROOT:?Set DCP_RUN_ROOT to the Cosmos3 training run directory containing config.yaml}"
DCP_ITER="${DCP_ITER:-$(cat "${DCP_RUN_ROOT}/checkpoints/latest_checkpoint.txt")}"
DCP_PATH="${DCP_PATH:-${DCP_RUN_ROOT}/checkpoints/${DCP_ITER}}"
HANDOFF_ROOT="${HANDOFF_ROOT:-${DCP_RUN_ROOT}/handoff_inference/${DCP_ITER}_place_burger_fries_clean0_steps10}"
HF_MODEL="${HF_MODEL:-${HANDOFF_ROOT}/hf_model}"
INPUT_DIR="${INPUT_DIR:-${HANDOFF_ROOT}/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${HANDOFF_ROOT}/output}"
LOG_PATH="${LOG_PATH:-${HANDOFF_ROOT}/minimal_inference.log}"

export LD_LIBRARY_PATH=""
export PYTHONUNBUFFERED=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HANDOFF_ROOT}/hf_datasets_cache}"
export AFD_ROOT
export AFD_VIDEO_SYMLINK_PREFIX_REMAP="${AFD_VIDEO_SYMLINK_PREFIX_REMAP:-/mnt/dataset/csx_workspace/Ideas/data=/mnt/dataset/sixiangchen_workspace/Ideas/data}"

mkdir -p "${HANDOFF_ROOT}" "${INPUT_DIR}"
test -f "${DCP_RUN_ROOT}/config.yaml"
test -d "${DCP_PATH}/model"

if [[ "${PREPARE_INPUT:-1}" == "1" ]]; then
  "${PYTHON}" "${REPO_ROOT}/examples/actionfollowing/prepare_minimal_inference.py" \
    --dataset-root "${AFD_ROOT}" \
    --output-dir "${INPUT_DIR}" \
    --protocol clean \
    --family clean \
    --task place_burger_fries \
    --num-steps 10 \
    --seed 20260721
else
  test -s "${INPUT_DIR}/current_tshape.png"
  test -s "${INPUT_DIR}/actions_physical_rot6d20.json"
  test -s "${INPUT_DIR}/full_description.txt"
  test -s "${INPUT_DIR}/cosmos3_forward_dynamics.json"
  test -s "${INPUT_DIR}/input_metadata.json"
fi

if [[ ! -f "${HF_MODEL}/checkpoint.json" ]]; then
  "${PYTHON}" -m cosmos_framework.scripts.export_model \
    --checkpoint-path "${DCP_PATH}" \
    --config-file "${DCP_RUN_ROOT}/config.yaml" \
    --no-vit \
    -o "${HF_MODEL}"
fi

"${PYTHON}" -m cosmos_framework.scripts.inference \
  -i "${INPUT_DIR}/cosmos3_forward_dynamics.json" \
  -o "${OUTPUT_DIR}" \
  --checkpoint-path "${HF_MODEL}" \
  2>&1 | tee "${LOG_PATH}"

SAMPLE_DIR="${OUTPUT_DIR}/actionfollowing_place_burger_fries_clean0"
test -f "${SAMPLE_DIR}/sample_outputs.json"
find "${SAMPLE_DIR}" -type f -name '*.mp4' -size +0c -print -quit | grep -q .

cat > "${HANDOFF_ROOT}/INFERENCE_RESULT.txt" <<EOF
status=passed
model=cosmos3
checkpoint=${DCP_PATH}
exported_model=${HF_MODEL}
input=${INPUT_DIR}/cosmos3_forward_dynamics.json
output=${SAMPLE_DIR}
action_space=physical_robot_base_frame_deltaee_rot6d20_no_normalization
camera_layout=head_top_left_wrist_bottom_left_right_wrist_bottom_right
EOF

echo "[PASS] Cosmos3 minimal inference: ${SAMPLE_DIR}"
