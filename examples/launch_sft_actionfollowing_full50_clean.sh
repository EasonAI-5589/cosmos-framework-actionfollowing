#!/usr/bin/env bash
# Cosmos3-Nano ActionFollowingData full-50 clean. No AIHC submission is performed here.

TOML_FILE="examples/toml/sft_config/actionfollowing_full50_clean.toml"
export AFD_PROTOCOL="clean"
: "${AFD_ROOT:=/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D}"
: "${BASE_CHECKPOINT_PATH:?export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP directory>}"
: "${WAN_VAE_PATH:?export WAN_VAE_PATH=<Wan2.2_VAE.pth>}"
export AFD_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH

EXTRA_DATASET_CHECK='[[ -f "$AFD_ROOT/demo_clean_zed2i_visible/adjust_bottle/meta/info.json" ]] || { echo "ERROR: canonical AFD root is not mounted: $AFD_ROOT" >&2; exit 1; }'
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
