#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/mnt/dataset/csx_workspace/Action-Following/jobs/20260718/probe_actionfollowing_video_assets.log
exec > >(tee "$LOG") 2>&1

echo "[PROBE] job=${AIHC_JOB_NAME:-UNKNOWN}"
date -Is

LEROBOT_ROOTS=(
  /mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D
  /mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot
  /mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D
  /mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot
)
CANONICAL_ROOTS=(
  /mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_Rot6D/enhanced_v1_split/train
  /mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split/train
  /mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_Rot6D/enhanced_v1_split/train
  /mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split/train
)

for root in "${LEROBOT_ROOTS[@]}"; do
  echo "[LEROBOT_ROOT] $root"
  ls -ld "$root" 2>&1 || true
  for split in train .; do
    base="$root"
    [[ "$split" == train ]] && base="$root/train"
    path="$base/exploration/tasks/turn_switch/videos/observation.images.cam_high/chunk-000/file-000000.mp4"
    echo "[VIDEO_REF] $path"
    ls -l "$path" 2>&1 || true
    if [[ -L "$path" ]]; then
      echo "[READLINK] $(readlink "$path")"
      echo "[READLINK_F] $(readlink -f "$path" || true)"
    fi
  done
  find "$root" -maxdepth 12 -path '*/exploration/tasks/turn_switch/*' \( -type f -o -type l \) -print 2>/dev/null | head -80 || true
done

for root in "${CANONICAL_ROOTS[@]}"; do
  echo "[CANONICAL_ROOT] $root"
  ls -ld "$root" 2>&1 || true
  find "$root" -maxdepth 12 -path '*exploration*turn_switch*' \( -type f -o -type l \) -print 2>/dev/null | head -120 || true
done

echo "[GLOBAL_VIDEO_SEARCH]"
find /mnt/dataset/sixiangchen_workspace/Ideas/data -path '*exploration*turn_switch*' -name 'video.mp4' -print 2>/dev/null | head -120 || true

source /mnt/dataset/csx_workspace/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-predict2.5
source /mnt/dataset/csx_workspace/cosmos-predict2.5/.venv/bin/activate
python - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

for root in (
    Path("/mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train"),
    Path("/mnt/dataset/sixiangchen_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D"),
):
    source = root / "exploration/tasks/turn_switch"
    print("[PY_SOURCE]", source, "exists=", source.is_dir())
    info_path = source / "meta/info.json"
    if not info_path.is_file():
        continue
    info = json.loads(info_path.read_text())
    print("[PY_INFO]", info.get("video_path"), info.get("total_episodes"), info.get("total_frames"))
    episode_files = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
    rows = [row for file in episode_files for row in pq.read_table(file).to_pylist()]
    for row in rows[:10]:
        chunk = int(row.get("videos/observation.images.cam_high/chunk_index", 0))
        file_index = int(row.get("videos/observation.images.cam_high/file_index", 0))
        relative = info["video_path"].format(
            video_key="observation.images.cam_high",
            chunk_index=chunk,
            file_index=file_index,
            episode_chunk=chunk,
            episode_file=file_index,
        )
        path = source / relative
        print(
            "[PY_EPISODE]",
            row.get("episode_index"),
            path,
            "exists=", path.exists(),
            "lexists=", os.path.lexists(path),
            "islink=", path.is_symlink(),
            "link=", os.readlink(path) if path.is_symlink() else None,
        )
PY

echo "[PROBE_RESULT] completed"
