#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare one reproducible ActionFollowing sample for Cosmos3 inference.

The output is deliberately model-agnostic at the asset boundary: it contains
the three current RGB cameras, Cosmos3's native T-shaped current frame, the
32x20 physical Rot6D action chunk, the 33-frame ground truth video, and an
official Cosmos inference JSON.  No action normalization is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.action.datasets.actionfollowing_lerobot_dataset import (
    ACTION_DIM,
    CAMERA_FEATURES,
    ActionFollowingLeRobotDataset,
)

ACTION_CHANNELS = (
    "left.dx",
    "left.dy",
    "left.dz",
    "left.rot6.r00",
    "left.rot6.r10",
    "left.rot6.r20",
    "left.rot6.r01",
    "left.rot6.r11",
    "left.rot6.r21",
    "left.gripper",
    "right.dx",
    "right.dy",
    "right.dz",
    "right.rot6.r00",
    "right.rot6.r10",
    "right.rot6.r20",
    "right.rot6.r01",
    "right.rot6.r11",
    "right.rot6.r21",
    "right.gripper",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_minimal_sample(video: torch.Tensor, action: torch.Tensor, prompt: str) -> None:
    if tuple(video.shape[:2]) != (3, 33):
        raise ValueError(f"Expected T-shaped RGB video [3,33,H,W], got {tuple(video.shape)}")
    if tuple(action.shape) != (32, ACTION_DIM):
        raise ValueError(f"Expected action [32,{ACTION_DIM}], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise ValueError("Action contains non-finite values")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("RoboTwin full_description is empty")
    height, width = int(video.shape[-2]), int(video.shape[-1])
    if height % 3 or width % 2:
        raise ValueError(
            "Expected Cosmos3 native T-shape with a 2/3-height head row and two equal wrist columns; "
            f"got HxW={height}x{width}"
        )


def split_cosmos_tshape(frame_chw: torch.Tensor) -> dict[str, np.ndarray]:
    """Recover head/left/right RGB views from Cosmos3's native T-shape."""

    frame = frame_chw.permute(1, 2, 0).cpu().numpy()
    height, width = frame.shape[:2]
    head_height = height * 2 // 3
    if head_height * 3 != height * 2 or width % 2:
        raise ValueError(f"Invalid Cosmos3 T-shape dimensions: {height}x{width}")
    return {
        CAMERA_FEATURES[0]: frame[:head_height],
        CAMERA_FEATURES[1]: frame[head_height:, : width // 2],
        CAMERA_FEATURES[2]: frame[head_height:, width // 2 :],
    }


def _find_virtual_index(dataset: ActionFollowingLeRobotDataset, task: str, family: str) -> int:
    # Mapping a virtual index is pure integer math.  This avoids decoding any
    # video until the requested task/family is found.
    search_limit = min(len(dataset), 1_000_000)
    for index in range(search_limit):
        record_idx, _ = dataset._weighted_record_offset(index)  # noqa: SLF001
        ds_idx = dataset._episode_records[record_idx][0]  # noqa: SLF001
        if dataset._task_by_source[ds_idx] == task and dataset._family_by_source[ds_idx] == family:  # noqa: SLF001
            return index
    raise LookupError(f"Could not find task={task!r}, family={family!r} in first {search_limit} virtual samples")


def _write_ground_truth_video(path: Path, video_cthw: torch.Tensor, fps: int) -> None:
    import imageio.v3 as iio

    frames = video_cthw.permute(1, 2, 3, 0).cpu().numpy()
    iio.imwrite(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ActionFollowingLeRobotDataset(
        root=str(args.dataset_root),
        protocol=args.protocol,
        audit_num_samples=args.audit_samples,
    )
    index = _find_virtual_index(dataset, args.task, args.family)
    sample = dataset[index]
    video = sample["video"]
    action = sample["action"].float()
    prompt = sample["ai_caption"].strip()
    validate_minimal_sample(video, action, prompt)

    current_tshape = args.output_dir / "current_tshape.png"
    Image.fromarray(video[:, 0].permute(1, 2, 0).cpu().numpy()).save(current_tshape)
    for camera_name, rgb in split_cosmos_tshape(video[:, 0]).items():
        Image.fromarray(rgb).save(args.output_dir / f"{camera_name}.png")

    actions_path = args.output_dir / "actions_physical_rot6d20.json"
    actions_path.write_text(json.dumps(action.cpu().tolist(), indent=2), encoding="utf-8")
    prompt_path = args.output_dir / "full_description.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    ground_truth_path = args.output_dir / "ground_truth_tshape_33frames.mp4"
    _write_ground_truth_video(ground_truth_path, video, int(sample["conditioning_fps"]))

    inference_input = {
        "action_chunk_size": 32,
        "action_path": str(actions_path.resolve()),
        "domain_name": "robotwin-actionfollowing",
        "fps": int(sample["conditioning_fps"]),
        "guidance": args.guidance,
        "image_size": 256,
        "model_mode": "forward_dynamics",
        "name": args.name,
        "num_steps": args.num_steps,
        "prompt": prompt,
        "seed": args.seed,
        "shift": args.shift,
        "view_point": "The head camera is on top; left and right wrist cameras are on the bottom row.",
        "vision_path": str(current_tshape.resolve()),
    }
    input_path = args.output_dir / "cosmos3_forward_dynamics.json"
    input_path.write_text(json.dumps(inference_input, indent=2), encoding="utf-8")

    metadata = {
        "status": "prepared",
        "dataset_root": str(args.dataset_root),
        "protocol": args.protocol,
        "family": args.family,
        "task": args.task,
        "virtual_index": index,
        "prompt": prompt,
        "timeline": "current1+future32",
        "observation_shape": list(video.shape),
        "action_shape": list(action.shape),
        "action_space": "physical robot-base-frame delta-EE Rot6D20; no normalization",
        "action_channels": ACTION_CHANNELS,
        "rot6d": "concat(R[:,0], R[:,1])",
        "camera_read_order": list(CAMERA_FEATURES),
        "cosmos_tshape": "head top; left wrist bottom-left; right wrist bottom-right",
        "sha256": {
            current_tshape.name: _sha256(current_tshape),
            actions_path.name: _sha256(actions_path),
            prompt_path.name: _sha256(prompt_path),
            ground_truth_path.name: _sha256(ground_truth_path),
            input_path.name: _sha256(input_path),
        },
    }
    metadata_path = args.output_dir / "input_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("clean", "mix4"), default="clean")
    parser.add_argument("--family", default="clean")
    parser.add_argument("--task", default="place_burger_fries")
    parser.add_argument("--name", default="actionfollowing_place_burger_fries_clean0")
    parser.add_argument("--audit-samples", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), indent=2))
