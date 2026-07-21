# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "examples" / "actionfollowing" / "prepare_minimal_inference.py"
SPEC = importlib.util.spec_from_file_location("prepare_minimal_inference", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_minimal_sample_accepts_native_contract() -> None:
    MODULE.validate_minimal_sample(
        torch.zeros(3, 33, 384, 256, dtype=torch.uint8),
        torch.zeros(32, 20),
        "Place the burger beside the fries.",
    )


@pytest.mark.parametrize(
    ("video", "action", "prompt"),
    [
        (torch.zeros(3, 32, 384, 256), torch.zeros(32, 20), "prompt"),
        (torch.zeros(3, 33, 384, 256), torch.zeros(32, 14), "prompt"),
        (torch.zeros(3, 33, 384, 256), torch.zeros(32, 20), ""),
    ],
)
def test_validate_minimal_sample_rejects_contract_mismatch(video, action, prompt) -> None:
    with pytest.raises(ValueError):
        MODULE.validate_minimal_sample(video, action, prompt)


def test_split_cosmos_tshape() -> None:
    frame = torch.arange(3 * 384 * 256, dtype=torch.int64).reshape(3, 384, 256).to(torch.uint8)
    views = MODULE.split_cosmos_tshape(frame)
    assert views["observation.images.cam_high"].shape == (256, 256, 3)
    assert views["observation.images.cam_left_wrist"].shape == (128, 128, 3)
    assert views["observation.images.cam_right_wrist"].shape == (128, 128, 3)
