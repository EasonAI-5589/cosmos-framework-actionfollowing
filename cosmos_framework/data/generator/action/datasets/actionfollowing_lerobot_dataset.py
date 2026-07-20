# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""ActionFollowingData full-50 LeRobot adapter for Cosmos3 action SFT.

The canonical dataset already stores dual-arm 20D Rot6D actions.  This adapter
therefore reads actions verbatim and never converts them through Euler angles.
It also implements the protocol-level clean and mix4 sampling rules over flat
chunk samples rather than selecting a family first.
"""

from __future__ import annotations

import math
import os
import random
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from tqdm.auto import tqdm as hf_tqdm

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets import cosmos3_action_lerobot as _base_lerobot
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import BaseActionLeRobotDataset
from cosmos_framework.utils import log

Protocol = Literal["clean", "mix4"]

ACTION_DIM = 20
STATE_DIM = 20
# Enhanced train videos can be 10 FPS while their action timeline remains at
# 30 Hz.  The decoder therefore needs to accept the nearest repeated video
# frame (maximum half-frame error is 0.05 s) without changing action timing.
VIDEO_TIMESTAMP_TOLERANCE_S = 0.051
ACTION_FEATURE = "action"
STATE_FEATURE = "observation.state"
CAMERA_FEATURES = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _build_actionfollowing_delta_timestamps(*, fps: float, chunk_length: int) -> dict[str, list[float]]:
    """Build the canonical ``current1 + future32`` forward-dynamics timeline.

    A 32-action chunk represents transitions ``A[t+i]: O[t+i] -> O[t+i+1]``.
    The action query therefore has 32 timestamps while every camera query has
    33 timestamps so the final action is supervised by the real ``O[t+32]``.
    """

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if chunk_length <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_length}")
    dt = 1.0 / float(fps)
    observation_timestamps = [i * dt for i in range(chunk_length + 1)]
    return {
        ACTION_FEATURE: observation_timestamps[:-1],
        CAMERA_FEATURES[0]: observation_timestamps,
        CAMERA_FEATURES[1]: observation_timestamps,
        CAMERA_FEATURES[2]: observation_timestamps,
    }


def _num_valid_forward_dynamics_windows(*, family: str, episode_length: int, chunk_length: int) -> int:
    """Count windows with 32 actions *and* 33 genuine observations.

    Perturbed records are already chunk-level, so a valid 33-row record yields
    exactly its canonical prefix.  Trajectory-level families use stride-1
    windows and deliberately exclude the terminal start that has no ``O[t+32]``.
    """

    required_observations = int(chunk_length) + 1
    if episode_length < required_observations:
        return 0
    if family == "perturbed":
        return 1
    return int(episode_length) - int(chunk_length)


def _full_description_from_sample(sample: dict[str, Any], *, task_name: str) -> str:
    """Return the RoboTwin ``full_description`` stored in LeRobot task metadata."""

    prompt = sample.get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{task_name}: missing non-empty RoboTwin full_description in LeRobot sample['task']; "
            "repair meta/tasks.parquet from RoboTwin description/task_instruction before training"
        )
    return prompt.strip()


ROBOTWIN_50_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

MIX4_TARGET_RATIOS = {
    "clean": 4.0,
    "perturbed": 1.0,
    "random_feasible": 1.0,
    "counterfactual_replay": 1.0,
    "exploration": 1.0,
}


def _resolve_actionfollowing_video_path(video_path: str | Path) -> Path:
    """Resolve a split video reference against the immutable full LeRobot root."""

    path = Path(video_path)
    if path.is_file():
        return path
    remap = os.environ.get("AFD_VIDEO_SYMLINK_PREFIX_REMAP", "")
    old_prefix, separator, new_prefix = remap.partition("=")
    if separator and path.is_symlink():
        target = Path(os.readlink(path))
        if target.is_absolute():
            try:
                remapped = Path(new_prefix) / target.relative_to(old_prefix)
            except ValueError:
                pass
            else:
                if remapped.is_file():
                    return remapped
    dataset_root_value = os.environ.get("AFD_ROOT")
    if not dataset_root_value:
        return path
    dataset_root = Path(dataset_root_value)
    try:
        relative = path.relative_to(dataset_root)
    except ValueError:
        return path
    fallback_roots: list[Path] = []
    if dataset_root.name == "train":
        fallback_roots.append(dataset_root.parent)
    fallback_roots.extend(Path(value) for value in os.environ.get("AFD_VIDEO_FALLBACK_ROOTS", "").split(":") if value)
    for fallback_root in fallback_roots:
        candidate = fallback_root / relative
        if candidate.is_file():
            return candidate
    return path


def _install_actionfollowing_video_fallback() -> None:
    cache_type = _base_lerobot._LRUVideoDecoderCache
    if getattr(cache_type, "_actionfollowing_video_fallback_installed", False):
        return
    original_get_decoder = cache_type.get_decoder

    def get_decoder_with_fallback(self, video_path: str) -> Any:
        return original_get_decoder(self, str(_resolve_actionfollowing_video_path(video_path)))

    cache_type.get_decoder = get_decoder_with_fallback
    cache_type._actionfollowing_video_fallback_installed = True


_install_actionfollowing_video_fallback()


def actionfollowing_sources(root: str | Path, protocol: Protocol) -> list[tuple[str, str, str]]:
    """Return ``(root, family, task)`` sources in a stable order."""

    root = Path(root)
    if protocol not in ("clean", "mix4"):
        raise ValueError(f"Unsupported ActionFollowingData protocol: {protocol!r}")

    sources: list[tuple[str, str, str]] = []
    for task in ROBOTWIN_50_TASKS:
        sources.append((str(root / "demo_clean_zed2i_visible" / task), "clean", task))
        if protocol == "mix4":
            sources.extend(
                [
                    (str(root / "perturbed" / "pca" / "tasks" / task), "perturbed", task),
                    (str(root / "perturbed" / "raw" / "tasks" / task), "perturbed", task),
                    (
                        str(root / "random_feasible" / "uniform" / "tasks" / task),
                        "random_feasible",
                        task,
                    ),
                    (
                        str(root / "random_feasible" / "weighted" / "tasks" / task),
                        "random_feasible",
                        task,
                    ),
                    (
                        str(root / "counterfactual_replay" / "tasks" / task),
                        "counterfactual_replay",
                        task,
                    ),
                    (str(root / "exploration" / "tasks" / task), "exploration", task),
                ]
            )
    return sources


class ActionFollowingLeRobotDataset(BaseActionLeRobotDataset):
    """Canonical full-50 ActionFollowingData dataset for Cosmos3.

    The public virtual index has the same length as the flattened canonical
    chunk index.  For ``mix4``, each virtual index is deterministically mapped
    through a weighted CDF so the family probabilities are 4:1:1:1:1 while
    retaining chunk-level sampling.
    """

    def __init__(
        self,
        root: str,
        protocol: Protocol = "clean",
        fps: float = 30.0,
        chunk_length: int = 32,
        mode: str = "forward_dynamics",
        split: str = "full",
        seed: int = 20260717,
        tolerance_s: float = VIDEO_TIMESTAMP_TOLERANCE_S,
        max_loaded_datasets: int = 4,
        fast_init_max_workers: int = 64,
        audit_num_samples: int = 10000,
        audit_max_abs_error: float = 0.02,
    ) -> None:
        if chunk_length != 32:
            raise ValueError(f"ActionFollowingData protocol requires chunk_length=32, got {chunk_length}")
        if split != "full":
            raise ValueError("ActionFollowingData baseline training uses the frozen full split; set split='full'.")
        if mode != "forward_dynamics":
            raise ValueError("ActionFollowingData Cosmos baseline is forward_dynamics, not action-policy training.")

        self.protocol = protocol
        self.seed = int(seed)
        self._sampling_epoch = 0
        self._source_specs = actionfollowing_sources(root, protocol)
        self._family_by_source = [family for _, family, _ in self._source_specs]
        self._task_by_source = [task for _, _, task in self._source_specs]
        self._source_fps: list[float] = []
        self._sample_mass_by_family: dict[str, float] = {}
        self._record_mass_cum_ends: list[float] = []
        self._record_families: list[str] = []
        self._total_sampling_mass = 0.0

        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=seed,
            split_val_ratio=0.01,
            split=split,
            mode=mode,
            embodiment_type="robotwin-actionfollowing",
            viewpoint="concat_view",
            pose_convention="backward_framewise",
            rotation_format="rot6d",
            action_normalization=None,
            tolerance_s=tolerance_s,
            max_loaded_datasets=max_loaded_datasets,
            enable_fast_init=True,
            fast_init_max_workers=fast_init_max_workers,
        )

        self._all_shard_roots = [source_root for source_root, _, _ in self._source_specs]
        self._register_sources_with_native_fps()
        self._build_protocol_sampling_index()
        audit = self.audit_sampling(num_samples=audit_num_samples)
        if audit["max_abs_error"] > float(audit_max_abs_error):
            raise RuntimeError(
                "ActionFollowingData sampling audit failed: "
                f"max_abs_error={audit['max_abs_error']:.6f} > {audit_max_abs_error:.6f}; audit={audit}"
            )
        log.info(f"ActionFollowingData sampling audit: {audit}")

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _build_action_spec(self) -> ActionSpec:
        return build_action_spec(
            Pos(prefix="left"),
            Rot("rot6d", prefix="left"),
            Gripper(prefix="left"),
            Pos(prefix="right"),
            Rot("rot6d", prefix="right"),
            Gripper(prefix="right"),
        )

    @staticmethod
    def _feature_width(meta: LeRobotDatasetMetadata, key: str) -> int | None:
        feature = meta.info.get("features", {}).get(key)
        if not feature:
            return None
        shape = feature.get("shape") or []
        return int(shape[-1]) if shape else None

    def _validate_metadata(self, meta: LeRobotDatasetMetadata, root: str) -> None:
        action_width = self._feature_width(meta, ACTION_FEATURE)
        state_width = self._feature_width(meta, STATE_FEATURE)
        if action_width != ACTION_DIM:
            raise ValueError(f"{root}: expected action width {ACTION_DIM}, got {action_width}")
        if state_width != STATE_DIM:
            raise ValueError(f"{root}: expected state width {STATE_DIM}, got {state_width}")
        missing_views = [key for key in CAMERA_FEATURES if key not in meta.info.get("features", {})]
        if missing_views:
            raise ValueError(f"{root}: missing required three-view features: {missing_views}")

    def _register_sources_with_native_fps(self) -> None:
        roots = self._all_shard_roots
        missing = [root for root in roots if not Path(root).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} ActionFollowingData roots; first={missing[0]}")

        # Hugging Face datasets may enter tqdm.contrib.concurrent.ensure_lock
        # from every metadata-loading thread.  Lazily creating that class lock
        # inside those threads is racy: one thread can delete ``tqdm._lock``
        # while another is still restoring it.  Materialize the shared lock
        # once before starting the outer metadata executor.
        hf_tqdm.get_lock()
        workers = min(max(1, self._fast_init_max_workers), len(roots))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            metas = list(
                executor.map(
                    lambda source_root: LeRobotDatasetMetadata(repo_id="local", root=source_root, revision="local"),
                    roots,
                )
            )

        for source_index, (root, meta) in enumerate(zip(roots, metas)):
            self._validate_metadata(meta, root)
            source_fps = float(meta.fps)
            self._source_fps.append(source_fps)
            delta_timestamps = _build_actionfollowing_delta_timestamps(
                fps=source_fps,
                chunk_length=self._chunk_length,
            )
            self._register_source(
                root=root,
                delta_timestamps=delta_timestamps,
                tolerance_s=self._tolerance_s,
                dataset_label=f"{self._family_by_source[source_index]}/{self._task_by_source[source_index]}",
                prefetched_meta=meta,
            )

    def _append_index_records(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        ds_idx: int,
        dataset_label: str | None = None,
    ) -> None:
        """Build protocol windows: perturbed prefix-only, others stride-1."""

        family = self._family_by_source[ds_idx]
        episodes = meta.episodes
        starts = list(episodes["dataset_from_index"])
        stops = list(episodes["dataset_to_index"])
        lengths = list(episodes["length"])
        kept = 0
        for episode_id in range(meta.total_episodes):
            length = int(lengths[episode_id])
            sample_start = int(starts[episode_id])
            valid_len = _num_valid_forward_dynamics_windows(
                family=family,
                episode_length=length,
                chunk_length=self._chunk_length,
            )
            if valid_len <= 0:
                continue
            if int(stops[episode_id]) - sample_start != length:
                raise ValueError(f"{dataset_label}: inconsistent episode metadata for episode {episode_id}")
            self._episode_records.append((ds_idx, sample_start, valid_len, episode_id))
            self._num_valid_indices += valid_len
            self._episode_cum_ends.append(self._num_valid_indices)
            kept += valid_len
        log.info(f"ActionFollowingData [{dataset_label}]: episodes={meta.total_episodes}, chunks={kept}")

    def _build_protocol_sampling_index(self) -> None:
        family_counts: Counter[str] = Counter()
        for ds_idx, _start, valid_len, _episode_id in self._episode_records:
            family_counts[self._family_by_source[ds_idx]] += int(valid_len)
        self.family_effective_counts = dict(sorted(family_counts.items()))

        raw_targets = {"clean": 1.0} if self.protocol == "clean" else MIX4_TARGET_RATIOS
        absent = sorted(set(raw_targets) - set(family_counts))
        unexpected = sorted(set(family_counts) - set(raw_targets))
        if absent or unexpected:
            raise ValueError(f"Protocol family mismatch: absent={absent}, unexpected={unexpected}")

        target_total = float(sum(raw_targets.values()))
        self.target_family_probabilities = {
            family: float(weight / target_total) for family, weight in raw_targets.items()
        }
        self._sample_mass_by_family = {
            family: self.target_family_probabilities[family] / float(family_counts[family]) for family in family_counts
        }

        running_mass = 0.0
        for ds_idx, _start, valid_len, _episode_id in self._episode_records:
            family = self._family_by_source[ds_idx]
            running_mass += float(valid_len) * self._sample_mass_by_family[family]
            self._record_mass_cum_ends.append(running_mass)
            self._record_families.append(family)
        self._total_sampling_mass = running_mass
        if not math.isclose(running_mass, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Protocol sampling mass must sum to 1, got {running_mass}")

        self.family_per_chunk_weights = {
            family: self._sample_mass_by_family[family]
            / self._sample_mass_by_family.get("perturbed", self._sample_mass_by_family[family])
            for family in sorted(self._sample_mass_by_family)
        }
        log.info(f"ActionFollowingData family effective counts: {self.family_effective_counts}")
        log.info(f"ActionFollowingData family per-chunk weights: {self.family_per_chunk_weights}")

    @staticmethod
    def _coprime_multiplier(length: int, rng: random.Random) -> int:
        if length <= 1:
            return 1
        candidate = rng.randrange(1, length)
        while math.gcd(candidate, length) != 1:
            candidate = (candidate + 1) % length or 1
        return candidate

    def _sample_quantile(self, index: int, length: int, seed: int, epoch: int) -> float:
        rng = random.Random(int(seed) + (int(epoch) + 1) * 1_000_003 + int(length) * 9_176)
        multiplier = self._coprime_multiplier(length, rng)
        offset = rng.randrange(length) if length > 1 else 0
        shift = rng.random()
        rank = (multiplier * (int(index) % length) + offset) % length
        return (float(rank) + shift) / float(length)

    def _weighted_record_offset(
        self,
        index: int,
        *,
        length: int | None = None,
        seed: int | None = None,
        epoch: int | None = None,
    ) -> tuple[int, int]:
        length = len(self) if length is None else int(length)
        if length <= 0:
            raise IndexError("Empty ActionFollowingData dataset")
        quantile = self._sample_quantile(
            index,
            length,
            self.seed if seed is None else int(seed),
            self._sampling_epoch if epoch is None else int(epoch),
        )
        target_mass = quantile * self._total_sampling_mass
        record_idx = min(bisect_right(self._record_mass_cum_ends, target_mass), len(self._episode_records) - 1)
        previous_mass = 0.0 if record_idx == 0 else self._record_mass_cum_ends[record_idx - 1]
        family = self._record_families[record_idx]
        sample_mass = self._sample_mass_by_family[family]
        valid_len = int(self._episode_records[record_idx][2])
        offset = min(int((target_mass - previous_mass) / sample_mass), valid_len - 1)
        return record_idx, max(0, offset)

    def _resolve_index(self, idx: int) -> tuple[int, int, int, int]:
        record_idx, frame_offset = self._weighted_record_offset(idx)
        ds_idx, row_start, _valid_len, episode_id = self._episode_records[record_idx]
        return ds_idx, int(row_start) + frame_offset, episode_id, frame_offset

    def get_shuffle_blocks(self, block_size: int = 4096) -> list[tuple[int, int]]:
        """Expose virtual-index blocks for rank/worker sharding."""

        return [(start, min(block_size, len(self) - start)) for start in range(0, len(self), block_size)]

    def audit_sampling(
        self,
        num_samples: int = 10000,
        seed: int = 20260717,
        epoch: int = 0,
    ) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        for index in range(int(num_samples)):
            record_idx, _ = self._weighted_record_offset(
                index,
                length=int(num_samples),
                seed=int(seed),
                epoch=int(epoch),
            )
            counts[self._record_families[record_idx]] += 1
        observed = {family: counts[family] / float(num_samples) for family in self.target_family_probabilities}
        max_abs_error = max(
            abs(observed[family] - target) for family, target in self.target_family_probabilities.items()
        )
        return {
            "num_samples": int(num_samples),
            "seed": int(seed),
            "epoch": int(epoch),
            "family_counts": dict(counts),
            "observed_family_probabilities": observed,
            "target_family_probabilities": self.target_family_probabilities,
            "max_abs_error": float(max_abs_error),
        }

    @staticmethod
    def _compose_three_views(sample: dict[str, Any]) -> torch.Tensor:
        head = sample[CAMERA_FEATURES[0]]
        left = sample[CAMERA_FEATURES[1]]
        right = sample[CAMERA_FEATURES[2]]
        _, _, height, width = head.shape
        half_height, half_width = height // 2, width // 2
        left = F.interpolate(left, size=(half_height, half_width), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_height, half_width), mode="bilinear", align_corners=False)
        return torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, ds_idx, _row_idx, sample = self._fetch_sample(idx)
        action = sample[ACTION_FEATURE].float()
        if tuple(action.shape) != (self._chunk_length, ACTION_DIM):
            raise ValueError(
                f"Expected canonical action shape ({self._chunk_length}, {ACTION_DIM}), got {tuple(action.shape)}"
            )

        video = self._compose_three_views(sample)
        expected_observations = self._chunk_length + 1
        if video.shape[0] != expected_observations:
            raise ValueError(f"Expected {expected_observations} decoded frames, got {video.shape[0]}")
        task_name = self._task_by_source[ds_idx]
        return self._build_result(
            mode=mode,
            video=video,
            action=action,
            ai_caption=_full_description_from_sample(sample, task_name=task_name),
            conditioning_fps=torch.tensor(round(self._source_fps[ds_idx]), dtype=torch.long),
            family=self._family_by_source[ds_idx],
            task_name=task_name,
            action_spec_names=self.action_names,
            additional_view_description=(
                "The top row is the head camera. The bottom row contains the left wrist camera "
                "on the left and the right wrist camera on the right."
            ),
        )


def get_actionfollowing_sft_dataset(
    *,
    root: str,
    protocol: Protocol,
    fps: float = 30.0,
    chunk_length: int = 32,
    mode: str = "forward_dynamics",
    resolution: str | int | None = "256",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 20260717,
    audit_num_samples: int = 10000,
    audit_max_abs_error: float = 0.02,
) -> torch.utils.data.Dataset:
    """Build the transformed Cosmos3 ActionFollowingData SFT dataset."""

    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
        ActionIterableShuffleDataset,
        ActionSFTDataset,
    )
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    dataset = ActionFollowingLeRobotDataset(
        root=root,
        protocol=protocol,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        split="full",
        seed=episode_shuffle_seed,
        audit_num_samples=audit_num_samples,
        audit_max_abs_error=audit_max_abs_error,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
    )
    transformed = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(transformed, seed=episode_shuffle_seed)
    return transformed


__all__ = [
    "ACTION_DIM",
    "CAMERA_FEATURES",
    "MIX4_TARGET_RATIOS",
    "ROBOTWIN_50_TASKS",
    "ActionFollowingLeRobotDataset",
    "actionfollowing_sources",
    "get_actionfollowing_sft_dataset",
]
