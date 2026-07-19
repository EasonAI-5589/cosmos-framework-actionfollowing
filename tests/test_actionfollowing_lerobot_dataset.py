from cosmos_framework.data.generator.action.datasets.actionfollowing_lerobot_dataset import (
    ACTION_DIM,
    ROBOTWIN_50_TASKS,
    VIDEO_TIMESTAMP_TOLERANCE_S,
    ActionFollowingLeRobotDataset,
    _resolve_actionfollowing_video_path,
    actionfollowing_sources,
)


def test_video_timestamp_tolerance_supports_10fps_video_on_30hz_actions() -> None:
    assert VIDEO_TIMESTAMP_TOLERANCE_S == 0.051
    assert VIDEO_TIMESTAMP_TOLERANCE_S > 0.05


def test_source_inventory_is_full50() -> None:
    clean = actionfollowing_sources("/dataset", "clean")
    mix4 = actionfollowing_sources("/dataset", "mix4")
    assert len(ROBOTWIN_50_TASKS) == 50
    assert len(clean) == 50
    assert len(mix4) == 350
    assert {family for _, family, _ in mix4} == {
        "clean",
        "perturbed",
        "random_feasible",
        "counterfactual_replay",
        "exploration",
    }


def test_action_spec_is_dual_arm_rot6d20() -> None:
    dataset = ActionFollowingLeRobotDataset.__new__(ActionFollowingLeRobotDataset)
    spec = dataset._build_action_spec()
    assert spec.dim == ACTION_DIM == 20
    assert spec.rotation_format == "rot6d"
    assert spec.names[:3] == ["left_pos_x", "left_pos_y", "left_pos_z"]
    assert spec.names[10:13] == ["right_pos_x", "right_pos_y", "right_pos_z"]


def test_mix4_sampling_audit_matches_protocol() -> None:
    counts = {
        "clean": 475122,
        "perturbed": 1000000,
        "random_feasible": 2700000,
        "counterfactual_replay": 957810,
        "exploration": 239844,
    }
    dataset = ActionFollowingLeRobotDataset.__new__(ActionFollowingLeRobotDataset)
    dataset.protocol = "mix4"
    dataset.seed = 20260717
    dataset._sampling_epoch = 0
    dataset._family_by_source = list(counts)
    dataset._episode_records = [
        (source_index, 0, count, source_index) for source_index, count in enumerate(counts.values())
    ]
    dataset._num_valid_indices = sum(counts.values())
    dataset._sample_mass_by_family = {}
    dataset._record_mass_cum_ends = []
    dataset._record_families = []
    dataset._total_sampling_mass = 0.0

    dataset._build_protocol_sampling_index()
    audit = dataset.audit_sampling(num_samples=100000, seed=20260717)

    assert audit["max_abs_error"] <= 0.001
    assert audit["target_family_probabilities"] == {
        "clean": 0.5,
        "perturbed": 0.125,
        "random_feasible": 0.125,
        "counterfactual_replay": 0.125,
        "exploration": 0.125,
    }


def test_split_video_path_falls_back_to_full_root(tmp_path, monkeypatch) -> None:
    split_root = tmp_path / "train"
    relative = "exploration/tasks/turn_switch/videos/observation.images.cam_high/chunk-000/file-000003.mp4"
    split_video = split_root / relative
    full_video = tmp_path / relative
    full_video.parent.mkdir(parents=True)
    full_video.touch()
    monkeypatch.setenv("AFD_ROOT", str(split_root))
    monkeypatch.delenv("AFD_VIDEO_FALLBACK_ROOTS", raising=False)

    assert _resolve_actionfollowing_video_path(split_video) == full_video


def test_broken_absolute_video_symlink_uses_mount_prefix_remap(tmp_path, monkeypatch) -> None:
    split_root = tmp_path / "train"
    broken_link = split_root / "exploration/tasks/turn_switch/video.mp4"
    broken_link.parent.mkdir(parents=True)
    broken_link.symlink_to("/old/data/ActionFollowingData/enhanced/sample/video.mp4")
    remapped = tmp_path / "mounted" / "ActionFollowingData/enhanced/sample/video.mp4"
    remapped.parent.mkdir(parents=True)
    remapped.touch()
    monkeypatch.setenv("AFD_ROOT", str(split_root))
    monkeypatch.setenv("AFD_VIDEO_SYMLINK_PREFIX_REMAP", f"/old/data={tmp_path / 'mounted'}")

    assert _resolve_actionfollowing_video_path(broken_link) == remapped
