#!/usr/bin/env python3
"""Validate an AIHC Cosmos ActionFollowing job JSON and launch script."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

EXPECTED_RESOURCES = {
    "baidu.com/a800_80g_cgpu": 8,
    "cpu": 123,
    "memory": 975,
    "rdma/hca": 1,
    "sharedMemory": 0,
}

QUEUE_RESOURCE_OVERRIDES = {
    # Live train21 nodes use the c128m2048 resource template. AIHC rejects
    # train/train22's c128m1024 quantities before creating a job.
    "train21": {"cpu": 122, "memory": 1960},
}

EXPECTED_COUNTS_BY_MODEL = {
    # Cosmos3 needs O[t]..O[t+32] for A[t]..A[t+31], so the terminal
    # action-only start of every trajectory episode is intentionally excluded.
    "cosmos3": {
        "clean": 472622,
        "perturbed": 250000,
        "random_feasible": 1345000,
        "counterfactual_replay": 472145,
        "exploration": 120821,
    },
    # Predict2.5 consumes the existing 32-action chunk contract directly.
    "cosmos25": {
        "clean": 475122,
        "perturbed": 250000,
        "random_feasible": 1350000,
        "counterfactual_replay": 474645,
        "exploration": 121071,
    },
}

CANONICAL_DATA_ROOT = "/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingData_LeRobot_Rot6D_nosymlink/train"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--model", choices=("cosmos3", "cosmos25"), required=True)
    parser.add_argument("--steps", type=int, choices=(20, 40000), required=True)
    parser.add_argument("--queue", default="train")
    parser.add_argument("--checkpoint-save-iter", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = json.loads(args.job_json.read_text())
    script_text = args.script.read_text()

    name = job.get("name", "")
    require(name.startswith("ACWM_"), "job name must start with ACWM_")
    require(job.get("queue") == args.queue, f"queue must be {args.queue}")
    spec = job.get("jobSpec", {})
    require(spec.get("replicas") == 1, "replicas must be 1")
    require(spec.get("enableRDMA") is True, "RDMA must be enabled")
    resources = {item["name"]: item["quantity"] for item in spec.get("resources", [])}
    expected_resources = EXPECTED_RESOURCES | QUEUE_RESOURCE_OVERRIDES.get(args.queue, {})
    require(resources == expected_resources, f"unexpected resources for {args.queue}: {resources}")

    envs = {item["name"]: item["value"] for item in spec.get("envs", [])}
    require(envs.get("AIHC_JOB_NAME") == name, "AIHC_JOB_NAME must equal job name")
    command = spec.get("command", "")
    require(args.script.name in command, "job command does not reference the supplied script")
    require(args.model in name.lower(), f"job name does not identify {args.model}")

    mounts = {item.get("mountPath"): item for item in job.get("datasources", [])}
    checkpoint_mount = mounts.get("/mnt/dataset/csx_ckp")
    require(checkpoint_mount is not None, "persistent checkpoint mount is missing")
    require(checkpoint_mount.get("name") == "pfs-Zx30ll", "checkpoint mount must use pfs-Zx30ll")
    require(not checkpoint_mount.get("options", {}).get("readOnly", False), "checkpoint mount must be writable")
    data_mount = mounts.get("/mnt/dataset/public_data")
    require(data_mount is not None, "canonical ActionFollowingData mount is missing")
    require(data_mount.get("sourcePath") == "/datasets", "unexpected data mount source")
    require(data_mount.get("options", {}).get("readOnly") is True, "canonical data mount must be read-only")

    compact_script = re.sub(r"\s|_", "", script_text)
    lower_compact_script = compact_script.lower()
    compact_data_root = re.sub(r"\s|_", "", CANONICAL_DATA_ROOT)
    require(f"AFDROOT={compact_data_root}" in compact_script, "script does not use the canonical train root")
    require(
        "ActionFollowingData_LeRobot_Rot6D_nosymlink" in script_text, "Motus-aligned symlink-free data root is missing"
    )
    require('forprotocolin("clean","mix4")' in compact_script, "script must audit both clean and mix4")
    require("auditnumsamples=100000" in compact_script, "script must run a 100k sampling audit")
    require(
        'asserttuple(item["action"].shape)==(32,20)' in compact_script,
        "script must assert canonical Rot6D20 action chunks",
    )
    for family, count in EXPECTED_COUNTS_BY_MODEL[args.model].items():
        compact_family = family.replace("_", "")
        require(f'"{compact_family}":{count}' in compact_script, f"script lacks effective count for {family}")

    expected_views = '["camhigh","camleftwrist","camrightwrist"]' if args.model == "cosmos3" else '["camhigh"]'
    require(f'"views":{expected_views}' in compact_script, f"script lacks canonical {args.model} views")
    if args.model == "cosmos3":
        prompt_manifest_candidates = (
            args.script.parent.parent / "assets" / "robotwin_50_full_descriptions.json",
            Path(__file__).parent.parent / "assets" / "robotwin_50_full_descriptions.json",
        )
        prompt_manifest_path = next((path for path in prompt_manifest_candidates if path.is_file()), None)
        require(
            prompt_manifest_path is not None,
            f"RoboTwin prompt manifest is missing; checked: {prompt_manifest_candidates}",
        )
        assert prompt_manifest_path is not None
        prompt_manifest = json.loads(prompt_manifest_path.read_text())
        require(prompt_manifest.get("schema_version") == 1, "unexpected RoboTwin prompt manifest schema")
        require(
            prompt_manifest.get("source_repository") == "https://github.com/RoboTwin-Platform/RoboTwin",
            "RoboTwin prompt manifest must identify the official source repository",
        )
        require(
            prompt_manifest.get("source_commit") == "c3ddfa8b97d5519efa828b075999bd0006778e5e",
            "RoboTwin prompt manifest source commit is not pinned",
        )
        full_descriptions = prompt_manifest.get("full_descriptions", {})
        require(len(full_descriptions) == 50, "RoboTwin prompt manifest must contain exactly 50 tasks")
        require(
            all(isinstance(value, str) and value.strip() for value in full_descriptions.values()),
            "RoboTwin prompt manifest contains an empty full_description",
        )
        require("future32" in name.lower(), "Cosmos3 job name must identify real future32 supervision")
        require("prompt" in name.lower(), "Cosmos3 job name must identify full-description prompting")
        require(
            "robotwinfulldescriptionmanifest" in lower_compact_script,
            "RoboTwin full_description manifest is missing",
        )
        require(
            "bootstraplogs" in lower_compact_script and "bootstraperror" in lower_compact_script,
            "Cosmos3 script must persist errors that occur before the main output directory is created",
        )
        require(
            'hfdatasetscache="$runtimecache/huggingface/datasets"' in lower_compact_script,
            "Cosmos3 script must redirect the Hugging Face datasets cache to the writable output mount",
        )
        require(
            '[[ -w "$HF_DATASETS_CACHE" ]]' in script_text,
            "Cosmos3 script must verify that its datasets cache is writable before data audit",
        )
        require(
            "__index_level_0__" in script_text and "metadata_prompt_columns" in script_text,
            "Cosmos3 script must audit the real LeRobot v2 tasks.parquet prompt column",
        )
        require('"timeline":"current1+future32"' in lower_compact_script, "current1+future32 audit evidence is missing")
        require(
            "len(deltatimestamps[actionfeature])==32" in lower_compact_script,
            "Cosmos3 script must assert 32 action timestamps",
        )
        require(
            "len(deltatimestamps[feature])==33" in lower_compact_script,
            "Cosmos3 script must assert 33 camera timestamps",
        )
        require(
            'item["aicaption"]==canonicalfulldescription(item["taskname"])' in lower_compact_script,
            "Cosmos3 script must compare the prompt with RoboTwin full_description",
        )

    step_pattern = re.compile(rf"trainer\.max_iter\s*=\s*{args.steps}\b")
    require(step_pattern.search(script_text) is not None, f"script lacks explicit max_iter={args.steps}")
    if args.steps == 20:
        require("20step" in name.lower(), "smoke job name must contain 20step")
        require("checkpoint.save_iter=20" in script_text, "smoke script must save at step 20")
        require("runtrain2bs16" in compact_script, "smoke must try global batch 16 first")
        require("runtrain1bs8" in compact_script, "smoke lacks the batch-8 OOM fallback")
        require("CUDAoutofmemory" in compact_script, "batch fallback must detect CUDA OOM")
        require("OutOfMemoryError" in compact_script, "batch fallback must detect PyTorch OOM")
        require("latest_checkpoint.txt" in script_text, "smoke must verify the latest checkpoint marker")
        require("iter_000000020" in script_text, "smoke must require the exact step-20 checkpoint")
        require(
            "expected exactly rank-0 optimizer steps 1..20" in script_text,
            "smoke must require exactly 20 optimizer steps",
        )
        require("math.isfinite" in script_text, "smoke must require finite optimizer losses")
        require("SMOKE_RESULT.txt" in script_text, "smoke must write a result record")
        require("effective_global_batch=%s" in script_text, "result record must include effective batch")
    else:
        require("20step" not in name.lower(), "40k job must not use a smoke name")
        require("40000" in name.lower(), "40k job name must identify 40000 steps")
        checkpoint_save_iter = args.checkpoint_save_iter or 10000
        require(
            f"checkpoint.save_iter={checkpoint_save_iter}" in script_text,
            f"40k script must checkpoint every {checkpoint_save_iter} steps",
        )
        require("scheduler.cyclelengths=[40000]" in compact_script, "40k scheduler cycle must be 40000")
        require("scheduler.warmupsteps=[1000]" in compact_script, "40k scheduler warmup must be 1000")
        require("runtrain2bs16" in compact_script, "40k run must try global batch 16 first")
        require("runtrain1bs8" in compact_script, "40k run lacks the batch-8 OOM fallback")
        require("CUDAoutofmemory" in compact_script, "40k batch fallback must detect CUDA OOM")
        require("OutOfMemoryError" in compact_script, "40k batch fallback must detect PyTorch OOM")
        require("latest_checkpoint.txt" in script_text, "40k run must verify the latest checkpoint marker")
        require("iter_000040000" in script_text, "40k run must require the exact step-40000 checkpoint")
        require(
            "expected exactly rank-0 optimizer steps 1..40000" in script_text,
            "40k run must require exactly 40000 optimizer steps",
        )
        require("iter_speed" in script_text, "40k audit must parse post-warmup per-step losses")
        require("math.isfinite" in script_text, "40k run must require finite optimizer losses")
        require("TRAIN_RESULT.txt" in script_text, "40k run must write a final result record")
        require("effective_global_batch=%s" in script_text, "40k result record must include effective batch")
        require("RUN_PROVENANCE.txt" in script_text, "40k run must persist code and bundle provenance")
        require("train_40000" in script_text, "40k run must use a dedicated non-smoke output root")
        require("refusing to reuse non-empty 40k output" in script_text, "40k output must be overwrite-safe")

    subprocess.run(["bash", "-n", str(args.script)], check=True)
    print(
        json.dumps(
            {
                "status": "valid",
                "name": name,
                "model": args.model,
                "queue": args.queue,
                "steps": args.steps,
                "resources": resources,
                "checkpoint_mount": checkpoint_mount["sourcePath"],
                "checkpoint_save_iter": args.checkpoint_save_iter or (20 if args.steps == 20 else 10000),
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
