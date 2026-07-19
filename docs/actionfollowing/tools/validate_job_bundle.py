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
    "memory": 970,
    "rdma/hca": 1,
    "sharedMemory": 120,
}

EXPECTED_COUNTS = {
    "clean": 475122,
    "perturbed": 250000,
    "random_feasible": 1350000,
    "counterfactual_replay": 474645,
    "exploration": 121071,
}

CANONICAL_DATA_ROOT = (
    "/mnt/dataset/sixiangchen_workspace/Ideas/data/"
    "ActionFollowingData_LeRobot_Rot6D/train"
)
VIDEO_SYMLINK_REMAP = (
    "AFD_VIDEO_SYMLINK_PREFIX_REMAP=/mnt/dataset/csx_workspace/Ideas/data="
    "/mnt/dataset/sixiangchen_workspace/Ideas/data"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--model", choices=("cosmos3", "cosmos25"), required=True)
    parser.add_argument("--steps", type=int, choices=(20, 40000), required=True)
    parser.add_argument("--queue", default="train22")
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
    require(resources == EXPECTED_RESOURCES, f"unexpected resources: {resources}")

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
    data_mount = mounts.get("/mnt/dataset/sixiangchen_workspace")
    require(data_mount is not None, "canonical ActionFollowingData mount is missing")
    require(data_mount.get("sourcePath") == "/damoxing/sixiangchen-fileset", "unexpected data mount source")
    require(data_mount.get("options", {}).get("readOnly") is True, "canonical data mount must be read-only")
    require("/mnt/dataset/public_data" in mounts, "public model-asset mount is missing")

    compact_script = re.sub(r"\s|_", "", script_text)
    compact_data_root = re.sub(r"\s|_", "", CANONICAL_DATA_ROOT)
    require(f"AFDROOT={compact_data_root}" in compact_script, "script does not use the canonical train root")
    require(VIDEO_SYMLINK_REMAP in script_text, "broken enhanced-video symlink remap is missing")
    require('forprotocolin("clean","mix4")' in compact_script, "script must audit both clean and mix4")
    require("auditnumsamples=100000" in compact_script, "script must run a 100k sampling audit")
    require(
        'asserttuple(item["action"].shape)==(32,20)' in compact_script,
        "script must assert canonical Rot6D20 action chunks",
    )
    for family, count in EXPECTED_COUNTS.items():
        compact_family = family.replace("_", "")
        require(f'"{compact_family}":{count}' in compact_script, f"script lacks effective count for {family}")

    expected_views = (
        '["camhigh","camleftwrist","camrightwrist"]'
        if args.model == "cosmos3"
        else '["camhigh"]'
    )
    require(f'"views":{expected_views}' in compact_script, f"script lacks canonical {args.model} views")

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
        require("SMOKE_RESULT.txt" in script_text, "smoke must write a result record")
        require("effective_global_batch=%s" in script_text, "result record must include effective batch")
    else:
        require("20step" not in name.lower(), "40k job must not use a smoke name")

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
                "command": command,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
