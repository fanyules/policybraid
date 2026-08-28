#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-score PM-AR anchor histories in one fresh A100 engine"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--restart-index", type=int, choices=range(5), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pm_ar.json"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "pm_ar" / "PM_AR0_SELECTION.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def configure_environment() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def prompt_token_logprobs(output, input_token_ids: list[int]) -> list[float]:
    entries = output.prompt_logprobs
    if entries is None or len(entries) != len(input_token_ids):
        raise RuntimeError("engine omitted prompt logprobs or returned the wrong length")
    values = [math.nan]
    for position in range(1, len(input_token_ids)):
        token_id = input_token_ids[position]
        token_entry = entries[position]
        if token_entry is None or token_id not in token_entry:
            raise RuntimeError(f"prompt token {token_id} absent at position {position}")
        value = float(token_entry[token_id].logprob)
        if not math.isfinite(value):
            raise RuntimeError("engine returned a non-finite prompt logprob")
        values.append(value)
    return values


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if trajectory.get("status") != "success" or trajectory.get("platform") != "a100":
        raise ValueError("anchor source trajectory is not a successful A100 rollout")
    if trajectory.get("workload_sha256") != config["workload"]["selected_sha256"]:
        raise ValueError("trajectory workload differs from the PM-AR lock")
    anchor_ids = {
        prompt_id
        for ids in selection["anchor_ids_by_family"].values()
        for prompt_id in ids
    }
    if len(anchor_ids) != config["pm_ar1_noise_lock"]["anchor_prompts_total"]:
        raise ValueError("selection manifest does not contain 32 unique anchors")
    records_by_id = {record["prompt_id"]: record for record in trajectory["records"]}
    if not anchor_ids <= set(records_by_id):
        raise ValueError("trajectory is missing at least one anchor prompt")
    configure_environment()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    environment = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "physical_device": args.physical_device,
        "device_info": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    sampling = config["sampling"]["primary"]
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=24300000 + args.restart_index,
        max_model_len=config["learner"]["maximum_sequence_tokens"],
        max_num_seqs=sampling["group_size"],
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=2147483648,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        logprobs_mode="processed_logprobs",
    )
    started = time.perf_counter()
    scored_records = []
    try:
        for prompt_id in sorted(anchor_ids):
            source = records_by_id[prompt_id]
            prompts = []
            params = []
            combined_by_sample = []
            for sample in source["samples"]:
                combined = [*source["prompt_token_ids"], *sample["token_ids"]]
                if len(combined) > config["learner"]["maximum_sequence_tokens"]:
                    raise RuntimeError(f"anchor sequence exceeds token limit: {prompt_id}")
                combined_by_sample.append(combined)
                prompts.append({"prompt_token_ids": combined})
                params.append(
                    SamplingParams(
                        temperature=sampling["temperature"],
                        top_p=sampling["top_p"],
                        top_k=sampling["top_k"],
                        max_tokens=1,
                        ignore_eos=True,
                        detokenize=False,
                        logprobs=0,
                        prompt_logprobs=0,
                        seed=24300000
                        + args.restart_index * 10000
                        + len(scored_records) * 16
                        + int(sample["sample_id"]),
                    )
                )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(source["samples"]):
                raise RuntimeError("scoring engine returned an incomplete anchor group")
            scored_samples = []
            for sample, combined, output in zip(
                source["samples"], combined_by_sample, outputs, strict=True
            ):
                all_logprobs = prompt_token_logprobs(output, combined)
                start = len(source["prompt_token_ids"])
                behavior = [float(value) for value in all_logprobs[start:]]
                if len(behavior) != len(sample["token_ids"]):
                    raise RuntimeError("anchor score length mismatch")
                source_behavior = sample["processed_behavior_logprobs"]
                scored_samples.append(
                    {
                        "sample_id": sample["sample_id"],
                        "token_ids": sample["token_ids"],
                        "processed_behavior_logprobs": behavior,
                        "max_abs_gap_from_rollout_reporter": max(
                            abs(left - right)
                            for left, right in zip(behavior, source_behavior, strict=True)
                        ),
                    }
                )
            scored_records.append(
                {
                    "prompt_id": prompt_id,
                    "task_family": source["task_family"],
                    "samples": scored_samples,
                }
            )
    finally:
        shutdown = getattr(llm.llm_engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return (
        {
            "schema": "policybraid.pm_ar.a100_scoring.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "config_sha256": sha256(args.config),
            "selection_sha256": sha256(args.selection),
            "trajectory_sha256": sha256(args.trajectory),
            "platform": "a100",
            "restart_index": args.restart_index,
            "environment": environment,
            "anchor_count": len(scored_records),
            "trajectory_count": sum(len(record["samples"]) for record in scored_records),
            "maximum_abs_gap_from_rollout_reporter": max(
                sample["max_abs_gap_from_rollout_reporter"]
                for record in scored_records
                for sample in record["samples"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "records": scored_records,
            "status": "success",
        },
        0,
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        payload, exit_code = execute(args)
    except Exception as error:
        payload = {
            "schema": "policybraid.pm_ar.a100_scoring.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "platform": "a100",
            "restart_index": args.restart_index,
            "status": "incomplete",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 3
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

