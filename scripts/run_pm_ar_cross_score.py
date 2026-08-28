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
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.cross_score import (
    CROSS_SCORE_ARG,
    start_capture,
    stop_capture,
    take_capture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode-context full-support cross-score for PM-AR histories"
    )
    parser.add_argument("--platform", choices=("a100", "910b"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--restart-index", type=int, choices=range(5), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pm_ar.json"
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


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sampling = config["sampling"]["primary"]
    if (sampling["temperature"], sampling["top_p"], sampling["top_k"]) != (
        1.0,
        1.0,
        0,
    ):
        raise ValueError("PM-AR cross-score requires the primary full-support sampler")
    trajectory_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.trajectories
    ]
    for path, payload in zip(args.trajectories, trajectory_payloads, strict=True):
        if payload.get("status") != "success":
            raise ValueError(f"trajectory is not successful: {path}")
        if payload.get("workload_sha256") != config["workload"]["selected_sha256"]:
            raise ValueError(f"trajectory workload differs from PM-AR: {path}")
    configure_environment()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if args.platform == "910b":
        import torch_npu

    environment = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "physical_device": args.physical_device,
    }
    if args.platform == "a100":
        environment["device_info"] = command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        )
    else:
        environment["torch_npu_version"] = torch_npu.__version__
        environment["vllm_ascend_version"] = importlib.metadata.version("vllm-ascend")
        environment["device_info"] = command_output(["npu-smi", "info"])

    engine_kwargs = {
        "model": str(args.model),
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "seed": 24500000 + args.restart_index,
        "max_model_len": config["learner"]["maximum_sequence_tokens"],
        "max_num_seqs": sampling["group_size"],
        "max_num_batched_tokens": 8192,
        "kv_cache_memory_bytes": 2147483648,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": True,
        "logprobs_mode": "processed_logprobs",
        "logits_processors": [
            "policybraid.cross_score:CrossScoreLogitsProcessor"
        ],
    }
    if args.platform == "910b":
        engine_kwargs["additional_config"] = {
            "enable_cpu_binding": True,
            "enable_flashcomm1": False,
        }
    llm = LLM(**engine_kwargs)

    expected_keys = []
    source_records = []
    for source_index, payload in enumerate(trajectory_payloads):
        for record in payload["records"]:
            for sample in record["samples"]:
                key = f"s{source_index}:{record['prompt_id']}:{sample['sample_id']}"
                expected_keys.append(key)
                source_records.append((source_index, record, sample, key))
    start_capture(expected_keys)
    started = time.perf_counter()
    forced_outputs: dict[str, list[int]] = {}
    try:
        for source_index, payload in enumerate(trajectory_payloads):
            for record in payload["records"]:
                prompts = []
                params = []
                keys = []
                for sample in sorted(
                    record["samples"], key=lambda item: int(item["sample_id"])
                ):
                    key = f"s{source_index}:{record['prompt_id']}:{sample['sample_id']}"
                    keys.append(key)
                    prompts.append({"prompt_token_ids": record["prompt_token_ids"]})
                    params.append(
                        SamplingParams(
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            max_tokens=len(sample["token_ids"]),
                            ignore_eos=True,
                            detokenize=False,
                            seed=24500000
                            + args.restart_index * 100000
                            + len(forced_outputs),
                            extra_args={
                                CROSS_SCORE_ARG: {
                                    "key": key,
                                    "token_ids": sample["token_ids"],
                                }
                            },
                        )
                    )
                outputs = llm.generate(prompts, params, use_tqdm=False)
                for key, sample, output in zip(
                    keys,
                    sorted(record["samples"], key=lambda item: int(item["sample_id"])),
                    outputs,
                    strict=True,
                ):
                    sequence = output.outputs[0]
                    observed = [int(token) for token in sequence.token_ids]
                    if observed != sample["token_ids"]:
                        raise RuntimeError(f"forced replay diverged for {key}")
                    forced_outputs[key] = observed
        captures = take_capture()
    except Exception:
        stop_capture()
        raise
    finally:
        shutdown = getattr(llm.llm_engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    output_sources = []
    maximum_self_gap = None
    for source_index, payload in enumerate(trajectory_payloads):
        output_records = []
        for record in payload["records"]:
            output_samples = []
            for sample in sorted(
                record["samples"], key=lambda item: int(item["sample_id"])
            ):
                key = f"s{source_index}:{record['prompt_id']}:{sample['sample_id']}"
                values = captures[key]
                if len(values) != len(sample["token_ids"]) or not all(
                    math.isfinite(value) for value in values
                ):
                    raise RuntimeError(f"invalid capture length or value for {key}")
                self_gap = None
                if payload.get("platform") == args.platform:
                    self_gap = max(
                        abs(left - right)
                        for left, right in zip(
                            values,
                            sample["processed_behavior_logprobs"],
                            strict=True,
                        )
                    )
                    maximum_self_gap = (
                        self_gap
                        if maximum_self_gap is None
                        else max(maximum_self_gap, self_gap)
                    )
                output_samples.append(
                    {
                        "sample_id": sample["sample_id"],
                        "token_ids": sample["token_ids"],
                        "cross_scored_logprobs": values,
                        "self_reporter_max_abs_gap": self_gap,
                    }
                )
            output_records.append(
                {
                    "prompt_id": record["prompt_id"],
                    "task_family": record["task_family"],
                    "samples": output_samples,
                }
            )
        output_sources.append(
            {
                "source_index": source_index,
                "source_platform": payload.get("platform"),
                "source_restart_index": payload.get("restart_index"),
                "source_seed_group": payload.get("seed_group"),
                "records": output_records,
            }
        )
    return (
        {
            "schema": "policybraid.pm_ar.decode_cross_score.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "config_sha256": sha256(args.config),
            "trajectory_sha256": [sha256(path) for path in args.trajectories],
            "platform": args.platform,
            "restart_index": args.restart_index,
            "environment": environment,
            "execution_context": {
                "path": "decode_with_pre_force_full_support_capture",
                "batch_composition": "one_complete_group_per_engine_call",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
            },
            "source_count": len(output_sources),
            "trajectory_count": len(expected_keys),
            "maximum_self_reporter_abs_gap": maximum_self_gap,
            "elapsed_seconds": time.perf_counter() - started,
            "sources": output_sources,
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
            "schema": "policybraid.pm_ar.decode_cross_score.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "platform": args.platform,
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

