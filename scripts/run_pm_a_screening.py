#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import load_jsonl
from policybraid.verifiers import validate_candidate, verify_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blinded PM-A0 A100 screening")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-verification", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "pm_a.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_a_candidates.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-device", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def _configure_environment() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _validate_inputs(config: dict, candidates: list[dict], verification: dict) -> None:
    workload = config["workload"]
    expected = workload["candidates_per_family"] * len(workload["families"])
    if len(candidates) != expected:
        raise ValueError(f"expected {expected} candidates, found {len(candidates)}")
    for candidate in candidates:
        validate_candidate(candidate)
    if verification.get("status") != "success":
        raise ValueError("model manifest verification did not pass")
    if verification.get("asset_revision") != config["model"]["revision"]:
        raise ValueError("model verification revision differs from PM-A freeze")
    if config["sampling"]["primary"] != {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_continuation_tokens": 128,
        "group_size": 8,
    }:
        raise ValueError("primary PM-A sampling configuration changed")


def _render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = load_jsonl(args.candidates)
    verification = json.loads(args.model_verification.read_text(encoding="utf-8"))
    _validate_inputs(config, candidates, verification)
    if _sha256(args.candidates) != config["workload"]["candidate_file_sha256"]:
        raise ValueError("candidate file differs from the PM-A hash lock")
    _configure_environment()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    environment = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_root": str(Path(vllm.__file__).resolve().parents[1]),
        "physical_device": args.physical_device,
        "device_info": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    primary = config["sampling"]["primary"]
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=int(config["workload"]["screening_seed"]),
        max_model_len=2048,
        max_num_seqs=primary["group_size"],
        max_num_batched_tokens=8192,
        kv_cache_memory_bytes=2147483648,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        logprobs_mode="processed_logprobs",
    )
    started = time.perf_counter()
    rows = []
    try:
        tokenizer = llm.get_tokenizer()
        seed_base = int(config["workload"]["screening_seed"])
        for candidate_index, candidate in enumerate(candidates):
            rendered = _render_prompt(tokenizer, candidate["prompt"])
            prompts = [rendered] * primary["group_size"]
            parameters = [
                SamplingParams(
                    temperature=primary["temperature"],
                    top_p=primary["top_p"],
                    top_k=primary["top_k"],
                    max_tokens=primary["max_continuation_tokens"],
                    ignore_eos=False,
                    detokenize=True,
                    seed=seed_base + candidate_index * 16 + sample_id,
                )
                for sample_id in range(primary["group_size"])
            ]
            outputs = llm.generate(prompts, parameters, use_tqdm=False)
            if len(outputs) != primary["group_size"]:
                raise RuntimeError("engine returned the wrong screening group size")
            samples = []
            for sample_id, output in enumerate(outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("screening request returned multiple sequences")
                sequence = output.outputs[0]
                text = sequence.text
                try:
                    decision = verify_output(candidate, text)
                    verifier_status = "completed"
                    reward = decision.reward
                    reason = decision.reason
                except Exception as error:  # verifier failure is evidence, not a retry
                    verifier_status = "error"
                    reward = None
                    reason = f"{type(error).__name__}: {error}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "seed_id": seed_base + candidate_index * 16 + sample_id,
                        "token_ids": [int(token) for token in sequence.token_ids],
                        "text": text,
                        "finish_reason": sequence.finish_reason,
                        "reward": reward,
                        "verifier_status": verifier_status,
                        "verifier_reason": reason,
                    }
                )
            rows.append(
                {
                    "prompt_id": candidate["prompt_id"],
                    "task_family": candidate["task_family"],
                    "samples": samples,
                }
            )
    finally:
        shutdown = getattr(llm.llm_engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    verifier_errors = sum(
        sample["verifier_status"] != "completed"
        for row in rows
        for sample in row["samples"]
    )
    return (
        {
            "schema": "policybraid.pm_a.a100_screening.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": _repository_commit(),
            "config_sha256": _sha256(args.config),
            "candidates_sha256": _sha256(args.candidates),
            "model_verification_sha256": _sha256(args.model_verification),
            "platform": "a100",
            "execution_context": {
                "batch_composition": "one_candidate_group_per_engine_call",
                "batch_size": primary["group_size"],
                "request_order": "prompt_id_then_sample_id",
                "mode": "normal",
                "tensor_parallel_size": 1,
                "packing_schedule": "disabled",
            },
            "environment": environment,
            "candidate_count": len(rows),
            "samples_per_candidate": primary["group_size"],
            "verifier_error_count": verifier_errors,
            "elapsed_seconds": time.perf_counter() - started,
            "candidates": rows,
            "status": "success" if verifier_errors == 0 else "incomplete",
        },
        0 if verifier_errors == 0 else 3,
    )


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        payload, exit_code = execute(args)
    except Exception as error:
        payload = {
            "schema": "policybraid.pm_a.a100_screening.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": _repository_commit(),
            "platform": "a100",
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
