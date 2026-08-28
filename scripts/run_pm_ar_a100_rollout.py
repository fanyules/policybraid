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

from policybraid.prompts import load_jsonl
from policybraid.tokenization import token_ids_from_chat_template
from policybraid.verifiers import verify_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PM-AR1 A100 resample set")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-verification", type=Path, required=True)
    parser.add_argument("--seed-group", type=int, choices=(0, 1), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pm_ar.json"
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_ar_selected.jsonl",
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
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def prompt_tokens(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    return token_ids_from_chat_template(tokens)


def sampled_logprobs(sequence) -> list[float]:
    if sequence.logprobs is None or len(sequence.logprobs) != len(sequence.token_ids):
        raise RuntimeError("engine omitted sampled-token logprobs")
    values = []
    for step, token_id in enumerate(sequence.token_ids):
        entry = sequence.logprobs[step].get(int(token_id))
        if entry is None:
            raise RuntimeError(f"sampled token {token_id} absent at step {step}")
        value = float(entry.logprob)
        if not math.isfinite(value):
            raise RuntimeError("engine returned a non-finite behavior logprob")
        values.append(value)
    return values


def validate_inputs(config: dict, workload_path: Path, verification: dict) -> None:
    if sha256(workload_path) != config["workload"]["selected_sha256"]:
        raise ValueError("selected workload differs from the PM-AR hash lock")
    if verification.get("status") != "success":
        raise ValueError("model verification did not pass")
    if verification.get("asset_revision") != config["model"]["revision"]:
        raise ValueError("model verification revision differs from PM-AR")


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workload = load_jsonl(args.workload)
    verification = json.loads(args.model_verification.read_text(encoding="utf-8"))
    validate_inputs(config, args.workload, verification)
    if len(workload) != config["workload"]["total_prompts"]:
        raise ValueError("selected workload has the wrong prompt count")
    configure_environment()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    sampling = config["sampling"]["primary"]
    seed_base = config["pm_ar1_noise_lock"]["resample_seed_bases"][args.seed_group]
    environment = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_root": str(Path(vllm.__file__).resolve().parents[1]),
        "physical_device": args.physical_device,
        "device_info": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=seed_base,
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
    records = []
    try:
        active_mode = llm.llm_engine.vllm_config.model_config.logprobs_mode
        if active_mode != "processed_logprobs":
            raise RuntimeError(f"unexpected logprobs mode: {active_mode}")
        tokenizer = llm.get_tokenizer()
        for prompt_index, candidate in enumerate(workload):
            prefix = prompt_tokens(tokenizer, candidate["prompt"])
            if not prefix:
                raise RuntimeError(f"empty tokenized prompt: {candidate['prompt_id']}")
            prompts = [{"prompt_token_ids": prefix} for _ in range(sampling["group_size"])]
            parameters = [
                SamplingParams(
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    top_k=sampling["top_k"],
                    max_tokens=sampling["max_continuation_tokens"],
                    ignore_eos=False,
                    detokenize=True,
                    logprobs=0,
                    seed=seed_base + prompt_index * 16 + sample_id,
                )
                for sample_id in range(sampling["group_size"])
            ]
            outputs = llm.generate(prompts, parameters, use_tqdm=False)
            if len(outputs) != sampling["group_size"]:
                raise RuntimeError("engine returned an incomplete GRPO group")
            samples = []
            for sample_id, output in enumerate(outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("rollout request returned multiple sequences")
                sequence = output.outputs[0]
                tokens = [int(token) for token in sequence.token_ids]
                if not tokens:
                    raise RuntimeError("rollout returned an empty continuation")
                decision = verify_output(candidate, sequence.text)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "seed_id": seed_base + prompt_index * 16 + sample_id,
                        "token_ids": tokens,
                        "processed_behavior_logprobs": sampled_logprobs(sequence),
                        "text": sequence.text,
                        "finish_reason": str(sequence.finish_reason),
                        "reward": float(decision.reward),
                        "verifier_reason": decision.reason,
                    }
                )
            records.append(
                {
                    "prompt_id": candidate["prompt_id"],
                    "task_family": candidate["task_family"],
                    "group_id": f"{candidate['prompt_id']}:a100:s{args.seed_group}",
                    "prompt_token_ids": prefix,
                    "samples": samples,
                }
            )
    finally:
        shutdown = getattr(llm.llm_engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    nonzero_variance = sum(
        len({sample["reward"] for sample in record["samples"]}) > 1
        for record in records
    )
    return (
        {
            "schema": "policybraid.pm_ar.a100_rollout.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "config_sha256": sha256(args.config),
            "workload_sha256": sha256(args.workload),
            "model_verification_sha256": sha256(args.model_verification),
            "platform": "a100",
            "seed_group": args.seed_group,
            "seed_base": seed_base,
            "policy_version": config["model"]["policy_version"],
            "execution_context": config["execution_context"],
            "sampling": sampling,
            "environment": environment,
            "prompt_count": len(records),
            "trajectory_count": sum(len(record["samples"]) for record in records),
            "nonzero_reward_variance_groups": nonzero_variance,
            "elapsed_seconds": time.perf_counter() - started,
            "records": records,
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
            "schema": "policybraid.pm_ar.a100_rollout.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "platform": "a100",
            "seed_group": args.seed_group,
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
