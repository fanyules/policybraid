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
    parser = argparse.ArgumentParser(description="Run one formal PM-AR2 rollout process")
    parser.add_argument("--platform", choices=("a100", "910b"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-verification", type=Path, required=True)
    parser.add_argument("--restart-index", type=int, choices=range(5), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pm_ar2.json"
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "pm_ar.json",
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


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def find_git_identity(source: str | None) -> dict[str, str | None]:
    if source is None:
        return {"source": None, "git_root": None, "git_commit": None}
    source_path = Path(source).resolve()
    for parent in (source_path.parent, *source_path.parents):
        commit = git_commit(parent)
        if commit is not None:
            return {"source": str(source_path), "git_root": str(parent), "git_commit": commit}
    return {"source": str(source_path), "git_root": None, "git_commit": None}


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def configure_environment() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def loaded_runner_identities(platform: str) -> list[dict[str, str | None]]:
    prefixes = (
        ("vllm.v1.worker", "vllm.model_executor")
        if platform == "a100"
        else ("vllm_ascend.worker",)
    )
    identities = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(prefixes) or "model_runner" not in name:
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        identities.append({"module": name, **find_git_identity(source)})
    if not identities:
        raise RuntimeError("no loaded model-runner module was found after engine init")
    return identities


def prompt_tokens(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        value = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    return token_ids_from_chat_template(value)


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


def execute(args: argparse.Namespace) -> tuple[dict, int]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_config.read_text(encoding="utf-8"))
    verification = json.loads(args.model_verification.read_text(encoding="utf-8"))
    workload = load_jsonl(args.workload)
    if sha256(args.workload) != config["workload"]["sha256"]:
        raise ValueError("workload differs from PM-AR2 lock")
    if len(workload) != config["workload"]["prompts"]:
        raise ValueError("workload prompt count differs from PM-AR2")
    if verification.get("status") != "success":
        raise ValueError("model verification did not pass")
    if verification.get("asset_revision") != config["model"]["revision"]:
        raise ValueError("model revision differs from PM-AR2")
    configure_environment()

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    if args.platform == "910b":
        import torch_npu
        import vllm_ascend

    seed_base = config["restart_seed_bases"][args.restart_index]
    environment = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "physical_device": args.physical_device,
        "vllm_platform": str(current_platform),
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
        environment.update(
            {
                "torch_npu_version": torch_npu.__version__,
                "vllm_ascend_version": importlib.metadata.version("vllm-ascend"),
                "vllm_ascend_package": str(Path(vllm_ascend.__file__).resolve()),
                "device_info": command_output(["npu-smi", "info"]),
            }
        )

    sampling = config["sampling"]
    engine_kwargs = {
        "model": str(args.model),
        "tensor_parallel_size": config["model"]["tensor_parallel_size"],
        "dtype": "bfloat16",
        "seed": seed_base,
        "max_model_len": parent["learner"]["maximum_sequence_tokens"],
        "max_num_seqs": sampling["group_size"],
        "max_num_batched_tokens": 8192,
        "kv_cache_memory_bytes": 2147483648,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": True,
        "logprobs_mode": "processed_logprobs",
    }
    if args.platform == "910b":
        engine_kwargs["additional_config"] = {
            "enable_cpu_binding": True,
            "enable_flashcomm1": False,
        }
    llm = LLM(**engine_kwargs)
    started = time.perf_counter()
    records = []
    try:
        top_level_mode = llm.llm_engine.vllm_config.model_config.logprobs_mode
        if top_level_mode != "processed_logprobs":
            raise RuntimeError(f"unexpected top-level logprobs mode: {top_level_mode}")
        runner_identities = loaded_runner_identities(args.platform)
        tokenizer = llm.get_tokenizer()
        for prompt_index, candidate in enumerate(workload):
            prefix = prompt_tokens(tokenizer, candidate["prompt"])
            prompts = [{"prompt_token_ids": prefix} for _ in range(sampling["group_size"])]
            params = [
                SamplingParams(
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    top_k=sampling["top_k"],
                    max_tokens=sampling["maximum_continuation_tokens"],
                    ignore_eos=False,
                    detokenize=True,
                    logprobs=0,
                    seed=seed_base + prompt_index * 16 + sample_id,
                )
                for sample_id in range(sampling["group_size"])
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            samples = []
            for sample_id, output in enumerate(outputs):
                sequence = output.outputs[0]
                tokens = [int(token) for token in sequence.token_ids]
                if not tokens:
                    raise RuntimeError("formal rollout returned an empty continuation")
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
                    "group_id": f"{candidate['prompt_id']}:{args.platform}:r{args.restart_index}",
                    "source_backend": args.platform,
                    "restart_id": args.restart_index,
                    "physical_device": args.physical_device,
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

    return (
        {
            "schema": "policybraid.pm_ar.cube_rollout.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "config_sha256": sha256(args.config),
            "parent_config_sha256": sha256(args.parent_config),
            "workload_sha256": sha256(args.workload),
            "model_verification_sha256": sha256(args.model_verification),
            "platform": args.platform,
            "restart_index": args.restart_index,
            "seed_base": seed_base,
            "policy_version": parent["model"]["policy_version"],
            "execution_context": {
                **parent["execution_context"],
                "top_level_logprobs_mode": top_level_mode,
                "loaded_runner_identities": runner_identities,
            },
            "sampling": sampling,
            "environment": environment,
            "prompt_count": len(records),
            "trajectory_count": sum(len(record["samples"]) for record in records),
            "nonzero_reward_variance_groups": sum(
                len({sample["reward"] for sample in record["samples"]}) > 1
                for record in records
            ),
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
            "schema": "policybraid.pm_ar.cube_rollout.v1",
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
