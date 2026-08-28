#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.grpo import group_advantages
from policybraid.lora import (
    flatten_gradients,
    inject_lora,
    parameter_manifest,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute canonical PM-AR A100 LoRA GRPO gradients"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-verification", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--behavior-scores", type=Path, nargs="*", default=[])
    parser.add_argument("--scope", choices=("all", "anchors"), required=True)
    parser.add_argument("--run-index", type=int, choices=range(5), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--save-prompt-gradients", action="store_true")
    parser.add_argument("--output-tensor", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
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
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def score_override(path: Path) -> tuple[str, dict[tuple[str, int], list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success" or payload.get("platform") != "a100":
        raise ValueError(f"behavior score is not a successful A100 result: {path}")
    mapping = {}
    for record in payload["records"]:
        for sample in record["samples"]:
            key = (record["prompt_id"], int(sample["sample_id"]))
            if key in mapping:
                raise ValueError(f"duplicate behavior score key: {key}")
            mapping[key] = [float(value) for value in sample["processed_behavior_logprobs"]]
    return f"a100_scoring_r{payload['restart_index']}", mapping


def source_scores(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[float]]:
    return {
        (record["prompt_id"], int(sample["sample_id"])): [
            float(value) for value in sample["processed_behavior_logprobs"]
        ]
        for record in records
        for sample in record["samples"]
    }


def select_records(
    trajectory: dict[str, Any], selection: dict[str, Any], scope: str
) -> list[dict[str, Any]]:
    records = trajectory["records"]
    if scope == "all":
        return records
    anchor_ids = {
        prompt_id
        for ids in selection["anchor_ids_by_family"].values()
        for prompt_id in ids
    }
    selected = [record for record in records if record["prompt_id"] in anchor_ids]
    if len(selected) != 32:
        raise ValueError("anchor scope did not select 32 prompts")
    return selected


def left_padded_batch(torch, sequences: list[list[int]], pad_token_id: int):
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_masks = []
    for sequence in sequences:
        padding = maximum - len(sequence)
        input_ids.append([pad_token_id] * padding + sequence)
        attention_masks.append([0] * padding + [1] * len(sequence))
    return (
        torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        torch.tensor(attention_masks, dtype=torch.long, device="cuda"),
    )


def backward_prompt(
    torch,
    model,
    record: dict[str, Any],
    behavior: dict[tuple[str, int], list[float]],
    config: dict[str, Any],
    dataset_scale: float,
    pad_token_id: int,
) -> float:
    samples = sorted(record["samples"], key=lambda sample: int(sample["sample_id"]))
    group_size = config["sampling"]["primary"]["group_size"]
    if len(samples) != group_size:
        raise ValueError(f"incomplete group: {record['prompt_id']}")
    rewards = [float(sample["reward"]) for sample in samples]
    advantages = group_advantages(
        rewards, config["objective"]["advantage_epsilon"]
    )
    microbatch_size = config["learner"]["microbatch_size"]
    objective_value = 0.0
    for start in range(0, group_size, microbatch_size):
        batch_samples = samples[start : start + microbatch_size]
        sequences = [
            [*record["prompt_token_ids"], *sample["token_ids"]]
            for sample in batch_samples
        ]
        if any(
            len(sequence) > config["learner"]["maximum_sequence_tokens"]
            for sequence in sequences
        ):
            raise ValueError(f"sequence exceeds token limit: {record['prompt_id']}")
        generated_lengths = [len(sample["token_ids"]) for sample in batch_samples]
        if any(length <= 0 for length in generated_lengths):
            raise ValueError("empty continuation in learner input")
        input_ids, attention_mask = left_padded_batch(torch, sequences, pad_token_id)
        logits_to_keep = max(generated_lengths) + 1
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
        logits = outputs.logits
        if logits.shape[1] != logits_to_keep:
            raise RuntimeError(
                f"model returned {logits.shape[1]} logits positions; expected {logits_to_keep}"
            )
        sequence_losses = []
        for local_index, (sample, generated_length) in enumerate(
            zip(batch_samples, generated_lengths, strict=True)
        ):
            first = logits_to_keep - generated_length - 1
            last = logits_to_keep - 1
            token_logits = logits[local_index, first:last].to(torch.float32)
            targets = torch.tensor(
                sample["token_ids"], dtype=torch.long, device=token_logits.device
            )
            target_logits = token_logits.gather(1, targets[:, None]).squeeze(1)
            log_pi = target_logits - torch.logsumexp(token_logits, dim=-1)
            key = (record["prompt_id"], int(sample["sample_id"]))
            behavior_values = behavior.get(key)
            if behavior_values is None or len(behavior_values) != generated_length:
                raise ValueError(f"behavior score missing or wrong length: {key}")
            log_mu = torch.tensor(
                behavior_values,
                dtype=torch.float32,
                device=token_logits.device,
            ).detach()
            lower, upper = config["objective"]["log_ratio_clamp"]
            ratio = torch.exp(torch.clamp(log_pi - log_mu, lower, upper))
            advantage = torch.tensor(
                float(advantages[start + local_index]),
                dtype=torch.float32,
                device=token_logits.device,
            )
            unclipped = ratio * advantage
            clip = config["objective"]["ratio_clip_epsilon"]
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
            sequence_losses.append(-torch.minimum(unclipped, clipped).mean())
        batch_loss = torch.stack(sequence_losses).sum() / group_size / dataset_scale
        objective_value += float(batch_loss.detach().cpu()) * dataset_scale
        batch_loss.backward()
        del outputs, logits, batch_loss, input_ids, attention_mask
    return objective_value


def compute_gradient_set(
    torch,
    model,
    parameters,
    records: list[dict[str, Any]],
    behavior: dict[tuple[str, int], list[float]],
    config: dict[str, Any],
    pad_token_id: int,
    save_prompt_gradients: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, float, int]:
    prompt_gradients = []
    total_objective = 0.0
    nonzero_groups = 0
    if not save_prompt_gradients:
        model.zero_grad(set_to_none=True)
    for record in records:
        rewards = [float(sample["reward"]) for sample in record["samples"]]
        if len(set(rewards)) > 1:
            nonzero_groups += 1
        if save_prompt_gradients:
            model.zero_grad(set_to_none=True)
        total_objective += backward_prompt(
            torch,
            model,
            record,
            behavior,
            config,
            dataset_scale=1.0 if save_prompt_gradients else float(len(records)),
            pad_token_id=pad_token_id,
        )
        if save_prompt_gradients:
            prompt_gradients.append(flatten_gradients(parameters))
    if save_prompt_gradients:
        matrix = torch.stack(prompt_gradients)
        gradient = matrix.mean(dim=0)
    else:
        matrix = None
        gradient = flatten_gradients(parameters)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("learner produced a non-finite gradient")
    return gradient, matrix, total_objective / len(records), nonzero_groups


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    configure_environment()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    verification = json.loads(args.model_verification.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if trajectory.get("status") != "success" or trajectory.get("platform") != "a100":
        raise ValueError("trajectory is not a successful A100 rollout")
    if trajectory.get("workload_sha256") != config["workload"]["selected_sha256"]:
        raise ValueError("trajectory workload differs from PM-AR")
    if verification.get("status") != "success":
        raise ValueError("model verification did not pass")
    records = select_records(trajectory, selection, args.scope)
    if args.save_prompt_gradients and args.behavior_scores:
        raise ValueError("per-prompt mode requires the rollout reporter denominator")
    if args.behavior_scores and args.scope != "anchors":
        raise ValueError("A100 scoring overrides are defined only for anchor scope")

    behavior_sets: list[tuple[str, dict[tuple[str, int], list[float]]]]
    if args.behavior_scores:
        behavior_sets = [score_override(path) for path in args.behavior_scores]
    else:
        behavior_sets = [("rollout_reporter", source_scores(records))]

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    torch.manual_seed(config["learner"]["lora_initialization_seed"])
    torch.cuda.manual_seed_all(config["learner"]["lora_initialization_seed"])
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model.eval()
    target_names = inject_lora(
        model,
        config["learner"]["target_modules"],
        config["learner"]["rank"],
        config["learner"]["alpha"],
        config["learner"]["lora_initialization_seed"],
    )
    parameters = trainable_parameters(model)
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    pad_token_id = int(eos_token_id if eos_token_id is not None else 0)

    gradients = []
    prompt_matrix = None
    objective_values = []
    nonzero_groups = []
    labels = []
    for label, behavior in behavior_sets:
        gradient, matrix, objective, nonzero = compute_gradient_set(
            torch,
            model,
            parameters,
            records,
            behavior,
            config,
            pad_token_id,
            save_prompt_gradients=args.save_prompt_gradients,
        )
        gradients.append(gradient)
        if matrix is not None:
            prompt_matrix = matrix
        objective_values.append(objective)
        nonzero_groups.append(nonzero)
        labels.append(label)

    tensor_payload = {
        "schema": "policybraid.pm_ar.gradient_tensor.v1",
        "labels": labels,
        "gradients": torch.stack(gradients),
        "prompt_ids": [record["prompt_id"] for record in records],
        "task_families": [record["task_family"] for record in records],
        "prompt_gradients": prompt_matrix,
        "parameters": parameter_manifest(parameters),
    }
    args.output_tensor.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_payload, args.output_tensor)
    tensor_sha256 = sha256(args.output_tensor)
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": "policybraid.pm_ar.learner_gradient.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "selection_sha256": sha256(args.selection),
        "trajectory_sha256": sha256(args.trajectory),
        "behavior_score_sha256": [sha256(path) for path in args.behavior_scores],
        "run_index": args.run_index,
        "scope": args.scope,
        "physical_device": args.physical_device,
        "hostname": socket.gethostname(),
        "python": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "device_info": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "labels": labels,
        "prompt_count": len(records),
        "nonzero_advantage_groups": nonzero_groups,
        "objective_values": objective_values,
        "gradient_elements": gradients[0].numel(),
        "gradient_norms": [float(torch.linalg.vector_norm(item)) for item in gradients],
        "prompt_gradient_shape": list(prompt_matrix.shape)
        if prompt_matrix is not None
        else None,
        "target_module_count": len(target_names),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": elapsed,
        "tensor_path": str(args.output_tensor),
        "tensor_sha256": tensor_sha256,
        "status": "success",
    }
    del model, gradients, prompt_matrix, tensor_payload
    gc.collect()
    torch.cuda.empty_cache()
    return manifest, 0


def main() -> int:
    args = parse_args()
    for output in (args.output_tensor, args.output_manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    try:
        payload, exit_code = execute(args)
    except Exception as error:
        payload = {
            "schema": "policybraid.pm_ar.learner_gradient.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "run_index": args.run_index,
            "scope": args.scope,
            "physical_device": args.physical_device,
            "status": "incomplete",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 3
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

