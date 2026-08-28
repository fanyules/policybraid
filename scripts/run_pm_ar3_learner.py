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
        description="Compute the four canonical PM-AR3 learner gradients"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-verification", type=Path, required=True)
    parser.add_argument("--a100-trajectory", type=Path, required=True)
    parser.add_argument("--npu-trajectory", type=Path, required=True)
    parser.add_argument("--a100-cross-score", type=Path, required=True)
    parser.add_argument("--npu-cross-score", type=Path, required=True)
    parser.add_argument("--restart-index", type=int, choices=range(5), required=True)
    parser.add_argument("--physical-device", required=True)
    parser.add_argument("--output-tensor", type=Path, required=True)
    parser.add_argument("--output-trainer-scores", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "pm_ar3.json"
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reporter_scores(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[float]]:
    return {
        (record["prompt_id"], int(sample["sample_id"])): [
            float(value) for value in sample["processed_behavior_logprobs"]
        ]
        for record in records
        for sample in record["samples"]
    }


def cross_scores(
    payload: dict[str, Any], source_platform: str, restart_index: int
) -> dict[tuple[str, int], list[float]]:
    matches = [
        source
        for source in payload["sources"]
        if source.get("source_platform") == source_platform
        and source.get("source_restart_index") == restart_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cross-score source is not unique: {source_platform} r{restart_index}"
        )
    mapping = {}
    for record in matches[0]["records"]:
        for sample in record["samples"]:
            key = (record["prompt_id"], int(sample["sample_id"]))
            if key in mapping:
                raise ValueError(f"duplicate cross-score key: {key}")
            values = [float(value) for value in sample["cross_scored_logprobs"]]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid cross-score values: {key}")
            mapping[key] = values
    return mapping


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
    parent_config: dict[str, Any],
    dataset_scale: float,
    pad_token_id: int,
    trainer_scores: dict[tuple[str, int], list[float]] | None,
) -> float:
    samples = sorted(record["samples"], key=lambda sample: int(sample["sample_id"]))
    group_size = parent_config["sampling"]["primary"]["group_size"]
    if len(samples) != group_size:
        raise ValueError(f"incomplete group: {record['prompt_id']}")
    rewards = [float(sample["reward"]) for sample in samples]
    advantages = group_advantages(
        rewards, parent_config["objective"]["advantage_epsilon"]
    )
    microbatch_size = parent_config["learner"]["microbatch_size"]
    objective_value = 0.0
    for start in range(0, group_size, microbatch_size):
        batch_samples = samples[start : start + microbatch_size]
        sequences = [
            [*record["prompt_token_ids"], *sample["token_ids"]]
            for sample in batch_samples
        ]
        if any(
            len(sequence) > parent_config["learner"]["maximum_sequence_tokens"]
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
            if trainer_scores is not None:
                if key in trainer_scores:
                    raise ValueError(f"duplicate trainer-score key: {key}")
                trainer_scores[key] = [
                    float(value) for value in log_pi.detach().to("cpu", torch.float32)
                ]
            behavior_values = behavior.get(key)
            if behavior_values is None or len(behavior_values) != generated_length:
                raise ValueError(f"behavior score missing or wrong length: {key}")
            log_mu = torch.tensor(
                behavior_values,
                dtype=torch.float32,
                device=token_logits.device,
            ).detach()
            lower, upper = parent_config["objective"]["log_ratio_clamp"]
            ratio = torch.exp(torch.clamp(log_pi - log_mu, lower, upper))
            advantage = torch.tensor(
                float(advantages[start + local_index]),
                dtype=torch.float32,
                device=token_logits.device,
            )
            unclipped = ratio * advantage
            clip = parent_config["objective"]["ratio_clip_epsilon"]
            clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
            sequence_losses.append(-torch.minimum(unclipped, clipped).mean())
        batch_loss = torch.stack(sequence_losses).sum() / group_size / dataset_scale
        objective_value += float(batch_loss.detach().cpu()) * dataset_scale
        batch_loss.backward()
        del outputs, logits, batch_loss, input_ids, attention_mask
    return objective_value


def compute_gradient(
    torch,
    model,
    parameters,
    records: list[dict[str, Any]],
    behavior: dict[tuple[str, int], list[float]],
    parent_config: dict[str, Any],
    pad_token_id: int,
    save_prompt_gradients: bool,
    capture_trainer_scores: bool,
):
    prompt_gradients = []
    captured: dict[tuple[str, int], list[float]] | None = (
        {} if capture_trainer_scores else None
    )
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
            parent_config,
            dataset_scale=1.0 if save_prompt_gradients else float(len(records)),
            pad_token_id=pad_token_id,
            trainer_scores=captured,
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
    return (
        gradient,
        matrix,
        total_objective / len(records),
        nonzero_groups,
        captured,
    )


def trainer_score_source(
    source_platform: str,
    restart_index: int,
    records: list[dict[str, Any]],
    values: dict[tuple[str, int], list[float]],
) -> dict[str, Any]:
    output_records = []
    for record in records:
        output_samples = []
        for sample in sorted(record["samples"], key=lambda item: int(item["sample_id"])):
            key = (record["prompt_id"], int(sample["sample_id"]))
            logprobs = values.get(key)
            if logprobs is None or len(logprobs) != len(sample["token_ids"]):
                raise ValueError(f"trainer score is missing or misaligned: {key}")
            output_samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "token_ids": sample["token_ids"],
                    "trainer_logprobs": logprobs,
                }
            )
        output_records.append(
            {
                "prompt_id": record["prompt_id"],
                "task_family": record["task_family"],
                "samples": output_samples,
            }
        )
    return {
        "source_platform": source_platform,
        "source_restart_index": restart_index,
        "records": output_records,
    }


def validate_inputs(args: argparse.Namespace, config: dict, parent_config: dict):
    parent = config["parent"]
    if sha256(REPOSITORY_ROOT / parent["pm_ar2_path"]) != parent["pm_ar2_sha256"]:
        raise ValueError("PM-AR2 config differs from the PM-AR3 freeze")
    if sha256(REPOSITORY_ROOT / parent["noise_lock_path"]) != parent["noise_lock_sha256"]:
        raise ValueError("PM-AR1 noise lock differs from the PM-AR3 freeze")
    if sha256(REPOSITORY_ROOT / parent["workload_path"]) != parent["workload_sha256"]:
        raise ValueError("PM-AR workload differs from the PM-AR3 freeze")
    verification = load_json(args.model_verification)
    if (
        verification.get("status") != "success"
        or verification.get("asset_revision") != parent_config["model"]["revision"]
    ):
        raise ValueError("model verification does not match the PM-AR checkpoint")
    trajectories = {
        "a100": load_json(args.a100_trajectory),
        "910b": load_json(args.npu_trajectory),
    }
    expected_prompt_ids = None
    for platform, trajectory in trajectories.items():
        if (
            trajectory.get("status") != "success"
            or trajectory.get("platform") != platform
            or trajectory.get("restart_index") != args.restart_index
            or trajectory.get("workload_sha256") != parent["workload_sha256"]
        ):
            raise ValueError(f"invalid {platform} trajectory input")
        prompt_ids = [record["prompt_id"] for record in trajectory["records"]]
        if expected_prompt_ids is None:
            expected_prompt_ids = prompt_ids
        elif prompt_ids != expected_prompt_ids:
            raise ValueError("A100 and 910B prompt orders are not aligned")
    score_payloads = {
        "a100": load_json(args.a100_cross_score),
        "910b": load_json(args.npu_cross_score),
    }
    expected_trajectory_hashes = [
        sha256(args.a100_trajectory),
        sha256(args.npu_trajectory),
    ]
    tolerance = config["thresholds"]["pm_rq_tolerance"]
    for platform, scores in score_payloads.items():
        if (
            scores.get("status") != "success"
            or scores.get("platform") != platform
            or scores.get("restart_index") != args.restart_index
            or scores.get("trajectory_sha256") != expected_trajectory_hashes
            or float(scores.get("maximum_self_reporter_abs_gap")) > tolerance
        ):
            raise ValueError(f"invalid {platform} cross-score input")
    return verification, trajectories, score_payloads


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_environment()
    config = load_json(args.config)
    parent_config = load_json(REPOSITORY_ROOT / config["parent"]["pm_ar_path"])
    verification, trajectories, score_payloads = validate_inputs(
        args, config, parent_config
    )
    records_a = trajectories["a100"]["records"]
    records_n = trajectories["910b"]["records"]
    behavior_a_reporter = reporter_scores(records_a)
    behavior_n_reporter = reporter_scores(records_n)
    behavior_n_on_a = cross_scores(score_payloads["910b"], "a100", args.restart_index)
    behavior_a_on_n = cross_scores(score_payloads["a100"], "910b", args.restart_index)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    torch.manual_seed(parent_config["learner"]["lora_initialization_seed"])
    torch.cuda.manual_seed_all(parent_config["learner"]["lora_initialization_seed"])
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
        parent_config["learner"]["target_modules"],
        parent_config["learner"]["rank"],
        parent_config["learner"]["alpha"],
        parent_config["learner"]["lora_initialization_seed"],
    )
    parameters = trainable_parameters(model)
    eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    pad_token_id = int(eos_token_id if eos_token_id is not None else 0)

    g_aa, prompt_aa, obj_aa, nz_a, trainer_a = compute_gradient(
        torch,
        model,
        parameters,
        records_a,
        behavior_a_reporter,
        parent_config,
        pad_token_id,
        save_prompt_gradients=True,
        capture_trainer_scores=True,
    )
    g_an, _unused, obj_an, nz_an, _unused_scores = compute_gradient(
        torch,
        model,
        parameters,
        records_a,
        behavior_n_on_a,
        parent_config,
        pad_token_id,
        save_prompt_gradients=False,
        capture_trainer_scores=False,
    )
    g_na, _unused, obj_na, nz_na, _unused_scores = compute_gradient(
        torch,
        model,
        parameters,
        records_n,
        behavior_a_on_n,
        parent_config,
        pad_token_id,
        save_prompt_gradients=False,
        capture_trainer_scores=False,
    )
    g_nn, prompt_nn, obj_nn, nz_n, trainer_n = compute_gradient(
        torch,
        model,
        parameters,
        records_n,
        behavior_n_reporter,
        parent_config,
        pad_token_id,
        save_prompt_gradients=True,
        capture_trainer_scores=True,
    )
    labels = config["gradient_components"]["labels"]
    gradients = [g_aa, g_an, g_na, g_nn]
    objectives = [obj_aa, obj_an, obj_na, obj_nn]
    nonzero = [nz_a, nz_an, nz_na, nz_n]
    prompt_ids = [record["prompt_id"] for record in records_a]
    task_families = [record["task_family"] for record in records_a]
    tensor_payload = {
        "schema": "policybraid.pm_ar3.gradient_tensor.v1",
        "restart_index": args.restart_index,
        "labels": labels,
        "gradients": torch.stack(gradients),
        "actual_prompt_gradient_labels": ["g_A_A", "g_N_N"],
        "actual_prompt_gradients": torch.stack([prompt_aa, prompt_nn]),
        "prompt_ids": prompt_ids,
        "task_families": task_families,
        "parameters": parameter_manifest(parameters),
    }
    args.output_tensor.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_payload, args.output_tensor)
    tensor_sha256 = sha256(args.output_tensor)
    trainer_payload = {
        "schema": "policybraid.pm_ar3.trainer_score.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "restart_index": args.restart_index,
        "source_count": 2,
        "trajectory_count": 1728,
        "sources": [
            trainer_score_source("a100", args.restart_index, records_a, trainer_a),
            trainer_score_source("910b", args.restart_index, records_n, trainer_n),
        ],
        "status": "success",
    }
    args.output_trainer_scores.parent.mkdir(parents=True, exist_ok=True)
    args.output_trainer_scores.write_text(
        json.dumps(
            trainer_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trainer_sha256 = sha256(args.output_trainer_scores)
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": "policybraid.pm_ar3.learner_gradient.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "model_verification_sha256": sha256(args.model_verification),
        "trajectory_sha256": {
            "a100": sha256(args.a100_trajectory),
            "910b": sha256(args.npu_trajectory),
        },
        "cross_score_sha256": {
            "a100": sha256(args.a100_cross_score),
            "910b": sha256(args.npu_cross_score),
        },
        "restart_index": args.restart_index,
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
        "prompt_count": len(prompt_ids),
        "trajectory_count": 1728,
        "nonzero_advantage_groups": nonzero,
        "objective_values": objectives,
        "gradient_elements": gradients[0].numel(),
        "gradient_norms": [float(torch.linalg.vector_norm(item)) for item in gradients],
        "actual_prompt_gradient_shape": [2, *list(prompt_aa.shape)],
        "target_module_count": len(target_names),
        "trainable_parameter_count": sum(
            parameter.numel() for _, parameter in parameters
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": elapsed,
        "tensor_path": str(args.output_tensor),
        "tensor_sha256": tensor_sha256,
        "trainer_score_path": str(args.output_trainer_scores),
        "trainer_score_sha256": trainer_sha256,
        "model_asset_revision": verification["asset_revision"],
        "status": "success",
    }
    del model, gradients, prompt_aa, prompt_nn, tensor_payload
    gc.collect()
    torch.cuda.empty_cache()
    return manifest, trainer_payload


def main() -> int:
    args = parse_args()
    for output in (
        args.output_tensor,
        args.output_trainer_scores,
        args.output_manifest,
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    try:
        manifest, _trainer_payload = execute(args)
        exit_code = 0
    except Exception as error:
        manifest = {
            "schema": "policybraid.pm_ar3.learner_gradient.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": repository_commit(),
            "restart_index": args.restart_index,
            "physical_device": args.physical_device,
            "status": "incomplete",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 3
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
