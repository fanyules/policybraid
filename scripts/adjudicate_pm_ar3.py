#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.adjudication import (
    ess_fraction_from_log_weights,
    hierarchical_resample_counts,
    quantile_lower,
    sequence_mixture_log_weight,
    spearman_correlation,
)
from policybraid.grpo import gradient_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate the frozen PM-AR gate")
    parser.add_argument("--gradient-tensors", type=Path, nargs=5, required=True)
    parser.add_argument("--gradient-manifests", type=Path, nargs=5, required=True)
    parser.add_argument("--trainer-scores", type=Path, nargs=5, required=True)
    parser.add_argument("--a100-trajectories", type=Path, nargs=5, required=True)
    parser.add_argument("--npu-trajectories", type=Path, nargs=5, required=True)
    parser.add_argument("--a100-cross-scores", type=Path, nargs=5, required=True)
    parser.add_argument("--npu-cross-scores", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_records(payload: dict, source_platform: str, restart: int) -> list[dict]:
    matches = [
        source
        for source in payload["sources"]
        if source.get("source_platform") == source_platform
        and source.get("source_restart_index") == restart
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source is not unique in scored payload: {source_platform} r{restart}"
        )
    return matches[0]["records"]


def trajectory_map(payload: dict) -> dict[tuple[str, int], dict]:
    mapping = {}
    for record in payload["records"]:
        for sample in record["samples"]:
            key = (record["prompt_id"], int(sample["sample_id"]))
            if key in mapping:
                raise ValueError(f"duplicate trajectory key: {key}")
            mapping[key] = sample
    return mapping


def score_map(
    payload: dict, source_platform: str, restart: int, value_key: str
) -> dict[tuple[str, int], list[float]]:
    mapping = {}
    for record in source_records(payload, source_platform, restart):
        for sample in record["samples"]:
            key = (record["prompt_id"], int(sample["sample_id"]))
            if key in mapping:
                raise ValueError(f"duplicate scored key: {key}")
            values = [float(value) for value in sample[value_key]]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid scored values: {key}")
            mapping[key] = values
    return mapping


def bootstrap_gradient_distances(
    gram: np.ndarray, counts: np.ndarray, epsilon: float, batch_size: int = 128
) -> np.ndarray:
    unit_count = counts.shape[1]
    if gram.shape != (2 * unit_count, 2 * unit_count):
        raise ValueError("gradient Gram matrix and bootstrap counts are misaligned")
    weights = counts / counts.sum(axis=1, keepdims=True)
    distances = np.empty(counts.shape[0], dtype=np.float64)
    left_gram = gram[:unit_count, :unit_count]
    for start in range(0, counts.shape[0], batch_size):
        stop = min(counts.shape[0], start + batch_size)
        left_weights = weights[start:stop]
        differences = np.concatenate([left_weights, -left_weights], axis=1)
        numerator_squared = np.einsum(
            "bi,ij,bj->b", differences, gram, differences, optimize=True
        )
        denominator_squared = np.einsum(
            "bi,ij,bj->b", left_weights, left_gram, left_weights, optimize=True
        )
        distances[start:stop] = np.sqrt(np.maximum(numerator_squared, 0.0)) / (
            np.sqrt(np.maximum(denominator_squared, 0.0)) + epsilon
        )
    return distances


def bootstrap_weighted_medians(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if counts.shape[1] != flat.size:
        raise ValueError("ESS bootstrap counts and values are misaligned")
    order = np.argsort(flat, kind="mergesort")
    ordered_values = flat[order]
    cumulative = np.cumsum(counts[:, order], axis=1)
    draw_count = int(counts[0].sum())
    lower_position = (draw_count - 1) // 2
    upper_position = draw_count // 2
    lower_index = np.argmax(cumulative > lower_position, axis=1)
    upper_index = np.argmax(cumulative > upper_position, axis=1)
    return (ordered_values[lower_index] + ordered_values[upper_index]) / 2.0


def prompt_structure_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    task_families: list[str],
    replicates: int,
    seed: int,
) -> np.ndarray:
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(task_families):
        by_family[family].append(index)
    rng = np.random.default_rng(seed)
    correlations = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = []
        for indices in by_family.values():
            index_array = np.asarray(indices, dtype=np.int64)
            selected.extend(
                rng.choice(index_array, size=len(indices), replace=True).tolist()
            )
        correlations[replicate] = spearman_correlation(
            left[selected], right[selected]
        )
    return correlations


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def load_gradient_evidence(args: argparse.Namespace, config: dict):
    import torch

    labels = config["gradient_components"]["labels"]
    gradients = []
    left_matrices = []
    right_matrices = []
    prompt_ids = None
    task_families = None
    tensor_summaries = []
    for restart, (tensor_path, manifest_path) in enumerate(
        zip(args.gradient_tensors, args.gradient_manifests, strict=True)
    ):
        manifest = load_json(manifest_path)
        if (
            manifest.get("status") != "success"
            or manifest.get("restart_index") != restart
            or manifest.get("labels") != labels
            or manifest.get("tensor_sha256") != sha256(tensor_path)
        ):
            raise ValueError(f"gradient manifest failed validation: {manifest_path}")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != "policybraid.pm_ar3.gradient_tensor.v1"
            or payload.get("restart_index") != restart
            or payload.get("labels") != labels
            or payload.get("actual_prompt_gradient_labels") != ["g_A_A", "g_N_N"]
        ):
            raise ValueError(f"gradient tensor metadata failed: {tensor_path}")
        observed_ids = payload["prompt_ids"]
        observed_families = payload["task_families"]
        if prompt_ids is None:
            prompt_ids = observed_ids
            task_families = observed_families
        elif observed_ids != prompt_ids or observed_families != task_families:
            raise ValueError("gradient prompt coordinates differ across restarts")
        gradient_array = payload["gradients"].to(torch.float32)
        prompt_array = payload["actual_prompt_gradients"].to(torch.float32)
        if gradient_array.ndim != 2 or gradient_array.shape[0] != 4:
            raise ValueError(f"gradient tensor has the wrong shape: {tensor_path}")
        if prompt_array.ndim != 3 or prompt_array.shape[:2] != (2, 108):
            raise ValueError(f"prompt-gradient tensor has the wrong shape: {tensor_path}")
        gradients.append(gradient_array.numpy())
        left_matrices.append(prompt_array[0])
        right_matrices.append(prompt_array[1])
        tensor_summaries.append(
            {
                "path": str(tensor_path),
                "sha256": sha256(tensor_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "restart_index": restart,
                "gradient_shape": list(gradient_array.shape),
                "actual_prompt_gradient_shape": list(prompt_array.shape),
            }
        )
    combined = torch.cat(left_matrices + right_matrices, dim=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_device = combined.to(device)
    gram = (
        (combined_device @ combined_device.T)
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    del combined, combined_device, left_matrices, right_matrices
    if device == "cuda":
        torch.cuda.empty_cache()
    return (
        np.stack(gradients),
        gram,
        prompt_ids,
        task_families,
        tensor_summaries,
        device,
    )


def validate_scientific_inputs(args: argparse.Namespace, config: dict) -> None:
    parent = config["parent"]
    for path_key, hash_key in (
        ("pm_ar2_path", "pm_ar2_sha256"),
        ("noise_lock_path", "noise_lock_sha256"),
        ("workload_path", "workload_sha256"),
    ):
        if sha256(REPOSITORY_ROOT / parent[path_key]) != parent[hash_key]:
            raise ValueError(f"frozen parent artifact differs: {path_key}")
    groups = (
        args.gradient_manifests,
        args.trainer_scores,
        args.a100_trajectories,
        args.npu_trajectories,
        args.a100_cross_scores,
        args.npu_cross_scores,
    )
    if any(len(paths) != 5 for paths in groups):
        raise ValueError("PM-AR3 requires exactly five matched restarts")


def collect_policy_metrics(args: argparse.Namespace, config: dict, prompt_ids: list[str]):
    restart_count = 5
    prompt_count = len(prompt_ids)
    risk = np.empty((restart_count, prompt_count), dtype=np.float64)
    ess = np.empty((restart_count, prompt_count), dtype=np.float64)
    all_policy_gaps = []
    self_noise = {"a100": [], "910b": []}
    nonzero_support = []
    clip_counts = {label: [0, 0] for label in config["gradient_components"]["labels"]}
    length_values = {"a100": [], "910b": []}
    reward_values = {"a100": [], "910b": []}
    primary_sampling_pass = True
    evidence_hashes = []
    selected_a = set(config["mixed_group_ess"]["a100_sample_ids"])
    selected_n = set(config["mixed_group_ess"]["910b_sample_ids"])
    for restart in range(restart_count):
        paths = {
            "trajectory_a": args.a100_trajectories[restart],
            "trajectory_n": args.npu_trajectories[restart],
            "score_a": args.a100_cross_scores[restart],
            "score_n": args.npu_cross_scores[restart],
            "trainer": args.trainer_scores[restart],
        }
        payloads = {name: load_json(path) for name, path in paths.items()}
        trajectory_a = payloads["trajectory_a"]
        trajectory_n = payloads["trajectory_n"]
        score_a = payloads["score_a"]
        score_n = payloads["score_n"]
        trainer = payloads["trainer"]
        for platform, trajectory in (("a100", trajectory_a), ("910b", trajectory_n)):
            if (
                trajectory.get("status") != "success"
                or trajectory.get("platform") != platform
                or trajectory.get("restart_index") != restart
            ):
                raise ValueError(f"invalid trajectory evidence: {platform} r{restart}")
            primary_sampling_pass &= trajectory.get("sampling") == {
                "group_size": 8,
                "maximum_continuation_tokens": 128,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            }
        expected_hashes = [sha256(paths["trajectory_a"]), sha256(paths["trajectory_n"])]
        for platform, score in (("a100", score_a), ("910b", score_n)):
            if (
                score.get("status") != "success"
                or score.get("platform") != platform
                or score.get("restart_index") != restart
                or score.get("trajectory_sha256") != expected_hashes
            ):
                raise ValueError(f"invalid cross-score evidence: {platform} r{restart}")
            self_noise[platform].append(
                float(score["maximum_self_reporter_abs_gap"])
            )
        if (
            trainer.get("status") != "success"
            or trainer.get("restart_index") != restart
            or trainer.get("trajectory_count") != 1728
        ):
            raise ValueError(f"invalid trainer score evidence: r{restart}")
        maps = {
            "trajectory_a": trajectory_map(trajectory_a),
            "trajectory_n": trajectory_map(trajectory_n),
            "a_on_a": score_map(score_a, "a100", restart, "cross_scored_logprobs"),
            "a_on_n": score_map(score_a, "910b", restart, "cross_scored_logprobs"),
            "n_on_a": score_map(score_n, "a100", restart, "cross_scored_logprobs"),
            "n_on_n": score_map(score_n, "910b", restart, "cross_scored_logprobs"),
            "pi_on_a": score_map(trainer, "a100", restart, "trainer_logprobs"),
            "pi_on_n": score_map(trainer, "910b", restart, "trainer_logprobs"),
        }
        records_by_platform = {
            "a100": trajectory_a["records"],
            "910b": trajectory_n["records"],
        }
        for platform, records in records_by_platform.items():
            nonzero = 0
            represented_families: set[str] = set()
            for record in records:
                rewards = [float(sample["reward"]) for sample in record["samples"]]
                if len(set(rewards)) > 1:
                    nonzero += 1
                    represented_families.add(record["task_family"])
                reward_values[platform].extend(rewards)
                length_values[platform].extend(
                    len(sample["token_ids"]) for sample in record["samples"]
                )
            nonzero_support.append(
                {
                    "platform": platform,
                    "restart_index": restart,
                    "groups": nonzero,
                    "families": sorted(represented_families),
                }
            )
        for prompt_index, prompt_id in enumerate(prompt_ids):
            prompt_gaps = []
            mixed_log_weights = []
            for source_platform in ("a100", "910b"):
                suffix = "a" if source_platform == "a100" else "n"
                trajectory_key = f"trajectory_{suffix}"
                pi_key = f"pi_on_{suffix}"
                a_key = f"a_on_{suffix}"
                n_key = f"n_on_{suffix}"
                selected_ids = selected_a if source_platform == "a100" else selected_n
                for sample_id in range(8):
                    key = (prompt_id, sample_id)
                    sample = maps[trajectory_key][key]
                    pi = maps[pi_key][key]
                    mu_a = maps[a_key][key]
                    mu_n = maps[n_key][key]
                    if not (
                        len(sample["token_ids"])
                        == len(pi)
                        == len(mu_a)
                        == len(mu_n)
                    ):
                        raise ValueError(f"token evidence is misaligned: {key}")
                    gaps = np.abs(
                        np.asarray(mu_n, dtype=np.float64)
                        - np.asarray(mu_a, dtype=np.float64)
                    )
                    prompt_gaps.extend(gaps.tolist())
                    all_policy_gaps.extend(gaps.tolist())
                    reporter = np.asarray(
                        sample["processed_behavior_logprobs"], dtype=np.float64
                    )
                    log_ratio_actual = np.asarray(pi, dtype=np.float64) - reporter
                    label_actual = "g_A_A" if source_platform == "a100" else "g_N_N"
                    clipped_actual = np.count_nonzero(
                        (log_ratio_actual < math.log(0.8))
                        | (log_ratio_actual > math.log(1.2))
                    )
                    clip_counts[label_actual][0] += int(clipped_actual)
                    clip_counts[label_actual][1] += int(log_ratio_actual.size)
                    cross_denominator = mu_n if source_platform == "a100" else mu_a
                    log_ratio_cross = np.asarray(pi, dtype=np.float64) - np.asarray(
                        cross_denominator, dtype=np.float64
                    )
                    label_cross = "g_A_N" if source_platform == "a100" else "g_N_A"
                    clipped_cross = np.count_nonzero(
                        (log_ratio_cross < math.log(0.8))
                        | (log_ratio_cross > math.log(1.2))
                    )
                    clip_counts[label_cross][0] += int(clipped_cross)
                    clip_counts[label_cross][1] += int(log_ratio_cross.size)
                    if sample_id in selected_ids:
                        mixed_log_weights.append(
                            sequence_mixture_log_weight(pi, mu_a, mu_n)
                        )
            if len(mixed_log_weights) != 8:
                raise RuntimeError("mixed group does not contain exactly 4A+4N sequences")
            risk[restart, prompt_index] = float(np.quantile(prompt_gaps, 0.95))
            ess[restart, prompt_index] = ess_fraction_from_log_weights(
                mixed_log_weights
            )
        evidence_hashes.append(
            {
                "restart_index": restart,
                **{name: sha256(path) for name, path in paths.items()},
            }
        )
    return {
        "risk": risk,
        "ess": ess,
        "all_policy_gaps": np.asarray(all_policy_gaps, dtype=np.float64),
        "self_noise": self_noise,
        "nonzero_support": nonzero_support,
        "clip_counts": clip_counts,
        "length_values": length_values,
        "reward_values": reward_values,
        "primary_sampling_pass": bool(primary_sampling_pass),
        "evidence_hashes": evidence_hashes,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    validate_scientific_inputs(args, config)
    parent_config = load_json(REPOSITORY_ROOT / config["parent"]["pm_ar_path"])
    noise_lock = load_json(REPOSITORY_ROOT / config["parent"]["noise_lock_path"])
    epsilon = float(noise_lock["gradient_distance_epsilon"])
    gradients, gram, prompt_ids, task_families, tensor_summaries, gram_device = (
        load_gradient_evidence(args, config)
    )
    if len(prompt_ids) != 108:
        raise ValueError("PM-AR3 gradient evidence does not contain 108 prompts")
    bootstrap = config["bootstrap"]
    gradient_counts = hierarchical_resample_counts(
        task_families,
        restart_count=5,
        replicates=bootstrap["replicates"],
        seed=bootstrap["gradient_seed"],
    )
    gradient_bootstrap = bootstrap_gradient_distances(gram, gradient_counts, epsilon)
    gradient_lower95 = quantile_lower(gradient_bootstrap, 0.05)
    restart_distances = [
        gradient_distance(gradients[index, 0], gradients[index, 3], epsilon)
        for index in range(5)
    ]
    mean_gradients = gradients.mean(axis=0, dtype=np.float64)
    aggregate_distance = gradient_distance(
        mean_gradients[0], mean_gradients[3], epsilon
    )
    origin_norm = float(np.linalg.norm(mean_gradients[0]))
    actual = mean_gradients[3] - mean_gradients[0]
    trajectory_component = mean_gradients[2] - mean_gradients[0]
    denominator_component = mean_gradients[1] - mean_gradients[0]
    interaction = (
        mean_gradients[3]
        - mean_gradients[2]
        - mean_gradients[1]
        + mean_gradients[0]
    )

    policy = collect_policy_metrics(args, config, prompt_ids)
    thresholds = config["thresholds"]
    u_noise = float(noise_lock["u_noise"])
    gradient_threshold = max(
        float(thresholds["gradient_distance_floor"]),
        float(thresholds["gradient_noise_multiplier"]) * u_noise,
    )
    restarts_above = sum(
        distance >= gradient_threshold for distance in restart_distances
    )
    conditional_gap = float(np.quantile(policy["all_policy_gaps"], 0.95))
    maximum_self_noise = {
        platform: max(values) for platform, values in policy["self_noise"].items()
    }
    conditional_pass = all(
        conditional_gap > value for value in maximum_self_noise.values()
    )
    pm_rq_pass = all(
        value <= thresholds["pm_rq_tolerance"]
        for values in policy["self_noise"].values()
        for value in values
    )

    ess_counts = hierarchical_resample_counts(
        task_families,
        restart_count=5,
        replicates=bootstrap["replicates"],
        seed=bootstrap["ess_seed"],
    )
    ess_bootstrap = bootstrap_weighted_medians(policy["ess"], ess_counts)
    ess_lower95 = quantile_lower(ess_bootstrap, 0.05)
    ess_point = float(np.median(policy["ess"]))

    prompt_config = config["prompt_structure"]
    aggregate_risk = np.median(policy["risk"], axis=0)
    risk_iqr = float(
        np.quantile(aggregate_risk, 0.75) - np.quantile(aggregate_risk, 0.25)
    )
    prompt_medians = np.median(policy["risk"], axis=0)
    within_mad = float(np.median(np.abs(policy["risk"] - prompt_medians[None, :])))
    if within_mad == 0.0:
        risk_spread_ratio = math.inf if risk_iqr > 0 else 0.0
    else:
        risk_spread_ratio = risk_iqr / within_mad
    left_risk = np.median(
        policy["risk"][prompt_config["restart_half_left"], :], axis=0
    )
    right_risk = np.median(
        policy["risk"][prompt_config["restart_half_right"], :], axis=0
    )
    half_spearman = spearman_correlation(left_risk, right_risk)
    prompt_bootstrap = prompt_structure_bootstrap(
        left_risk,
        right_risk,
        task_families,
        bootstrap["replicates"],
        bootstrap["prompt_structure_seed"],
    )
    half_spearman_lower95 = quantile_lower(prompt_bootstrap, 0.05)
    family_spearman = {}
    for family in sorted(set(task_families)):
        indices = np.asarray(
            [index for index, value in enumerate(task_families) if value == family]
        )
        family_spearman[family] = spearman_correlation(
            left_risk[indices], right_risk[indices]
        )
    directional_families = sum(
        value >= prompt_config["family_directional_min_spearman"]
        for value in family_spearman.values()
    )
    prompt_structure_pass = (
        risk_spread_ratio
        > float(thresholds["prompt_risk_iqr_over_within_mad_min"])
        and half_spearman_lower95
        > float(thresholds["prompt_half_spearman_lower95_min"])
        and directional_families >= int(thresholds["minimum_directional_families"])
    )

    minimum_nonzero = min(item["groups"] for item in policy["nonzero_support"])
    minimum_nonzero_families = min(
        len(item["families"]) for item in policy["nonzero_support"]
    )
    support_pass = (
        minimum_nonzero >= thresholds["minimum_nonzero_advantage_groups"]
        and minimum_nonzero_families >= thresholds["minimum_families"]
    )
    checks = {
        "pm_rq_sentinels": pm_rq_pass,
        "conditional_policy_gap": conditional_pass,
        "gradient_lower95": gradient_lower95 >= gradient_threshold,
        "restart_replication": restarts_above
        >= thresholds["minimum_restarts_above_threshold"],
        "mixed_group_ess": ess_lower95
        >= thresholds["mixed_ess_fraction_lower95_min"],
        "workload_support": support_pass,
        "prompt_structure": prompt_structure_pass,
        "primary_full_support": policy["primary_sampling_pass"],
    }
    passed = all(checks.values())
    if passed:
        decision = "enter_pm_b"
    elif not checks["gradient_lower95"] or not checks["restart_replication"]:
        decision = "stop_policybraid_gradient_within_registered_boundary"
    elif not checks["prompt_structure"]:
        decision = "no_prompt_aware_controller_global_or_unstable_effect_only"
    else:
        decision = "stop_policybraid_pm_ar_threshold_failure"

    clip_rates = {
        label: count / total if total else 0.0
        for label, (count, total) in policy["clip_counts"].items()
    }
    return {
        "schema": "policybraid.pm_ar.adjudication.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "status": "passed" if passed else "failed",
        "decision": decision,
        "claim_c_p": "passed" if passed else "failed",
        "predecessor_state_preserved": {
            "PM-A0": "workload_insufficient_stopped",
            "C-P_before_PM-AR": "not_tested",
        },
        "checks": checks,
        "gradient": {
            "u_noise": u_noise,
            "registered_threshold": gradient_threshold,
            "restart_distances": restart_distances,
            "restarts_above_threshold": restarts_above,
            "aggregate_distance": aggregate_distance,
            "bootstrap_replicates": bootstrap["replicates"],
            "bootstrap_seed": bootstrap["gradient_seed"],
            "lower95": gradient_lower95,
            "bootstrap_quantiles": {
                "p05_lower": gradient_lower95,
                "p50": float(np.quantile(gradient_bootstrap, 0.50)),
                "p95": float(np.quantile(gradient_bootstrap, 0.95)),
            },
            "g_A_A_norm": origin_norm,
            "g_N_N_norm": float(np.linalg.norm(mean_gradients[3])),
            "g_A_A_g_N_N_cosine": cosine(mean_gradients[0], mean_gradients[3]),
            "decomposition": {
                "actual_norm_over_origin": float(np.linalg.norm(actual) / (origin_norm + epsilon)),
                "trajectory_source_norm_over_origin": float(
                    np.linalg.norm(trajectory_component) / (origin_norm + epsilon)
                ),
                "denominator_norm_over_origin": float(
                    np.linalg.norm(denominator_component) / (origin_norm + epsilon)
                ),
                "interaction_norm_over_origin": float(
                    np.linalg.norm(interaction) / (origin_norm + epsilon)
                ),
            },
            "gram_compute_device": gram_device,
            "tensor_evidence": tensor_summaries,
        },
        "conditional_policy": {
            "p95_absolute_a100_npu_logprob_gap": conditional_gap,
            "maximum_self_reporter_abs_gap": maximum_self_noise,
            "token_count": int(policy["all_policy_gaps"].size),
        },
        "mixed_group_ess": {
            "composition": {
                "a100_sample_ids": config["mixed_group_ess"]["a100_sample_ids"],
                "910b_sample_ids": config["mixed_group_ess"]["910b_sample_ids"],
            },
            "group_count": int(policy["ess"].size),
            "median": ess_point,
            "lower95": ess_lower95,
            "bootstrap_replicates": bootstrap["replicates"],
            "bootstrap_seed": bootstrap["ess_seed"],
            "minimum": float(policy["ess"].min()),
            "p05": float(np.quantile(policy["ess"], 0.05)),
        },
        "prompt_structure": {
            "risk_iqr": risk_iqr,
            "within_process_mad": within_mad,
            "iqr_over_within_mad": risk_spread_ratio,
            "restart_half_left": prompt_config["restart_half_left"],
            "restart_half_right": prompt_config["restart_half_right"],
            "half_spearman": half_spearman,
            "half_spearman_lower95": half_spearman_lower95,
            "bootstrap_replicates": bootstrap["replicates"],
            "bootstrap_seed": bootstrap["prompt_structure_seed"],
            "family_spearman": family_spearman,
            "directional_family_count": directional_families,
            "aggregate_risk_quantiles": {
                "p25": float(np.quantile(aggregate_risk, 0.25)),
                "p50": float(np.quantile(aggregate_risk, 0.50)),
                "p75": float(np.quantile(aggregate_risk, 0.75)),
            },
        },
        "workload_support": {
            "minimum_nonzero_advantage_groups": minimum_nonzero,
            "minimum_nonzero_advantage_family_count": minimum_nonzero_families,
            "by_platform_restart": policy["nonzero_support"],
        },
        "objective_diagnostics": {
            "token_clip_rates": clip_rates,
            "length": {
                platform: {
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for platform, values in policy["length_values"].items()
            },
            "reward": {
                platform: {
                    "mean": float(np.mean(values)),
                    "variance": float(np.var(values)),
                }
                for platform, values in policy["reward_values"].items()
            },
        },
        "input_evidence_sha256": policy["evidence_hashes"],
        "statistics_contract": config["bootstrap"],
        "thresholds": thresholds,
        "scientific_scope": "primary_full_support_A100_and_patched_910B_only",
        "pm_b_unblocked": passed,
        "rtx_unblocked": False,
        "controller_unblocked": False,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = execute(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "gradient_lower95": payload["gradient"]["lower95"],
                "threshold": payload["gradient"]["registered_threshold"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
