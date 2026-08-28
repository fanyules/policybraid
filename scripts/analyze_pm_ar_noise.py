#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.grpo import gradient_distance, quantile_higher
from policybraid.noise import (
    aggregate_distance_from_gram,
    paired_stratified_bootstrap_distances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate the PM-AR1 A100 noise lock")
    parser.add_argument("--trainer-tensors", type=Path, nargs=5, required=True)
    parser.add_argument("--scoring-tensor", type=Path, required=True)
    parser.add_argument("--resample-tensors", type=Path, nargs=2, required=True)
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


def load_tensor(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "policybraid.pm_ar.gradient_tensor.v1":
        raise ValueError(f"unexpected gradient tensor schema: {path}")
    return payload


def execute(args: argparse.Namespace) -> dict:
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    epsilon = config["pm_ar1_noise_lock"]["gradient_distance_epsilon"]
    trainer_payloads = [load_tensor(path) for path in args.trainer_tensors]
    trainer_gradients = [
        payload["gradients"][0].to(torch.float32).numpy()
        for payload in trainer_payloads
    ]
    expected_shape = trainer_gradients[0].shape
    if any(gradient.shape != expected_shape for gradient in trainer_gradients):
        raise ValueError("trainer gradients have inconsistent shapes")
    trainer_distances = [
        gradient_distance(trainer_gradients[0], trainer_gradients[index], epsilon)
        for index in range(1, 5)
    ]

    scoring_payload = load_tensor(args.scoring_tensor)
    scoring_gradients = scoring_payload["gradients"].to(torch.float32).numpy()
    if scoring_gradients.shape != (5, expected_shape[0]):
        raise ValueError("scoring tensor does not contain five compatible gradients")
    scoring_distances = [
        gradient_distance(scoring_gradients[0], scoring_gradients[index], epsilon)
        for index in range(1, 5)
    ]

    resample_payloads = [load_tensor(path) for path in args.resample_tensors]
    left_matrix = resample_payloads[0].get("prompt_gradients")
    right_matrix = resample_payloads[1].get("prompt_gradients")
    if left_matrix is None or right_matrix is None:
        raise ValueError("resample tensors must contain per-prompt gradients")
    if resample_payloads[0]["prompt_ids"] != resample_payloads[1]["prompt_ids"]:
        raise ValueError("resample prompt IDs are not aligned")
    if resample_payloads[0]["task_families"] != resample_payloads[1]["task_families"]:
        raise ValueError("resample task families are not aligned")
    if left_matrix.shape != right_matrix.shape or left_matrix.shape[0] != 108:
        raise ValueError("resample prompt-gradient matrices have the wrong shape")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined = torch.cat(
        [left_matrix.to(torch.float32), right_matrix.to(torch.float32)], dim=0
    ).to(device)
    gram = (combined @ combined.T).to(device="cpu", dtype=torch.float64).numpy()
    del combined
    bootstrap = config["pm_ar1_noise_lock"]["prompt_bootstrap"]
    resample_distances = paired_stratified_bootstrap_distances(
        gram,
        resample_payloads[0]["task_families"],
        bootstrap["replicates"],
        bootstrap["seed"],
        epsilon,
    )
    resample_actual = aggregate_distance_from_gram(gram, 108, epsilon)

    trainer_upper = quantile_higher(trainer_distances, 0.95)
    scoring_upper = quantile_higher(scoring_distances, 0.95)
    resample_upper = quantile_higher(resample_distances.tolist(), 0.95)
    u_noise = max(trainer_upper, scoring_upper, resample_upper)
    if not np.isfinite(u_noise):
        raise RuntimeError("PM-AR1 produced a non-finite U_noise")
    return {
        "schema": "policybraid.pm_ar.noise_lock.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "status": "locked",
        "gradient_distance_epsilon": epsilon,
        "trainer_process": {
            "distances_r0_to_r1_r4": trainer_distances,
            "upper95": trainer_upper,
            "tensor_sha256": [sha256(path) for path in args.trainer_tensors],
        },
        "a100_scoring": {
            "distances_r0_to_r1_r4": scoring_distances,
            "upper95": scoring_upper,
            "tensor_sha256": sha256(args.scoring_tensor),
        },
        "a100_resample": {
            "actual_distance": resample_actual,
            "bootstrap_replicates": int(resample_distances.size),
            "bootstrap_seed": bootstrap["seed"],
            "quantiles": {
                "p50": float(np.quantile(resample_distances, 0.50)),
                "p90": float(np.quantile(resample_distances, 0.90)),
                "p95_higher": resample_upper,
                "p99": float(np.quantile(resample_distances, 0.99)),
            },
            "upper95": resample_upper,
            "tensor_sha256": [sha256(path) for path in args.resample_tensors],
        },
        "u_noise": u_noise,
        "u_noise_rule": "max(trainer_upper95,a100_scoring_upper95,a100_resample_upper95)",
        "pm_ar2_unblocked": True,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = execute(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "u_noise": payload["u_noise"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

