#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the immutable PM-AR2 cube")
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
    parser.add_argument(
        "--cube-root",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "pm_ar" / "cube",
    )
    parser.add_argument("--output", type=Path, required=True)
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


def load_workload(path: Path) -> tuple[list[str], dict[str, str]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_ids = [record["prompt_id"] for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("PM-AR workload contains duplicate prompt IDs")
    return prompt_ids, {record["prompt_id"]: record["task_family"] for record in records}


def validate_rollout(
    path: Path,
    platform: str,
    restart: int,
    config: dict,
    prompt_ids: list[str],
    families: dict[str, str],
) -> tuple[dict, dict]:
    payload = load_json(path)
    expected_seed = config["restart_seed_bases"][restart]
    expected_device = config["device_schedules"][platform][restart]
    checks = {
        "status": payload.get("status") == "success",
        "platform": payload.get("platform") == platform,
        "restart_index": payload.get("restart_index") == restart,
        "seed_base": payload.get("seed_base") == expected_seed,
        "prompt_count": payload.get("prompt_count") == 108,
        "trajectory_count": payload.get("trajectory_count") == 864,
        "workload_sha256": payload.get("workload_sha256")
        == config["workload"]["sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"rollout metadata failed for {path}: {checks}")
    records = payload.get("records", [])
    if [record.get("prompt_id") for record in records] != prompt_ids:
        raise ValueError(f"rollout prompt order differs from PM-AR: {path}")
    nonzero = 0
    nonzero_families: set[str] = set()
    for record in records:
        prompt_id = record["prompt_id"]
        if record.get("task_family") != families[prompt_id]:
            raise ValueError(f"task family mismatch for {prompt_id}: {path}")
        samples = sorted(record.get("samples", []), key=lambda item: int(item["sample_id"]))
        if [int(sample["sample_id"]) for sample in samples] != list(range(8)):
            raise ValueError(f"incomplete sample group for {prompt_id}: {path}")
        rewards = []
        for sample in samples:
            token_ids = sample.get("token_ids", [])
            logprobs = sample.get("processed_behavior_logprobs", [])
            reward = float(sample.get("reward"))
            if (
                not token_ids
                or len(token_ids) != len(logprobs)
                or not all(isinstance(token, int) for token in token_ids)
                or not math.isfinite(reward)
                or not all(math.isfinite(float(value)) for value in logprobs)
            ):
                raise ValueError(f"invalid trajectory values for {prompt_id}: {path}")
            rewards.append(reward)
        if len(set(rewards)) > 1:
            nonzero += 1
            nonzero_families.add(families[prompt_id])
    if payload.get("nonzero_reward_variance_groups") != nonzero:
        raise ValueError(f"nonzero group count mismatch: {path}")
    return payload, {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256(path),
        "platform": platform,
        "restart_index": restart,
        "seed_base": expected_seed,
        "scheduled_device_index": expected_device,
        "prompt_count": 108,
        "trajectory_count": 864,
        "nonzero_advantage_groups": nonzero,
        "nonzero_advantage_families": sorted(nonzero_families),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "repository_commit": payload.get("repository_commit"),
        "status": "valid",
    }


def source_by_platform(payload: dict, platform: str, restart: int) -> dict:
    matches = [
        source
        for source in payload.get("sources", [])
        if source.get("source_platform") == platform
        and source.get("source_restart_index") == restart
    ]
    if len(matches) != 1:
        raise ValueError(f"cross-score source is not unique: {platform} r{restart}")
    return matches[0]


def validate_cross_score(
    path: Path,
    scoring_platform: str,
    restart: int,
    a100_path: Path,
    npu_path: Path,
    trajectories: dict[str, dict],
    tolerance: float,
) -> dict:
    payload = load_json(path)
    expected_hashes = [sha256(a100_path), sha256(npu_path)]
    checks = {
        "status": payload.get("status") == "success",
        "platform": payload.get("platform") == scoring_platform,
        "restart_index": payload.get("restart_index") == restart,
        "source_count": payload.get("source_count") == 2,
        "trajectory_count": payload.get("trajectory_count") == 1728,
        "trajectory_sha256": payload.get("trajectory_sha256") == expected_hashes,
    }
    if not all(checks.values()):
        raise ValueError(f"cross-score metadata failed for {path}: {checks}")
    maximum_self_gap = float(payload.get("maximum_self_reporter_abs_gap"))
    if not math.isfinite(maximum_self_gap) or maximum_self_gap > tolerance:
        raise ValueError(f"cross-score self sentinel failed for {path}")
    observed_maximum = 0.0
    for source_platform in ("a100", "910b"):
        source = source_by_platform(payload, source_platform, restart)
        trajectory_records = trajectories[source_platform]["records"]
        if len(source.get("records", [])) != len(trajectory_records):
            raise ValueError(f"cross-score record count mismatch: {path}")
        for scored_record, source_record in zip(
            source["records"], trajectory_records, strict=True
        ):
            if scored_record["prompt_id"] != source_record["prompt_id"]:
                raise ValueError(f"cross-score prompt order mismatch: {path}")
            scored_samples = sorted(
                scored_record["samples"], key=lambda item: int(item["sample_id"])
            )
            source_samples = sorted(
                source_record["samples"], key=lambda item: int(item["sample_id"])
            )
            for scored, original in zip(scored_samples, source_samples, strict=True):
                values = [float(value) for value in scored["cross_scored_logprobs"]]
                if scored["token_ids"] != original["token_ids"] or len(values) != len(
                    original["token_ids"]
                ):
                    raise ValueError(f"cross-score trajectory mismatch: {path}")
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"non-finite cross-score: {path}")
                if source_platform == scoring_platform:
                    gap = max(
                        abs(left - float(right))
                        for left, right in zip(
                            values,
                            original["processed_behavior_logprobs"],
                            strict=True,
                        )
                    )
                    observed_maximum = max(observed_maximum, gap)
    if abs(observed_maximum - maximum_self_gap) > 1e-12:
        raise ValueError(f"reported self gap differs from recomputation: {path}")
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256(path),
        "platform": scoring_platform,
        "restart_index": restart,
        "source_count": 2,
        "trajectory_count": 1728,
        "maximum_self_reporter_abs_gap": maximum_self_gap,
        "pm_rq_tolerance": tolerance,
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "repository_commit": payload.get("repository_commit"),
        "status": "valid",
    }


def execute(args: argparse.Namespace) -> dict:
    config = load_json(args.config)
    if sha256(args.parent_config) != config["parent_protocol"]["compute_host_sha256"]:
        raise ValueError("PM-AR parent config hash differs from the PM-AR2 freeze")
    if sha256(args.workload) != config["workload"]["sha256"]:
        raise ValueError("PM-AR workload hash differs from the PM-AR2 freeze")
    prompt_ids, families = load_workload(args.workload)
    if len(prompt_ids) != 108:
        raise ValueError("PM-AR workload does not contain 108 prompts")
    rollout_summaries = []
    cross_summaries = []
    for restart in range(5):
        paths = {
            platform: args.cube_root / f"{platform}_r{restart}.json"
            for platform in ("a100", "910b")
        }
        trajectories = {}
        for platform, path in paths.items():
            payload, summary = validate_rollout(
                path, platform, restart, config, prompt_ids, families
            )
            trajectories[platform] = payload
            rollout_summaries.append(summary)
        for platform in ("a100", "910b"):
            cross_summaries.append(
                validate_cross_score(
                    args.cube_root / "cross_score" / f"{platform}_r{restart}.json",
                    platform,
                    restart,
                    paths["a100"],
                    paths["910b"],
                    trajectories,
                    config["cross_score_qualification"][
                        "required_self_reporter_abs_tolerance"
                    ],
                )
            )
    return {
        "schema": "policybraid.pm_ar2.cube_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "config_sha256": sha256(args.config),
        "parent_config_sha256": sha256(args.parent_config),
        "workload_sha256": sha256(args.workload),
        "rollout_processes": rollout_summaries,
        "cross_score_processes": cross_summaries,
        "rollout_process_count": len(rollout_summaries),
        "cross_score_process_count": len(cross_summaries),
        "total_free_trajectories": sum(
            item["trajectory_count"] for item in rollout_summaries
        ),
        "maximum_self_reporter_abs_gap": max(
            item["maximum_self_reporter_abs_gap"] for item in cross_summaries
        ),
        "pm_rq_sentinels_pass": True,
        "status": "valid",
        "scientific_claim_adjudicated": False,
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
    print(
        json.dumps(
            {
                "status": payload["status"],
                "free_trajectories": payload["total_free_trajectories"],
                "max_self_gap": payload["maximum_self_reporter_abs_gap"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
