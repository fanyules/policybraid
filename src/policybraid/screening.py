from __future__ import annotations

from collections import Counter
from typing import Any

from policybraid.prompts import FAMILIES


def adjudicate_screening(
    candidates: list[dict[str, Any]],
    screening: dict[str, Any],
    samples_per_candidate: int,
    selected_per_family: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {candidate["prompt_id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate IDs are not unique")
    if screening.get("status") != "success":
        raise ValueError("screening process is not complete")
    rows = screening.get("candidates", [])
    if len(rows) != len(candidates):
        raise ValueError("screening candidate count differs from the frozen workload")

    eligibility: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        prompt_id = row.get("prompt_id")
        if prompt_id in seen or prompt_id not in by_id:
            raise ValueError(f"unexpected or duplicate screening ID: {prompt_id}")
        seen.add(prompt_id)
        candidate = by_id[prompt_id]
        if row.get("task_family") != candidate["task_family"]:
            raise ValueError(f"family mismatch for {prompt_id}")
        samples = row.get("samples", [])
        complete = (
            len(samples) == samples_per_candidate
            and sorted(sample.get("sample_id") for sample in samples)
            == list(range(samples_per_candidate))
        )
        verifier_healthy = complete and all(
            sample.get("verifier_status") == "completed" for sample in samples
        )
        rewards = [sample.get("reward") for sample in samples]
        rewards_valid = complete and all(reward in (0, 1) for reward in rewards)
        nonzero_variance = rewards_valid and len(set(rewards)) == 2
        eligibility.append(
            {
                "prompt_id": prompt_id,
                "task_family": candidate["task_family"],
                "complete": complete,
                "verifier_healthy": verifier_healthy,
                "reward_counts": dict(sorted(Counter(rewards).items()))
                if rewards_valid
                else None,
                "nonzero_reward_variance": nonzero_variance,
                "eligible": verifier_healthy and nonzero_variance,
            }
        )

    selected_ids: list[str] = []
    family_counts = {}
    for family in FAMILIES:
        eligible_ids = sorted(
            record["prompt_id"]
            for record in eligibility
            if record["task_family"] == family and record["eligible"]
        )
        family_counts[family] = {
            "eligible": len(eligible_ids),
            "required": selected_per_family,
        }
        selected_ids.extend(eligible_ids[:selected_per_family])

    sufficient = all(
        counts["eligible"] >= selected_per_family for counts in family_counts.values()
    )
    selected_set = set(selected_ids) if sufficient else set()
    selected = [
        candidate for candidate in candidates if candidate["prompt_id"] in selected_set
    ]
    adjudication = {
        "schema": "policybraid.pm_a.screening_adjudication.v1",
        "status": "passed" if sufficient else "workload_insufficient",
        "selection_inputs": ["verifier_healthy", "nonzero_reward_variance"],
        "backend_difference_used": False,
        "screening_trajectories_excluded": True,
        "family_counts": family_counts,
        "selected_prompt_ids": [candidate["prompt_id"] for candidate in selected],
        "selected_count": len(selected),
        "eligibility": sorted(eligibility, key=lambda record: record["prompt_id"]),
    }
    return adjudication, selected

