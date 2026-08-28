from __future__ import annotations

from typing import Any

from policybraid.prompts import FAMILIES


def select_maximal_balanced(
    candidates: list[dict[str, Any]],
    screening_adjudication: dict[str, Any],
    expected_per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if screening_adjudication.get("status") != "workload_insufficient":
        raise ValueError("PM-AR requires the frozen revision-3 insufficient result")
    eligibility_rows = screening_adjudication.get("eligibility", [])
    eligibility = {}
    for row in eligibility_rows:
        prompt_id = row.get("prompt_id")
        if prompt_id in eligibility:
            raise ValueError(f"duplicate eligibility row: {prompt_id}")
        eligibility[prompt_id] = row

    candidates_by_id = {candidate["prompt_id"]: candidate for candidate in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("candidate IDs are not unique")
    if set(eligibility) != set(candidates_by_id):
        raise ValueError("eligibility rows do not match the frozen candidate set")

    eligible_by_family: dict[str, list[str]] = {}
    for family in FAMILIES:
        eligible_by_family[family] = sorted(
            prompt_id
            for prompt_id, row in eligibility.items()
            if row.get("task_family") == family
            and row.get("eligible") is True
            and row.get("verifier_healthy") is True
            and row.get("nonzero_reward_variance") is True
        )
    balanced_count = min(len(ids) for ids in eligible_by_family.values())
    if balanced_count != expected_per_family:
        raise ValueError(
            f"frozen maximal-balanced count is {balanced_count}, expected {expected_per_family}"
        )

    selected_ids_by_family = {
        family: ids[:balanced_count] for family, ids in eligible_by_family.items()
    }
    surplus_ids_by_family = {
        family: ids[balanced_count:] for family, ids in eligible_by_family.items()
    }
    selected_ids = {
        prompt_id
        for ids in selected_ids_by_family.values()
        for prompt_id in ids
    }
    selected = [
        candidate
        for family in FAMILIES
        for candidate in sorted(
            (
                candidate
                for candidate in candidates
                if candidate["task_family"] == family
                and candidate["prompt_id"] in selected_ids
            ),
            key=lambda candidate: candidate["prompt_id"],
        )
    ]
    expected_total = balanced_count * len(FAMILIES)
    if len(selected) != expected_total:
        raise RuntimeError("selected workload has an unexpected size")

    anchors_by_family = {
        family: ids[:8] for family, ids in selected_ids_by_family.items()
    }
    metadata = {
        "balanced_count_per_family": balanced_count,
        "selected_count": len(selected),
        "selected_ids_by_family": selected_ids_by_family,
        "surplus_ids_by_family": surplus_ids_by_family,
        "surplus_count": sum(len(ids) for ids in surplus_ids_by_family.values()),
        "surplus_replacement_allowed": False,
        "anchor_ids_by_family": anchors_by_family,
        "selection_order": "eligible_true_then_prompt_id_ascending",
        "reward_or_variance_magnitude_used": False,
        "backend_metric_used": False,
    }
    return selected, metadata

