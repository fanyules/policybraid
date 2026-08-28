#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import jsonl_bytes, load_jsonl
from policybraid.screening import adjudicate_screening


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate blinded PM-A0 screening")
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
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_a_selected.jsonl",
    )
    parser.add_argument(
        "--adjudication-output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "pm_a"
            / "screening"
            / "PM_A0_ADJUDICATION.json"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.adjudication_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.adjudication_output}")
    if args.selected_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.selected_output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = load_jsonl(args.candidates)
    screening = json.loads(args.screening.read_text(encoding="utf-8"))
    if screening.get("config_sha256") != _sha256(args.config):
        raise ValueError("screening result does not match the frozen PM-A config")
    if screening.get("candidates_sha256") != _sha256(args.candidates):
        raise ValueError("screening result does not match the candidate workload")
    adjudication, selected = adjudicate_screening(
        candidates,
        screening,
        samples_per_candidate=config["workload"]["samples_per_candidate"],
        selected_per_family=config["workload"]["selected_per_family"],
    )
    adjudication.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256(args.config),
            "candidates_sha256": _sha256(args.candidates),
            "screening_sha256": _sha256(args.screening),
        }
    )
    args.adjudication_output.parent.mkdir(parents=True, exist_ok=True)
    args.adjudication_output.write_text(
        json.dumps(
            adjudication,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if adjudication["status"] != "passed":
        return 2
    args.selected_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_output.write_bytes(jsonl_bytes(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

