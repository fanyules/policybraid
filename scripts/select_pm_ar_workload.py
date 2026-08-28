#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import jsonl_bytes, load_jsonl
from policybraid.reentry import select_maximal_balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the PM-AR maximal-balanced revision-3 workload"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "pm_ar.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_a_candidates.jsonl",
    )
    parser.add_argument(
        "--screening-adjudication",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "pm_a"
            / "screening"
            / "candidate_set_81261a00"
            / "PM_A0_ADJUDICATION.json"
        ),
    )
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_ar_selected.jsonl",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "pm_ar" / "PM_AR0_SELECTION.json",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["source_workload"]
    if sha256(args.candidates) != source["candidate_sha256"]:
        raise ValueError("candidate file differs from the PM-AR source lock")
    if sha256(args.screening_adjudication) != source["adjudication_sha256"]:
        raise ValueError("screening adjudication differs from the PM-AR source lock")

    candidates = load_jsonl(args.candidates)
    adjudication = json.loads(args.screening_adjudication.read_text(encoding="utf-8"))
    selected, metadata = select_maximal_balanced(
        candidates,
        adjudication,
        expected_per_family=config["workload"]["prompts_per_family"],
    )
    selected_bytes = jsonl_bytes(selected)
    manifest = {
        "schema": "policybraid.pm_ar.selection.v1",
        "created_at": config["frozen_at"],
        "repository_commit": repository_commit(),
        "status": "passed",
        "gate": "PM-AR0",
        "source_candidate_sha256": sha256(args.candidates),
        "source_adjudication_sha256": sha256(args.screening_adjudication),
        "selected_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        **metadata,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    if args.check:
        if args.selected_output.read_bytes() != selected_bytes:
            raise RuntimeError("selected workload is not reproducible")
        observed_manifest = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        for key in ("repository_commit", "created_at"):
            manifest[key] = observed_manifest[key]
        if observed_manifest != manifest:
            raise RuntimeError("selection manifest is not reproducible")
    else:
        for output in (args.selected_output, args.manifest_output):
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
        args.selected_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_output.write_bytes(selected_bytes)
        args.manifest_output.write_bytes(manifest_bytes)
    print(
        json.dumps(
            {
                "selected": metadata["selected_count"],
                "per_family": metadata["balanced_count_per_family"],
                "surplus_sealed": metadata["surplus_count"],
                "selected_sha256": manifest["selected_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

