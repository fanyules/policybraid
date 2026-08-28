#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from policybraid.prompts import build_candidates, jsonl_bytes
from policybraid.verifiers import validate_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen PM-A candidates")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "pm_a.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "workloads" / "pm_a_candidates.jsonl",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the existing output is byte-for-byte reproducible",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["workload"]["candidate_generator_seed"])
    records = build_candidates(seed)
    for record in records:
        validate_candidate(record)
    payload = jsonl_bytes(records)
    digest = hashlib.sha256(payload).hexdigest()
    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        if args.output.read_bytes() != payload:
            raise RuntimeError("candidate file differs from deterministic generator")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(
        json.dumps(
            {"records": len(records), "sha256": digest, "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

