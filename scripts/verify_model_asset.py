#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a frozen model asset")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
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


def verify(manifest_path: Path, model_dir: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = {entry["name"] for entry in manifest["files"]}
    actual_names = {entry.name for entry in model_dir.iterdir() if entry.is_file()}
    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    checks = []
    for entry in manifest["files"]:
        path = model_dir / entry["name"]
        if not path.is_file():
            continue
        observed_size = path.stat().st_size
        observed_sha256 = sha256(path)
        checks.append(
            {
                "name": entry["name"],
                "size_match": observed_size == entry["bytes"],
                "sha256_match": observed_sha256 == entry["sha256"],
                "observed_bytes": observed_size,
                "observed_sha256": observed_sha256,
            }
        )
    passed = (
        not unexpected
        and not missing
        and len(checks) == len(manifest["files"])
        and all(check["size_match"] and check["sha256_match"] for check in checks)
    )
    return {
        "schema": "policybraid.model_asset_verification.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "hostname": socket.gethostname(),
        "asset_id": manifest["asset_id"],
        "asset_revision": manifest["revision"],
        "manifest_sha256": sha256(manifest_path),
        "unexpected_entries": unexpected,
        "missing_entries": missing,
        "checks": checks,
        "status": "success" if passed else "failed",
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = verify(args.manifest, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
