#!/usr/bin/env python3
"""Replay AdaCGR-U paired-utility deployment decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adacgr import PairedUtilitySignal


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Input must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON paired-loss windows")
    parser.add_argument("--output", type=Path, required=True, help="JSON report")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--dispersion-floor", type=float, default=0.05)
    parser.add_argument("--max-users", type=int, default=128)
    args = parser.parse_args()

    payload = load_json(args.input)
    signal = PairedUtilitySignal(
        threshold=args.threshold,
        tolerance=args.tolerance,
        dispersion_floor=args.dispersion_floor,
        max_users=args.max_users,
    )
    decisions = []
    for index, window in enumerate(payload.get("windows", []), start=1):
        if not isinstance(window, dict):
            raise ValueError(f"Window {index} must be an object")
        replace, gain = signal.observe(
            window.get("before", []),
            window.get("after", []),
            window.get("users", []),
        )
        row = {
            "window_index": index,
            "replace": replace,
            "gain": gain,
            "evidence_after": signal.evidence,
        }
        if replace:
            signal.reset_after_replacement()
            row["evidence_reset"] = True
        else:
            row["evidence_reset"] = False
        decisions.append(row)

    report = {
        "method": "AdaCGR-U",
        "policy": "persistent candidate with paired-utility deployment",
        "config": {
            "threshold": args.threshold,
            "tolerance": args.tolerance,
            "dispersion_floor": args.dispersion_floor,
            "max_users": args.max_users,
        },
        "decisions": decisions,
        "deployments": [
            row["window_index"] for row in decisions if row["replace"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "deployments": report["deployments"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
