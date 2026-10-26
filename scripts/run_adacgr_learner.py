#!/usr/bin/env python3
"""Replay learner-specific AdaCGR stage decisions from chronological losses."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adacgr import AdaptiveStageController, MeanCUSUMSignal


LEARNER_NAMES = {
    "ft": "AdaCGR-FT",
    "peso": "AdaCGR-PESO",
    "dact": "AdaCGR-DACT",
    "reformer": "AdaCGR-Reformer",
    "lsat": "AdaCGR-LSAT",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Input must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON loss stream")
    parser.add_argument("--output", type=Path, required=True, help="JSON report")
    parser.add_argument("--learner", choices=tuple(LEARNER_NAMES), required=True)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--variance-floor", type=float, default=0.05**2)
    parser.add_argument("--min-feedback", type=int, default=256)
    parser.add_argument("--min-budget", type=int, default=16)
    args = parser.parse_args()

    payload = load_json(args.input)
    signal = MeanCUSUMSignal(
        threshold=args.threshold,
        tolerance=args.tolerance,
        variance_floor=args.variance_floor,
    )
    calibration = payload.get("calibration_losses")
    if calibration is None:
        raise ValueError("Input must contain calibration_losses")
    signal.calibrate(calibration)
    controller = AdaptiveStageController(
        signal,
        min_feedback=args.min_feedback,
        min_budget=args.min_budget,
    )

    decisions = []
    for index, batch in enumerate(payload.get("batches", []), start=1):
        if not isinstance(batch, dict):
            raise ValueError(f"Batch {index} must be an object")
        decision = controller.observe_batch(
            batch.get("losses", []),
            feedback_count=int(batch["feedback_count"]),
            eligible_count=int(batch["eligible_count"]),
            available_steps=int(batch["available_steps"]),
        )
        row = asdict(decision)
        row["batch_index"] = index
        if decision.accepted:
            post_fit = batch.get("post_fit_calibration_losses")
            if post_fit is None:
                raise ValueError(
                    f"Accepted batch {index} needs post_fit_calibration_losses"
                )
            controller.complete_fit(post_fit)
            row["reference_renewed"] = True
        else:
            row["reference_renewed"] = False
        decisions.append(row)

    report = {
        "method": LEARNER_NAMES[args.learner],
        "learner": args.learner,
        "policy": "MeanCUSUM adaptive stage formation",
        "config": {
            "threshold": args.threshold,
            "tolerance": args.tolerance,
            "variance_floor": args.variance_floor,
            "min_feedback": args.min_feedback,
            "min_budget": args.min_budget,
        },
        "decisions": decisions,
        "accepted_boundaries": [
            row["batch_index"] for row in decisions if row["accepted"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "accepted_boundaries": report["accepted_boundaries"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
