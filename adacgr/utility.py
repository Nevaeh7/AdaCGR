"""Paired utility policy for selective candidate replacement."""

from __future__ import annotations

from typing import Any

import numpy as np


class PairedUtilitySignal:
    """Accumulate standardized candidate gains on shared validation events.

    ``before`` and ``after`` must be losses for the incumbent and candidate on
    the same events. Positive paired differences favor the candidate. The
    evidence is reset only after the caller actually deploys the candidate.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        tolerance: float = 0.05,
        dispersion_floor: float = 0.05,
        max_users: int = 128,
    ) -> None:
        if threshold <= 0 or tolerance < 0 or dispersion_floor <= 0 or max_users < 1:
            raise ValueError("Invalid utility signal parameters")
        self.threshold = float(threshold)
        self.tolerance = float(tolerance)
        self.dispersion_floor = float(dispersion_floor)
        self.max_users = int(max_users)
        self.evidence = 0.0
        self.last_observation: dict[str, Any] | None = None

    def observe(
        self,
        before: Any,
        after: Any,
        users: Any,
    ) -> tuple[bool, float]:
        """Evaluate one paired validation window.

        Returns ``(replace, normalized_gain)``. The caller is responsible for
        copying candidate parameters into the serving model when ``replace``
        is true, then calling :meth:`reset_after_replacement`.
        """
        incumbent = np.asarray(before, dtype=float)
        candidate = np.asarray(after, dtype=float)
        user_ids = np.asarray(users)
        if incumbent.ndim != 1 or candidate.ndim != 1 or user_ids.ndim != 1:
            raise ValueError("Paired observations must be one-dimensional")
        if not (len(incumbent) == len(candidate) == len(user_ids)):
            raise ValueError("Paired losses and users must have equal length")
        if not np.isfinite(incumbent).all() or not np.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite paired utility")
        if not len(incumbent):
            self.last_observation = {
                "replace": False,
                "gain": 0.0,
                "evidence_before": self.evidence,
                "evidence_after": self.evidence,
            }
            return False, 0.0

        delta = incumbent - candidate
        effective = max(1, min(len(delta), len(np.unique(user_ids)), self.max_users))
        standard_error = max(float(delta.std()), self.dispersion_floor) / np.sqrt(effective)
        gain = float(delta.mean()) / standard_error
        evidence_before = self.evidence
        self.evidence = max(0.0, self.evidence + gain - self.tolerance)
        replace = bool(self.evidence >= self.threshold)
        self.last_observation = {
            "replace": replace,
            "gain": gain,
            "paired_mean": float(delta.mean()),
            "paired_std": float(delta.std()),
            "effective_users": effective,
            "evidence_before": evidence_before,
            "evidence_after": self.evidence,
        }
        return replace, gain

    def reset_after_replacement(self) -> None:
        """Reset utility evidence after the candidate becomes the incumbent."""
        self.evidence = 0.0
