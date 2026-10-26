"""Causal mean-CUSUM clock for adaptive stage formation.

The clock makes every decision against the calibrated reference available
before the current observation group. The reference is renewed only after an
adaptation, using losses from the post-fit calibration tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional sequence")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Nonfinite {label}")
    return array


class MeanCUSUMSignal:
    """Calibrated predictive-surprise statistic from the paper's main method.

    ``observe`` computes

    ``z_t = (mean(losses_t) - mu) / max(sigma, variance_floor)``

    and updates ``C_t = max(0, C_{t-1} + z_t - kappa)``. The reference remains
    fixed within a stage. After fitting, call ``calibrate`` with the reserved
    post-fit tail before observing the next stage.
    """

    def __init__(
        self,
        threshold: float = 2.0,
        tolerance: float = 0.1,
        variance_floor: float = 0.05**2,
    ) -> None:
        values = (threshold, tolerance, variance_floor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Signal parameters must be finite")
        if threshold <= 0 or variance_floor <= 0 or tolerance < 0:
            raise ValueError("Invalid mean-CUSUM parameters")
        self.threshold = float(threshold)
        self.tolerance = float(tolerance)
        self.variance_floor = float(variance_floor)
        self.evidence = 0.0
        self.last_observation: dict[str, Any] | None = None

    def calibrate(self, losses: Any) -> None:
        """Set the current reference from a chronological calibration tail."""
        values = _vector(losses, "calibration losses")
        if len(values) < 2:
            raise ValueError("At least two calibration losses are required")
        self.mu = float(values.mean())
        self.raw_sigma = float(values.std(ddof=0))
        self.sigma = max(self.raw_sigma, self.variance_floor**0.5)
        self.count = len(values)
        self.evidence = 0.0
        self.last_observation = None

    def reference(self) -> dict[str, float]:
        """Return the current calibration reference."""
        return {
            "mu": self.mu,
            "raw_sigma": self.raw_sigma,
            "sigma": self.sigma,
            "count": self.count,
        }

    def observe(self, losses: Any, *, active: bool = True) -> tuple[bool, float]:
        """Observe one chronological group using the prior reference."""
        values = _vector(losses, "prequential losses")
        prior = self.reference()
        count = len(values)
        evidence_before = self.evidence
        mean = float(values.mean()) if count else None
        standardized = (
            (mean - prior["mu"]) / prior["sigma"] if count else 0.0
        )
        increment = (
            standardized - self.tolerance if active and count else 0.0
        )
        self.evidence = max(0.0, evidence_before + increment)
        fired = self.evidence >= self.threshold
        self.last_observation = {
            "reference_before": prior,
            "segment_n": count,
            "segment_mean": mean,
            "standardized_statistic": standardized,
            "increment": increment,
            "evidence_before": evidence_before,
            "evidence_after": self.evidence,
            "fired": bool(fired),
            "active": bool(active),
            "reference_after": self.reference(),
        }
        return bool(fired), float(standardized)


@dataclass
class StageDecision:
    """Decision returned by :class:`AdaptiveStageController`."""

    batch: int
    fired: bool
    accepted: bool
    standardized_statistic: float
    evidence: float
    feedback_count: int
    eligible_count: int
    available_steps: int


class AdaptiveStageController:
    """Apply feedback and budget guards around a mean-CUSUM clock.

    The controller does not train a model. Once ``accepted`` is true, the
    caller fits its learner on the causal feedback buffer, obtains a reserved
    post-fit calibration tail, and passes that tail to ``complete_fit``.
    """

    def __init__(
        self,
        signal: MeanCUSUMSignal,
        *,
        min_feedback: int,
        min_budget: int,
    ) -> None:
        if min_feedback < 1 or min_budget < 1:
            raise ValueError("Stage guards must be positive")
        self.signal = signal
        self.min_feedback = int(min_feedback)
        self.min_budget = int(min_budget)
        self.batch = 0
        self.feedback_count = 0

    def observe_batch(
        self,
        losses: Any,
        *,
        feedback_count: int,
        eligible_count: int,
        available_steps: int,
        active: bool = True,
    ) -> StageDecision:
        """Observe one chronological group and evaluate boundary guards."""
        if feedback_count < 0 or eligible_count < 0 or available_steps < 0:
            raise ValueError("Counts must be non-negative")

        self.batch += 1
        self.feedback_count += int(feedback_count)
        fired, standardized = self.signal.observe(losses, active=active)
        accepted = bool(
            fired
            and eligible_count > 0
            and self.feedback_count >= self.min_feedback
            and available_steps >= self.min_budget
        )
        return StageDecision(
            batch=self.batch,
            fired=fired,
            accepted=accepted,
            standardized_statistic=standardized,
            evidence=self.signal.evidence,
            feedback_count=self.feedback_count,
            eligible_count=int(eligible_count),
            available_steps=int(available_steps),
        )

    def complete_fit(self, post_fit_calibration_losses: Any) -> None:
        """Renew the reference after fitting and begin a new causal stage."""
        self.feedback_count = 0
        self.signal.calibrate(post_fit_calibration_losses)
