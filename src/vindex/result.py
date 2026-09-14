"""The shared result shape every metric in this package returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricResult:
    """The output of running one metric on one (prompt, response) pair.

    score  : float in [0, 1]. Higher is better, regardless of metric.
    passed : bool. The metric's own pass/fail call at its default threshold.
    label  : short metric-specific outcome string, e.g. "script_mismatch".
             Stable and matchable -- code should branch on this, not on
             parsing `reason`.
    reason : one sentence, written for whoever is staring at a red CI job
             at 11pm with no other context. State what happened and why,
             in plain language -- not "score below threshold" but
             "response was in Devanagari; prompt was Romanized Hindi."
    detail : raw evidence backing score/label/reason (counts, thresholds,
             intermediate values) for anyone who needs to go deeper than
             the one-sentence reason.
    """

    score: float
    passed: bool
    label: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")
        if not self.label:
            raise ValueError("label must be a non-empty string")
        if not self.reason:
            raise ValueError("reason must be a non-empty string")
