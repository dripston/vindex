"""The shared result shape every metric in this package returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def coerce_text(value: Any) -> str:
    """Coerce a metric's text input to a string.

    Strings pass through unchanged. ``None`` and any non-string value
    (e.g. the ``NaN`` pandas uses for an empty cell) become ``""``, which
    every metric reports with its ``"empty"`` label rather than crashing.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ""


@dataclass(frozen=True, slots=True)
class MetricResult:
    """The result of running one metric on one input.

    Attributes:
        score: Float in [0, 1]; higher is better for every metric.
        passed: The metric's pass/fail verdict at its default threshold.
        label: Short, stable, metric-specific outcome such as
            ``"script_mismatch"``. Branch on this, not on ``reason``.
        reason: One plain-language sentence explaining what happened, e.g.
            "response was in Devanagari; prompt was Romanized Hindi."
        detail: Supporting evidence (counts, thresholds, intermediate
            values, model ids).

    Raises:
        ValueError: If ``score`` is outside [0, 1], or ``label`` or
            ``reason`` is empty.
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
