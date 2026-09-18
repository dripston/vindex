"""The shared result shape every metric in this package returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def coerce_text(value: Any) -> str:
    """Coerce a metric's text input to a string: `None` and any
    non-string value both become "" (treated the same as an empty
    string by every metric's own empty-input handling), a real string
    passes through unchanged.

    FIXED (a real bug, found by an independent outside review): every
    metric's own text arguments were typed `str | None` and coerced
    with a plain `value or ""`, which only handles `None` and the
    empty string -- any OTHER falsy-adjacent-looking non-string value
    passed straight through uncoerced. `float("nan")` is truthy in
    Python (only 0.0/None/""/etc. are falsy), so `float("nan") or ""`
    returns `float("nan")` unchanged, and the first `.strip()` call
    inside the metric crashed with `AttributeError: 'float' object has
    no attribute 'strip'`. This is exactly the shape pandas produces
    for an empty cell (`df["column"]` puts `nan`, not `None` or `""`,
    in a blank row) -- a real, likely input for an eval library called
    from a dataframe, not a contrived type-violation test. A
    non-string, non-None value is not a valid text input either way,
    so it is treated the same as an empty string rather than crashing
    -- this is consistent with the label="empty" path every metric
    already has for an actually-empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ""


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
