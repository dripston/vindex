"""
calibrated_similarity: is a response semantically close to a gold
answer, using a per-encoder, per-language calibrated threshold instead
of an uncalibrated 0.5 (Milestone 3).

THE ARGUMENT FOR THIS METRIC (Milestone 3.5)

At the naive default of 0.5 cosine similarity, 11 of the 15 (encoder,
language) configurations in this package's own calibration experiment
score at or near chance (<=0.5 accuracy) for correct-vs-wrong
discrimination -- a 0.5 threshold is not a safe default, it is close
to a coin flip. Calibrated per (encoder, language), the better
encoders reach 0.63 to 0.90 accuracy on the same cases; the single
best cell (multilingual-e5-base, English) reaches 0.90. See
vindex.calibration's module docstring and
experiments/results_clean/discrimination_summary.csv for the full
table this is drawn from, and the note on why this package's own
count differs slightly from an earlier "13 of 15" estimate written
before this exact number was computed -- see calibration.py.

WHAT THIS METRIC DOES AND DOES NOT DO

calibrated_similarity(gold, response, language) encodes both texts
with a multilingual sentence encoder, computes cosine similarity, and
compares it against a calibrated threshold for (encoder, language) --
falling back to 0.5 with a stated caveat if no calibration exists for
that combination. It is a semantic-closeness check, not a factual-
correctness judge: two answers can be topically similar and still
disagree on the actual fact (see BUILD_PLAN.md's indic_judge
milestone for that problem). It requires a gold reference, unlike
script_adherence (Milestone 1) which does not.

DEPENDENCIES: sentence-transformers, transformers, and torch are NOT
core vindex dependencies (see vindex.encoder's module docstring) --
install via `pip install vindex[similarity]`. Calling this function
without them installed raises ImportError from the underlying import,
not a custom vindex error, since there is nothing vindex-specific to
add to that message.

DEGENERATE-CELL WARNINGS (fixed after an independent outside review
found this metric silently returned a clean "similar"/"dissimilar"
label even for shipped (encoder, language) cells with no real
discrimination, e.g. LaBSE/en): calibration.CALIBRATION_TABLE's
entries now carry real, calibrate()-computed `.warnings` (see
calibration.py's module comment above CALIBRATION_TABLE for exactly
what triggers one and what doesn't). When the cell used has any, this
function copies them into `detail["calibration_warnings"]` and appends
them to `reason` -- so a caller reading only `reason`, or only
`detail`, both see it, not just someone who separately reads the
README's AUC table. Note this does NOT catch every weak cell: it
reuses calibrate()'s own guard (same-sample fitted accuracy at or
below chance), which is a weaker signal than the threshold-independent
ROC AUC in README.md's "argument for calibrated_similarity" table --
a cell can fit above chance in-sample and still have near-chance real
AUC (e.g. multilingual-e5-base/hi), and that case is NOT warned about
here. Check the README's AUC table for the full picture; this warning
only catches the most degenerate cells.
"""

from __future__ import annotations

from vindex.calibration import DEFAULT_ENCODER, get_threshold
from vindex.encoder import Encoder, cosine_similarity
from vindex.result import MetricResult

_KNOWN_LANGUAGES = frozenset({"en", "hi", "hinglish"})

_encoder_instances: dict[tuple[str, bool], Encoder] = {}


def _get_loaded_encoder(encoder_name: str, allow_muril: bool) -> Encoder:
    key = (encoder_name, allow_muril)
    if key not in _encoder_instances:
        _encoder_instances[key] = Encoder(encoder_name, allow_muril=allow_muril).load()
    return _encoder_instances[key]


def calibrated_similarity(
    gold: str | None,
    response: str | None,
    language: str,
    encoder_name: str = "",
    allow_muril: bool = False,
) -> MetricResult:
    """Is `response` semantically close enough to `gold`, per a
    calibrated threshold for (encoder_name, language)?

    language must be one of "en", "hi", "hinglish" -- the three
    variants the shipped calibration table covers. encoder_name
    defaults to calibration.DEFAULT_ENCODER (Milestone 3.2's sensible
    default). Passing google/muril-base-cased raises
    MurilWithoutOverrideError unless allow_muril=True -- see
    vindex.calibration.MURIL_WARNING.

    If no shipped calibration exists for (encoder_name, language), the
    metric falls back to a 0.5 threshold and says so plainly in
    `reason` and `detail` -- it does not silently pretend to be
    calibrated when it isn't.
    """
    gold = gold or ""
    response = response or ""

    if gold.strip() == "" or response.strip() == "":
        return MetricResult(
            score=0.0,
            passed=False,
            label="empty",
            reason="gold or response is empty.",
            detail={"gold": gold, "response": response},
        )

    if language not in _KNOWN_LANGUAGES:
        raise ValueError(
            f"language must be one of {sorted(_KNOWN_LANGUAGES)}, got {language!r}."
        )

    encoder_name = encoder_name or DEFAULT_ENCODER
    encoder = _get_loaded_encoder(encoder_name, allow_muril)

    gold_vec = encoder.encode(gold, is_query=False)
    response_vec = encoder.encode(response, is_query=True)
    score = cosine_similarity(gold_vec, response_vec)
    score = max(0.0, min(1.0, score))

    try:
        calibration = get_threshold(encoder_name, language)
        threshold = calibration.threshold
        calibrated = True
    except KeyError:
        threshold = 0.5
        calibration = None
        calibrated = False

    passed = score >= threshold
    detail = {
        "encoder": encoder_name,
        "language": language,
        "threshold": threshold,
        "calibrated": calibrated,
        "raw_cosine_similarity": score,
    }
    if calibration is not None:
        detail["calibration_accuracy_at_threshold"] = calibration.accuracy_at_threshold
        detail["calibration_n_cases"] = calibration.n_cases
        if calibration.warnings:
            detail["calibration_warnings"] = calibration.warnings

    if calibrated:
        reason = (
            f"cosine similarity {score:.3f} "
            f"{'>=' if passed else '<'} calibrated threshold "
            f"{threshold:.3f} for ({encoder_name}, {language})."
        )
        if calibration is not None and calibration.warnings:
            # Found by an independent outside review: CalibratedThreshold
            # gained a .warnings field for calibrate()'s own callers, but
            # the shipped CALIBRATION_TABLE cells never surfaced any --
            # so a caller using the shipped table for a degenerate cell
            # (e.g. LaBSE/en, fitted accuracy at chance) got a clean-
            # looking "similar"/"dissimilar" result with no indication
            # the underlying calibration itself is suspect. Now:
            # whatever calibrate() would have warned about this cell's
            # own data is surfaced here too, in both detail and reason.
            reason += " WARNING: " + " ".join(calibration.warnings)
        label = "similar" if passed else "dissimilar"
    else:
        reason = (
            f"no shipped calibration for ({encoder_name}, {language}); "
            f"used uncalibrated default 0.5. cosine similarity {score:.3f} "
            f"{'>=' if passed else '<'} 0.5."
        )
        label = "similar_uncalibrated" if passed else "dissimilar_uncalibrated"

    return MetricResult(score=score, passed=passed, label=label, reason=reason, detail=detail)
