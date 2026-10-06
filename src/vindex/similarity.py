"""calibrated_similarity: semantic closeness with a calibrated threshold.

Encodes a response and a gold answer with a multilingual sentence
encoder and compares their cosine similarity against a threshold
calibrated for that encoder and language (see :mod:`vindex.calibration`),
instead of an uncalibrated 0.5. At 0.5, most shipped (encoder, language)
configurations discriminate correct from wrong answers at or near
chance; calibrated, the better encoders reach 0.63-0.90 accuracy.

This is a semantic-closeness check, not a correctness judge: two answers
can be topically similar yet disagree on the fact. Use
:func:`vindex.indic_judge` for correctness. A gold reference is required.

Requires the ``similarity`` extra (``pip install vindex[similarity]``);
without it, calling the metric raises ``ImportError``.

Loaded encoders are cached for the lifetime of the process. To bound
memory when using several encoders in one process, call
``vindex.similarity._encoder_instances.clear()`` between them.
"""

from __future__ import annotations

import math

from vindex.calibration import DEFAULT_ENCODER, get_threshold
from vindex.encoder import Encoder, cosine_similarity
from vindex.result import MetricResult, coerce_text

_KNOWN_LANGUAGES = frozenset({"en", "hi", "hinglish"})

# Process-lifetime encoder cache, never evicted (see module docstring).
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
    min_auc: float = 0.7,
) -> MetricResult:
    """Check whether a response is semantically close to a gold answer.

    The response passes if its cosine similarity to ``gold`` (clamped to
    [0, 1]) is at or above the shipped threshold for
    ``(encoder_name, language)``. If no calibration exists for that pair,
    a 0.5 threshold is used and the result is labelled
    ``"similar_uncalibrated"`` / ``"dissimilar_uncalibrated"``.

    Cells whose threshold-independent ROC AUC is below ``min_auc`` are
    labelled ``"low_discrimination"`` with ``passed=False`` regardless of
    the similarity score, because their fitted threshold can look
    reasonable in-sample while the encoder cannot actually discriminate
    (e.g. multilingual-e5-base/hi: 0.632 fitted accuracy, AUC 0.500). Of
    the 15 shipped cells, only mpnet-v2/en, e5-base/en, and
    MiniLM-L6/hinglish clear the default of 0.7. Any calibration warnings
    for the cell are copied into ``detail["calibration_warnings"]`` and
    appended to ``reason``.

    Args:
        gold: Reference answer. ``None`` is treated as empty.
        response: Response to score. ``None`` is treated as empty.
        language: One of ``"en"``, ``"hi"``, ``"hinglish"``.
        encoder_name: Hugging Face model id. Defaults to
            ``calibration.DEFAULT_ENCODER``.
        allow_muril: Permit ``google/muril-base-cased`` (see
            ``calibration.MURIL_WARNING``).
        min_auc: Minimum shipped ROC AUC required for a similar/dissimilar
            verdict. Set to 0.0 to disable, only after validating the cell
            on your own held-out data.

    Returns:
        A :class:`vindex.result.MetricResult` labelled ``"similar"``,
        ``"dissimilar"``, ``"similar_uncalibrated"``,
        ``"dissimilar_uncalibrated"``, ``"low_discrimination"``, or
        ``"empty"``. ``detail`` includes the encoder, threshold,
        ``calibrated`` flag, ``unclamped_cosine_similarity``, and
        ``raw_cosine_similarity`` (the clamped value used for scoring;
        the name is kept for compatibility).

    Raises:
        ValueError: If ``language`` is not supported, or if the cosine
            similarity is NaN (a corrupted embedding or cache entry; clear
            ``vindex.encoder.CACHE_ROOT`` and retry).
        MurilWithoutOverrideError: If MuRIL is requested without
            ``allow_muril=True``.
        ImportError: If the ``similarity`` extra is not installed.
    """
    gold = coerce_text(gold)
    response = coerce_text(response)

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
    if math.isnan(score):
        # The clamp below would turn NaN into 1.0, so fail loudly instead.
        raise ValueError(
            f"cosine similarity between gold and response was NaN for encoder "
            f"{encoder_name!r} -- this indicates a corrupted embedding (e.g. a "
            "bad cached .npy file, or fp16 overflow), not a valid similarity "
            "score. Clear the encoder cache (see vindex.encoder.CACHE_ROOT) and "
            "retry."
        )
    raw_cosine_similarity = score
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
        # "raw_cosine_similarity" holds the clamped score; the name is
        # kept for backwards compatibility.
        "unclamped_cosine_similarity": raw_cosine_similarity,
        "raw_cosine_similarity": score,
    }
    if calibration is not None:
        detail["calibration_accuracy_at_threshold"] = calibration.accuracy_at_threshold
        detail["calibration_n_cases"] = calibration.n_cases
        if calibration.warnings:
            detail["calibration_warnings"] = calibration.warnings

    if calibrated:
        assert calibration is not None  # calibrated is only True when calibration was found
        if calibration.roc_auc is not None and calibration.roc_auc < min_auc:
            # AUC below the bar: no confident verdict either way.
            detail["roc_auc"] = calibration.roc_auc
            return MetricResult(
                score=score,
                passed=False,
                label="low_discrimination",
                reason=(
                    f"({encoder_name}, {language})'s real ROC AUC is "
                    f"{calibration.roc_auc:.3f}, below min_auc={min_auc} -- this "
                    "(encoder, language) pair has little to no real ability to "
                    "discriminate correct from wrong answers, regardless of "
                    f"where cosine similarity {score:.3f} falls relative to the "
                    f"calibrated threshold {threshold:.3f}. Pass "
                    "min_auc=0.0 to disable this check."
                ),
                detail=detail,
            )
        reason = (
            f"cosine similarity {score:.3f} "
            f"{'>=' if passed else '<'} calibrated threshold "
            f"{threshold:.3f} for ({encoder_name}, {language})."
        )
        if calibration.warnings:
            # Surface degenerate-calibration warnings in reason too.
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
