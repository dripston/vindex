"""
Tests for vindex.similarity's calibrated_similarity metric (Milestone
3). These load a real encoder (sentence-transformers/all-MiniLM-L6-v2,
the smallest of the five calibrated encoders) and therefore need
sentence-transformers installed (`pip install vindex[similarity]`) --
skipped automatically if it isn't. Slower than the rest of the suite
by design: this is testing real encode-and-compare behavior, not
mocked logic, matching this project's preference for real data over
mocks (see experiments/scripts/validate_vindex_port.py).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("sentence_transformers")
np = pytest.importorskip("numpy")

from vindex.calibration import MURIL_MODEL_NAME  # noqa: E402
from vindex.encoder import Encoder, MurilWithoutOverrideError, _cache_path  # noqa: E402
from vindex.similarity import calibrated_similarity  # noqa: E402

_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# --- empty / validation ---


def test_calibrated_similarity_empty_gold_scores_empty() -> None:
    r = calibrated_similarity(gold="", response="hello", language="en", encoder_name=_ENCODER)
    assert r.label == "empty"
    assert r.passed is False
    assert r.score == 0.0


def test_calibrated_similarity_empty_response_scores_empty() -> None:
    r = calibrated_similarity(gold="hello", response="", language="en", encoder_name=_ENCODER)
    assert r.label == "empty"


def test_calibrated_similarity_none_gold_scores_empty() -> None:
    r = calibrated_similarity(gold=None, response="hello", language="en", encoder_name=_ENCODER)
    assert r.label == "empty"


def test_calibrated_similarity_invalid_language_raises() -> None:
    with pytest.raises(ValueError, match="language must be one of"):
        calibrated_similarity(gold="hi", response="hi", language="tamil", encoder_name=_ENCODER)


def test_calibrated_similarity_muril_without_override_raises() -> None:
    with pytest.raises(MurilWithoutOverrideError):
        calibrated_similarity(
            gold="hi", response="hi", language="en", encoder_name=MURIL_MODEL_NAME
        )


# --- real encoding: identical text ---


def test_calibrated_similarity_identical_text_is_similar() -> None:
    # min_auc=0.0 disables the AUC gate (default 0.7) -- this test is
    # about the encode-and-compare mechanism itself (identical text
    # should score ~1.0 similarity), not about whether MiniLM-L6-v2/en
    # (real AUC 0.6, below the default gate) is a trustworthy cell. See
    # test_calibrated_similarity_low_auc_cell_is_low_discrimination_by_default
    # below for what happens without min_auc=0.0.
    r = calibrated_similarity(
        gold="The capital of Maharashtra is Mumbai.",
        response="The capital of Maharashtra is Mumbai.",
        language="en",
        encoder_name=_ENCODER,
        min_auc=0.0,
    )
    assert r.label == "similar"
    assert r.passed is True
    assert r.score > 0.99


# --- real encoding: correct vs wrong-entity answer ---


def test_calibrated_similarity_scores_higher_for_correct_than_wrong_entity() -> None:
    gold = "The capital of Maharashtra is Mumbai."
    correct = "Mumbai is the capital of Maharashtra."
    wrong = "The capital of Tamil Nadu is Chennai."

    r_correct = calibrated_similarity(
        gold, correct, language="en", encoder_name=_ENCODER, min_auc=0.0
    )
    r_wrong = calibrated_similarity(gold, wrong, language="en", encoder_name=_ENCODER, min_auc=0.0)

    assert r_correct.score > r_wrong.score


# --- detail payload / calibration metadata ---


def test_calibrated_similarity_detail_reports_calibration_metadata() -> None:
    r = calibrated_similarity(
        gold="hello world",
        response="hello world",
        language="en",
        encoder_name=_ENCODER,
        min_auc=0.0,
    )
    assert r.detail["encoder"] == _ENCODER
    assert r.detail["language"] == "en"
    assert r.detail["calibrated"] is True
    assert "raw_cosine_similarity" in r.detail
    assert "calibration_n_cases" in r.detail


def test_calibrated_similarity_detail_has_unclamped_cosine_similarity() -> None:
    # Regression (found by an independent outside review): the
    # detail key "raw_cosine_similarity" was actually the POST-clamp
    # value ([0, 1]), not the true unclamped cosine similarity -- a
    # misleading name. "unclamped_cosine_similarity" now carries the
    # real pre-clamp value; "raw_cosine_similarity" is kept (same
    # post-clamp value as before) for backwards compatibility.
    r = calibrated_similarity(
        gold="hello world", response="hello world", language="en", encoder_name=_ENCODER
    )
    assert "unclamped_cosine_similarity" in r.detail
    unclamped = r.detail["unclamped_cosine_similarity"]
    assert r.detail["raw_cosine_similarity"] == max(0.0, min(1.0, unclamped))


def test_calibrated_similarity_score_is_bounded() -> None:
    r = calibrated_similarity(
        gold="hello world",
        response="completely unrelated text",
        language="en",
        encoder_name=_ENCODER,
        min_auc=0.0,
    )
    assert 0.0 <= r.score <= 1.0


# --- degenerate-cell warnings (found by an independent outside review:
# CalibratedThreshold.warnings existed for calibrate()'s own callers
# but the shipped CALIBRATION_TABLE never surfaced any) ---


def test_calibrated_similarity_surfaces_calibration_warning_for_degenerate_cell() -> None:
    # LaBSE/en's shipped cell has fitted accuracy at chance (see
    # calibration.py's CALIBRATION_TABLE comment) -- calibrate() itself
    # would warn about this data, and calibrated_similarity must now
    # copy that warning into both detail and reason, not just leave it
    # sitting undiscoverable in the README's AUC table. min_auc=0.0
    # disables the (separate, stronger) AUC gate below so this test can
    # still reach the .warnings path specifically -- LaBSE/en's real
    # AUC (0.070) is also below the default min_auc=0.7, so without
    # this override the result would be "low_discrimination" instead.
    r = calibrated_similarity(
        gold="The capital of Maharashtra is Mumbai.",
        response="The capital of Maharashtra is Mumbai.",
        language="en",
        encoder_name="sentence-transformers/LaBSE",
        min_auc=0.0,
    )
    assert "calibration_warnings" in r.detail
    assert len(r.detail["calibration_warnings"]) > 0
    assert "WARNING" in r.reason


def test_calibrated_similarity_no_warning_key_for_clean_cell() -> None:
    # multilingual-e5-base/en is a clean, strong cell -- no warning.
    r = calibrated_similarity(
        gold="The capital of Maharashtra is Mumbai.",
        response="The capital of Maharashtra is Mumbai.",
        language="en",
        encoder_name="intfloat/multilingual-e5-base",
    )
    assert "calibration_warnings" not in r.detail
    assert "WARNING" not in r.reason


# --- min_auc gate (SAFE DEFAULT, added after repeated independent
# outside review kept finding calibrated_similarity return a clean,
# confident similar/dissimilar verdict for cells with no real AUC) ---


def test_calibrated_similarity_low_auc_cell_is_low_discrimination_by_default() -> None:
    # multilingual-e5-base/hi has real AUC exactly 0.500 (chance) --
    # see calibration.py's CALIBRATION_TABLE comment -- but its
    # in-sample argmax-fitted accuracy (0.632) looks fine, and
    # calibrate()'s own .warnings guard cannot catch this (it only
    # checks same-sample fitted accuracy, not AUC). With the default
    # min_auc=0.7, this must now return low_discrimination, not a
    # clean similar/dissimilar verdict, regardless of the raw cosine
    # similarity.
    r = calibrated_similarity(
        gold="भारत की राजधानी नई दिल्ली है।",
        response="भारत की राजधानई दिल्ली है।",
        language="hi",
        encoder_name="intfloat/multilingual-e5-base",
    )
    assert r.label == "low_discrimination"
    assert r.passed is False
    assert r.detail["roc_auc"] < 0.7


def test_calibrated_similarity_low_auc_cell_passes_with_min_auc_zero() -> None:
    r = calibrated_similarity(
        gold="भारत की राजधानी नई दिल्ली है।",
        response="भारत की राजधानी नई दिल्ली है।",
        language="hi",
        encoder_name="intfloat/multilingual-e5-base",
        min_auc=0.0,
    )
    assert r.label != "low_discrimination"


def test_calibrated_similarity_strong_auc_cell_not_low_discrimination() -> None:
    # multilingual-e5-base/en has real AUC 0.860, well above the
    # default min_auc=0.7 -- must not be gated by min_auc, regardless
    # of whether the raw cosine similarity happens to clear e5-base/en's
    # calibrated threshold (0.881) for this specific text pair --
    # that's what the "similar"/"dissimilar" label decides, a separate
    # question from whether this cell has real discrimination at all,
    # which is what this test is actually checking.
    r = calibrated_similarity(
        gold="The capital of Maharashtra is Mumbai.",
        response="The capital of Maharashtra is Mumbai.",
        language="en",
        encoder_name="intfloat/multilingual-e5-base",
    )
    assert r.label != "low_discrimination"
    assert r.label in ("similar", "dissimilar")


# --- NaN cosine similarity (found by an independent outside review) ---


def test_calibrated_similarity_raises_on_nan_cached_embedding() -> None:
    # Regression: `max(0.0, min(1.0, float("nan")))` returns 1.0 in
    # Python (every comparison against NaN is False), so a NaN cosine
    # similarity used to silently become score=1.0, passed=True,
    # label="similar" -- the most confident possible result, from a
    # numerically undefined comparison. Reachable via a corrupted cache
    # entry, which this test exercises for real: write an actual NaN
    # .npy file to the real cache path encode() will read, then confirm
    # calibrated_similarity() raises instead of silently scoring 1.0.
    gold_text = "__vindex_test_nan_regression_gold__"
    response_text = "__vindex_test_nan_regression_response__"
    # calibrated_similarity() reuses a process-lifetime Encoder instance
    # keyed on (encoder_name, allow_muril) -- see similarity.py's
    # _encoder_instances -- so this test's fake cache entry must be
    # written under that same instance's resolved revision, or the
    # revision-pinned cache key (see encoder.py's "CACHE KEY INCLUDES
    # RESOLVED MODEL REVISION") will miss the fake entry instead of
    # reading the poisoned NaN payload.
    revision = Encoder(_ENCODER).load().revision
    for text, is_query in ((gold_text, False), (response_text, True)):
        path, directory = _cache_path(_ENCODER, revision, text, is_query)
        os.makedirs(directory, exist_ok=True)
        np.save(path[:-4], np.array([float("nan")] * 8, dtype="float32"))

    try:
        with pytest.raises(ValueError, match="NaN"):
            calibrated_similarity(
                gold=gold_text, response=response_text, language="en", encoder_name=_ENCODER
            )
    finally:
        for text, is_query in ((gold_text, False), (response_text, True)):
            path, _ = _cache_path(_ENCODER, revision, text, is_query)
            if os.path.exists(path):
                os.remove(path)
