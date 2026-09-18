"""
Tests for vindex.calibration: the shipped calibration table (Milestone
3.1), MuRIL warning (3.2), and calibrate() for user-fitted thresholds
(3.3). See src/vindex/calibration.py's module docstring for where the
shipped table's numbers come from and their stated small-sample caveat.
"""

import pytest

from vindex.calibration import (
    CALIBRATION_TABLE,
    DEFAULT_ENCODER,
    MURIL_MODEL_NAME,
    MURIL_WARNING,
    CalibratedThreshold,
    calibrate,
    get_threshold,
)

# --- shipped table shape ---


def test_calibration_table_covers_five_encoders() -> None:
    assert len(CALIBRATION_TABLE) == 5


def test_calibration_table_covers_three_languages_per_encoder() -> None:
    for encoder_name, by_language in CALIBRATION_TABLE.items():
        assert set(by_language) == {"en", "hi", "hinglish"}, encoder_name


def test_calibration_table_includes_muril() -> None:
    assert MURIL_MODEL_NAME in CALIBRATION_TABLE


def test_default_encoder_is_in_the_table() -> None:
    assert DEFAULT_ENCODER in CALIBRATION_TABLE


def test_every_cell_states_a_small_case_count() -> None:
    # Milestone 3.1: "state plainly it came from ~10 cases per cell."
    # Regenerated from real per-case data (see calibration.py's comment
    # above CALIBRATION_TABLE): "hi" cells are 9, not 10 -- one task's
    # english_gold/full_sentence Hindi variant has 9 correct-answer
    # rows in discrimination_per_case.csv, not 10. The old hand-typed
    # table used the CalibratedThreshold.n_cases default of 10
    # everywhere, which was never actually checked against the real
    # per-case count until this regeneration.
    for by_language in CALIBRATION_TABLE.values():
        for cell in by_language.values():
            assert cell.n_cases in (9, 10)


def test_hi_cells_have_nine_cases_not_ten() -> None:
    for by_language in CALIBRATION_TABLE.values():
        assert by_language["hi"].n_cases == 9


# --- get_threshold ---


def test_get_threshold_known_cell() -> None:
    t = get_threshold(DEFAULT_ENCODER, "en")
    assert isinstance(t, CalibratedThreshold)
    assert 0.0 <= t.threshold <= 1.0


def test_get_threshold_unknown_encoder_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no shipped calibration for encoder"):
        get_threshold("not-a-real-encoder", "en")


def test_get_threshold_unknown_language_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no shipped calibration for language"):
        get_threshold(DEFAULT_ENCODER, "tamil")


# --- MuRIL warning ---


def test_muril_warning_mentions_hindiwic() -> None:
    assert "HindiWiC" in MURIL_WARNING


def test_muril_warning_states_the_two_numbers() -> None:
    assert "55%" in MURIL_WARNING
    assert "90%" in MURIL_WARNING


def test_muril_cells_score_at_or_near_chance_at_default() -> None:
    for cell in CALIBRATION_TABLE[MURIL_MODEL_NAME].values():
        assert cell.accuracy_at_default <= 0.55


# --- shipped table warnings (found by an independent outside review:
# CalibratedThreshold.warnings existed but the shipped table never
# populated it -- now regenerated from real per-case data, see the
# comment above CALIBRATION_TABLE for exactly what calibrate()'s guard
# does and doesn't catch) ---


def test_labse_en_cell_warns_at_or_below_chance() -> None:
    cell = CALIBRATION_TABLE["sentence-transformers/LaBSE"]["en"]
    assert any("at or below chance" in w for w in cell.warnings)


def test_muril_en_cell_warns_at_or_below_chance() -> None:
    cell = CALIBRATION_TABLE[MURIL_MODEL_NAME]["en"]
    assert any("at or below chance" in w for w in cell.warnings)


def test_e5_base_en_cell_has_no_warnings() -> None:
    # A genuinely strong, clean cell -- no warning.
    cell = CALIBRATION_TABLE["intfloat/multilingual-e5-base"]["en"]
    assert cell.warnings == ()


# --- calibrate() ---


def test_calibrate_perfect_separation() -> None:
    result = calibrate([0.9, 0.85, 0.8], [0.3, 0.2, 0.1])
    assert result.accuracy_at_threshold == 1.0
    assert 0.3 <= result.threshold <= 0.8


def test_calibrate_returns_calibrated_threshold() -> None:
    result = calibrate([0.9, 0.8], [0.2, 0.3])
    assert isinstance(result, CalibratedThreshold)


def test_calibrate_reports_accuracy_at_default_for_comparison() -> None:
    # All correct scores just above 0.5, all wrong scores just above 0.5
    # too -- default 0.5 threshold should misclassify the wrong ones.
    result = calibrate([0.9, 0.8], [0.6, 0.55])
    assert result.accuracy_at_default < 1.0


def test_calibrate_n_cases_is_min_of_both_lists() -> None:
    result = calibrate([0.9, 0.8, 0.7], [0.2])
    assert result.n_cases == 1


def test_calibrate_empty_correct_raises() -> None:
    with pytest.raises(ValueError, match="at least one correct and one wrong"):
        calibrate([], [0.5])


def test_calibrate_empty_wrong_raises() -> None:
    with pytest.raises(ValueError, match="at least one correct and one wrong"):
        calibrate([0.5], [])


# --- calibrate() warnings (added after an outside review found these
# degenerate inputs were accepted silently, with no signal anything
# was wrong) ---


def test_calibrate_single_identical_point_warns_too_few_cases() -> None:
    # Regression: calibrate([0.5], [0.5]) used to return threshold=0.5
    # looking like a real fit, with nothing indicating it was one point
    # each with accuracy_at_threshold == 0.0 (correct must be >=
    # threshold, wrong must be < threshold; a tie goes to "correct").
    result = calibrate([0.5], [0.5])
    assert any("too few cases" in w for w in result.warnings)


def test_calibrate_few_cases_below_five_warns() -> None:
    result = calibrate([0.9, 0.8, 0.7], [0.2, 0.1])
    assert any("too few cases" in w for w in result.warnings)


def test_calibrate_enough_cases_does_not_warn_too_few() -> None:
    result = calibrate([0.9, 0.85, 0.8, 0.75, 0.7], [0.3, 0.25, 0.2, 0.15, 0.1])
    assert not any("too few cases" in w for w in result.warnings)


def test_calibrate_out_of_range_scores_warns() -> None:
    # Regression: calibrate([50.0], [-50.0]) used to report
    # accuracy_at_threshold == 1.0 with no indication these aren't
    # valid cosine similarities at all.
    result = calibrate([50.0, 40.0, 45.0, 42.0, 48.0], [-50.0, -40.0, -45.0, -42.0, -48.0])
    assert any("[-1, 1]" in w for w in result.warnings)


def test_calibrate_in_range_scores_does_not_warn_out_of_range() -> None:
    result = calibrate([0.9, 0.85, 0.8, 0.75, 0.7], [0.3, 0.25, 0.2, 0.15, 0.1])
    assert not any("[-1, 1]" in w for w in result.warnings)


def test_calibrate_inverted_data_warns_at_or_below_chance() -> None:
    # Regression: calibrate([0.1, 0.2], [0.9, 0.8]) -- correct/wrong
    # swapped, or a genuinely inverted encoder -- used to return a
    # threshold with no signal that the fitted accuracy is no better
    # than a coin flip.
    result = calibrate([0.1, 0.2, 0.15, 0.25, 0.12], [0.9, 0.8, 0.85, 0.75, 0.88])
    assert any("at or below chance" in w for w in result.warnings)


def test_calibrate_good_separation_does_not_warn_at_or_below_chance() -> None:
    result = calibrate([0.9, 0.85, 0.8, 0.75, 0.7], [0.3, 0.25, 0.2, 0.15, 0.1])
    assert not any("at or below chance" in w for w in result.warnings)


def test_calibrate_clean_fit_has_no_warnings() -> None:
    result = calibrate([0.9, 0.85, 0.8, 0.75, 0.7], [0.3, 0.25, 0.2, 0.15, 0.1])
    assert result.warnings == ()
