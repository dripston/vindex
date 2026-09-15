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


def test_every_cell_states_n_cases_as_ten() -> None:
    # Milestone 3.1: "state plainly it came from 10 cases per cell."
    for by_language in CALIBRATION_TABLE.values():
        for cell in by_language.values():
            assert cell.n_cases == 10


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
