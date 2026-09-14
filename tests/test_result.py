import dataclasses

import pytest

from vindex import MetricResult


def test_basic_construction() -> None:
    r = MetricResult(
        score=0.8,
        passed=True,
        label="matched",
        reason="response matched the expected script.",
    )
    assert r.score == 0.8
    assert r.passed is True
    assert r.label == "matched"
    assert r.reason == "response matched the expected script."
    assert r.detail == {}


def test_detail_defaults_to_empty_dict_and_is_not_shared() -> None:
    a = MetricResult(score=1.0, passed=True, label="ok", reason="fine.")
    b = MetricResult(score=1.0, passed=True, label="ok", reason="fine.")
    assert a.detail == {} and b.detail == {}
    assert a.detail is not b.detail


def test_detail_carries_arbitrary_evidence() -> None:
    r = MetricResult(
        score=0.0,
        passed=False,
        label="script_mismatch",
        reason="response was in Devanagari; prompt was Romanized Hindi.",
        detail={"expected_script": "roman", "actual_script": "devanagari", "attempts": 3},
    )
    assert r.detail["expected_script"] == "roman"
    assert r.detail["attempts"] == 3


@pytest.mark.parametrize("score", [0.0, 1.0, 0.5])
def test_score_boundaries_are_valid(score: float) -> None:
    MetricResult(score=score, passed=True, label="ok", reason="fine.")


@pytest.mark.parametrize("score", [-0.01, 1.01, -1.0, 2.0])
def test_score_out_of_range_raises(score: float) -> None:
    with pytest.raises(ValueError, match="score must be in"):
        MetricResult(score=score, passed=True, label="ok", reason="fine.")


def test_empty_label_raises() -> None:
    with pytest.raises(ValueError, match="label must be"):
        MetricResult(score=1.0, passed=True, label="", reason="fine.")


def test_empty_reason_raises() -> None:
    with pytest.raises(ValueError, match="reason must be"):
        MetricResult(score=1.0, passed=True, label="ok", reason="")


def test_result_is_frozen() -> None:
    r = MetricResult(score=1.0, passed=True, label="ok", reason="fine.")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.score = 0.0  # type: ignore[misc]


def test_result_equality() -> None:
    a = MetricResult(score=1.0, passed=True, label="ok", reason="fine.", detail={"x": 1})
    b = MetricResult(score=1.0, passed=True, label="ok", reason="fine.", detail={"x": 1})
    assert a == b
