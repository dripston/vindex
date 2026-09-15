"""
Tests for vindex.judge (Milestone 5): indic_judge, the Hindi-rubric,
script-aware, reference-free-by-default LLM judge.

Real Groq API calls, not mocks -- matches this project's stated
preference for real data over mocks (see test_similarity.py). Skipped
if GROQ_API_KEY isn't set. Slower and non-free by design: this is
testing real judge behavior, including the exact समुद्र तल case that
motivated Milestone 5.1 in the first place (experiments/FINDINGS.md) --
a mock could not meaningfully test whether the Hindi rubric actually
avoids that mistranslation, only that some code path was reached.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("groq")

if not os.environ.get("GROQ_API_KEY"):
    pytest.skip(
        "GROQ_API_KEY not set; skipping real-API indic_judge tests", allow_module_level=True
    )

from vindex.judge import indic_judge  # noqa: E402
from vindex.judge_align import align  # noqa: E402
from vindex.judge_model import GroqJudge  # noqa: E402

# --- empty / validation (no LLM call needed) ---


def test_indic_judge_empty_question_scores_empty() -> None:
    r = indic_judge("", "answer")
    assert r.label == "empty"
    assert r.passed is False


def test_indic_judge_empty_answer_scores_empty() -> None:
    r = indic_judge("question", "")
    assert r.label == "empty"


def test_indic_judge_none_question_scores_empty() -> None:
    r = indic_judge(None, "answer")
    assert r.label == "empty"


# --- the समुद्र तल regression case (Milestone 5.1's whole reason to exist) ---


def test_indic_judge_does_not_mistranslate_samudra_tal() -> None:
    # Exact case from experiments/FINDINGS.md: an English-rubric judge
    # mistranslated समुद्र तल ("sea level") as "sea floor" mid-reasoning
    # and scored this CORRECT answer 0.0. The Hindi rubric must not
    # repeat that error.
    question = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    answer = "समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।"

    result = indic_judge(question, answer)

    assert result.passed is True
    assert result.score >= 0.8
    assert "समुद्र" not in result.detail["judge_reasoning"] or "तलहटी" not in result.detail[
        "judge_reasoning"
    ]


# --- reference-free mode (default): correct vs wrong ---


def test_indic_judge_reference_free_correct_answer() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
    )
    assert result.passed is True
    assert result.score >= 0.8
    assert result.detail["mode"] == "reference_free"


def test_indic_judge_reference_free_wrong_answer() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी मुंबई है।",
    )
    assert result.passed is False
    assert result.score < 0.5


# --- script-aware prompting (Milestone 5.2): Romanized Hindi is not an error ---


def test_indic_judge_does_not_penalize_romanized_hindi() -> None:
    result = indic_judge(
        "Bharat ki rajdhani kya hai?",
        "Bharat ki rajdhani New Delhi hai.",
    )
    assert result.passed is True
    assert result.score >= 0.8


# --- align-then-judge (Milestone 5.4): reference-based mode ---


def test_indic_judge_reference_based_exact_match_skips_llm_call() -> None:
    q = "भारत की राजधानी क्या है?"
    gold = "भारत की राजधानी नई दिल्ली है।"

    result = indic_judge(q, gold, gold=gold)

    assert result.passed is True
    assert result.score == 1.0
    assert result.detail["aligned"] is True
    assert "no LLM call made" in result.detail["reasoning"]


def test_indic_judge_reference_based_mismatch_still_flagged() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी मुंबई है।",
        gold="भारत की राजधानी नई दिल्ली है।",
    )
    assert result.passed is False
    assert result.detail["mode"] == "reference_based"


# --- conservative default (Milestone 5.5) ---


def test_indic_judge_result_has_confidence_in_detail() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
    )
    assert result.detail["confidence"] in ("high", "low")


# --- determinism discipline (Milestone 5.6) ---


def test_indic_judge_records_judge_model_id() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
    )
    assert result.detail["judge_model_id"] == GroqJudge.DEFAULT_MODEL_ID


def test_groq_judge_temperature_is_not_configurable() -> None:
    # No temperature parameter on GroqJudge's constructor at all --
    # determinism is not optional.
    import inspect

    sig = inspect.signature(GroqJudge.__init__)
    assert "temperature" not in sig.parameters


# --- self-enhancement bias (Milestone 5.7) ---


def test_indic_judge_warns_on_same_family_answering_model() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
        answering_model_id="openai/gpt-oss-20b",
    )
    assert "self_enhancement_bias_warning" in result.detail
    assert "self-enhancement bias" in result.detail["self_enhancement_bias_warning"]


def test_indic_judge_no_warning_for_different_family() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
        answering_model_id="anthropic/claude-3-haiku",
    )
    assert "self_enhancement_bias_warning" not in result.detail


def test_indic_judge_no_warning_when_answering_model_unknown() -> None:
    result = indic_judge(
        "भारत की राजधानी क्या है?",
        "भारत की राजधानी नई दिल्ली है।",
    )
    assert "self_enhancement_bias_warning" not in result.detail


# --- align() unit tests (pure logic, no LLM call) ---


def test_align_identical_text_is_aligned() -> None:
    result = align("hello world", "hello world")
    assert result.aligned is True
    assert result.similarity_ratio == 1.0


def test_align_different_text_is_not_aligned() -> None:
    result = align("Mumbai is the capital", "The capital is Mumbai")
    assert result.aligned is False
    assert result.mismatched_response != ""
    assert result.mismatched_gold != ""


def test_align_empty_both_is_aligned() -> None:
    result = align("", "")
    assert result.aligned is True
