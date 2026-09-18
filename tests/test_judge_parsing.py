"""
Tests for vindex.judge's pure parsing logic (_parse_judge_response,
_family). Unlike test_judge.py, these do NOT call a real judge model --
they test parsing/heuristic logic directly on crafted strings, which is
not "mocking judge behavior" (this project's stated no-mocks preference
is about not faking what a real LLM would say), just unit-testing a
pure function. No GROQ_API_KEY needed.
"""

from __future__ import annotations

import json

import pytest

from vindex.judge import _family, _parse_judge_response

# --- _parse_judge_response: out-of-range score ---


def test_parse_judge_response_valid_score_in_range() -> None:
    raw = json.dumps({"score": 5, "confidence": "high", "reasoning": "correct"})
    score, reasoning, confidence, raw_confidence = _parse_judge_response(raw)
    assert score == 1.0
    assert confidence == "high"
    assert raw_confidence == "high"
    assert reasoning == "correct"


def test_parse_judge_response_score_at_low_end_of_range() -> None:
    raw = json.dumps({"score": 1, "confidence": "low", "reasoning": "wrong"})
    score, _, _, _ = _parse_judge_response(raw)
    assert score == 0.0


def test_parse_judge_response_medium_confidence_normalizes_to_low_but_reports_raw() -> None:
    # Regression: a judge sending "medium" (not "high"/"low") used to
    # have its raw value silently overwritten by the normalized "low"
    # everywhere, including where a caller might report it -- so the
    # audit trail claimed the judge said "low" when it actually said
    # "medium". The normalized value used for gating is still "low"
    # (the safe default), but raw_confidence must report what was
    # actually sent.
    raw = json.dumps({"score": 5, "confidence": "medium", "reasoning": "x"})
    _, _, confidence, raw_confidence = _parse_judge_response(raw)
    assert confidence == "low"
    assert raw_confidence == "medium"


def test_parse_judge_response_boolean_score_raises() -> None:
    # Regression: bool is a subclass of int in Python, so
    # float(True) == 1.0 succeeded silently and {"score": true} was
    # accepted as a valid score of 1.
    raw = '{"score": true, "confidence": "high", "reasoning": "x"}'
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_false_boolean_score_raises() -> None:
    raw = '{"score": false, "confidence": "high", "reasoning": "x"}'
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_prose_before_json_extracts_correctly() -> None:
    # Regression: _extract_json used to greedily match from the FIRST
    # "{" to the LAST "}" in the whole response. A judge response that
    # mentions any brace in prose before its real JSON answer used to
    # have the prose glued into the "JSON" and fail to parse.
    raw = (
        'Reasoning: use {a:1}. Final answer: '
        '{"score": 5, "confidence": "high", "reasoning": "ok"}'
    )
    score, reasoning, confidence, _ = _parse_judge_response(raw)
    assert score == 1.0
    assert confidence == "high"


def test_parse_judge_response_out_of_range_score_raises_not_clamps() -> None:
    # Regression: a score of 100 (e.g. a judge misreading "1-5" as
    # "out of 100") used to be silently clamped to 5 -- the package's
    # most confident possible pass -- instead of being treated as the
    # malformed response it actually is.
    raw = json.dumps({"score": 100, "confidence": "high", "reasoning": "x"})
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_zero_score_raises() -> None:
    raw = json.dumps({"score": 0, "confidence": "high", "reasoning": "x"})
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_negative_score_raises() -> None:
    raw = json.dumps({"score": -3, "confidence": "high", "reasoning": "x"})
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_infinity_score_raises_value_error_not_overflow() -> None:
    # Regression: json.loads accepts the non-standard "Infinity" token,
    # and float("inf") passes float() -- but round()/int() on an
    # infinite float raises OverflowError, which wasn't in the
    # score-range check's path and crashed indic_judge() uncaught
    # instead of degrading to judge_error like every other malformed
    # judge response.
    raw = '{"score": Infinity, "confidence": "high", "reasoning": "x"}'
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


def test_parse_judge_response_negative_infinity_score_raises() -> None:
    raw = '{"score": -Infinity, "confidence": "high", "reasoning": "x"}'
    with pytest.raises(ValueError):
        _parse_judge_response(raw)


# --- _family: same-model self-enhancement heuristic ---


def test_family_strips_trailing_size_token() -> None:
    assert _family("openai/gpt-oss-120b") == _family("openai/gpt-oss-20b")


def test_family_provider_prefix_is_not_stripped_known_limitation() -> None:
    # Documented narrow behavior (see _family's docstring): a provider
    # prefix difference is NOT normalized away, so the same base model
    # hosted by two different providers is treated as a different
    # family -- no self-enhancement warning fires.
    assert _family("openai/gpt-oss-120b") != _family("groq/gpt-oss-120b")


def test_family_finetune_suffix_is_not_stripped_known_limitation() -> None:
    # Documented narrow behavior: a suffix after the size token (e.g.
    # "-instruct") is not stripped either, so a fine-tune of the same
    # base model is treated as a different family.
    assert _family("llama-3.1-70b") != _family("llama-3.1-70b-instruct")


def test_family_gpt4o_mini_pairing_is_not_caught_known_limitation() -> None:
    # KNOWN LIMITATION (found by an independent outside review, see
    # _family's docstring's "MISSES THE MOST COMMON REAL PAIRING"
    # paragraph): "gpt-4o" vs "gpt-4o-mini" is not caught -- "mini" is
    # not a bare size digit, so neither name gets stripped, and this is
    # probably the single most common self-judging pair in production.
    assert _family("gpt-4o") != _family("gpt-4o-mini")


def test_family_instruct_suffix_after_size_token_is_not_caught_known_limitation() -> None:
    assert _family("llama-3.1-70b-instruct") != _family("llama-3.1-8b-instruct")
