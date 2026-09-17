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
    score, reasoning, confidence = _parse_judge_response(raw)
    assert score == 1.0
    assert confidence == "high"
    assert reasoning == "correct"


def test_parse_judge_response_score_at_low_end_of_range() -> None:
    raw = json.dumps({"score": 1, "confidence": "low", "reasoning": "wrong"})
    score, _, _ = _parse_judge_response(raw)
    assert score == 0.0


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
