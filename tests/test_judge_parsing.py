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

from vindex.judge import _family, _parse_judge_response, indic_judge
from vindex.judge_align import align

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


def test_parse_judge_response_raw_score_4_normalizes_below_the_pass_gate() -> None:
    # Documents the exact behavior indic_judge's docstring previously
    # misstated (found by an independent outside review): a raw score
    # of 4 normalizes to (4-1)/(5-1) = 0.75, which is below the 0.8
    # pass-gate threshold -- only a raw 5 (normalized 1.0) passes at
    # high confidence. This is not a bug; the code was always correct.
    # Only the docstring's "a raw 4 or 5 out of 5" claim was wrong.
    raw = json.dumps({"score": 4, "confidence": "high", "reasoning": "good"})
    score, _, confidence, _ = _parse_judge_response(raw)
    assert score == 0.75
    assert confidence == "high"
    passes_gate = confidence == "high" and score >= 0.8
    assert passes_gate is False


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


# --- indic_judge: exact-match gold path must not require a judge/API
# key at all (found by an independent outside review) ---


def test_indic_judge_exact_match_gold_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: indic_judge's own docstring promises "an exact match
    # with gold skips the LLM call entirely" -- but GroqJudge() (which
    # validates GROQ_API_KEY and raises ValueError if it's missing) used
    # to be constructed BEFORE the exact-match alignment check, so a
    # caller in pure reference-based mode whose answer exactly matches
    # gold still got a hard ValueError for an API key it was never
    # going to need. monkeypatch.delenv guarantees no key is present
    # regardless of this test run's actual environment.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = indic_judge("भारत की राजधानी क्या है?", "नई दिल्ली", gold="नई दिल्ली")
    assert result.label == "matched"
    assert result.passed is True
    assert result.detail["aligned"] is True


def test_indic_judge_mismatched_gold_still_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real judge call is genuinely needed here (answer doesn't align
    # with gold), so this must still raise without a key -- confirms
    # the fix above didn't accidentally make the key optional when a
    # judge call is actually required.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        indic_judge("भारत की राजधानी क्या है?", "मुंबई", gold="नई दिल्ली")


def test_indic_judge_reference_free_mode_still_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        indic_judge("भारत की राजधानी क्या है?", "नई दिल्ली")


# --- align(): case sensitivity (found and documented, not changed, by
# an independent outside review) ---


def test_align_is_case_sensitive_known_behavior() -> None:
    # Documented, deliberate: "Same Answer" vs "same answer" is NOT
    # aligned=True, so it does not qualify for the free exact-match
    # short-circuit and instead falls through to a real judge call.
    # See align()'s docstring for why case is left unnormalized while
    # whitespace is.
    result = align("Same Answer", "same answer")
    assert result.aligned is False


def test_align_whitespace_differences_are_normalized() -> None:
    # Unlike case, whitespace differences ARE normalized (via .split())
    # -- this is the contrast the case-sensitivity docstring note draws.
    result = align("same  answer", "same answer")
    assert result.aligned is True


def test_indic_judge_case_mismatch_gold_falls_through_to_judge_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Consequence of align()'s case sensitivity: a gold/answer pair
    # that differs only in case does not qualify for the free
    # exact-match short-circuit, so it requires a real judge call (and
    # therefore an API key) even though a case-insensitive comparison
    # would have called them equal.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        indic_judge("भारत की राजधानी क्या है?", "New Delhi", gold="new delhi")
