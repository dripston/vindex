"""
Tests for vindex.judge_trace_check (Milestone 6). check_trace() (6.1)
is pure, deterministic, offline logic -- tested with injected TrapWord
fixtures (the समुद्र तल and उत्तर cases from this project's own
documented findings), not real dictionary content (which is currently
empty -- see test_trap_words.py). check_trace_llm_fallback() (6.4)
needs a real Groq call; skipped if GROQ_API_KEY isn't set, matching
test_judge.py's convention.
"""

from __future__ import annotations

import os

import pytest

from vindex.judge_trace_check import check_trace
from vindex.trap_words import TrapWord

_SAMUDRA_TAL = TrapWord(term="समुद्र तल", reading_a="sea level", reading_b="sea floor", source="own")
_UTTAR = TrapWord(term="उत्तर", reading_a="north", reading_b="answer", source="hindiwic")

# --- check_trace: empty / validation ---


def test_check_trace_empty_source_scores_empty() -> None:
    r = check_trace("", "some trace", trap_words=[_SAMUDRA_TAL])
    assert r.label == "empty"
    assert r.passed is False


def test_check_trace_empty_trace_scores_empty() -> None:
    r = check_trace("some source", "", trap_words=[_SAMUDRA_TAL])
    assert r.label == "empty"


def test_check_trace_default_dictionary_reports_size() -> None:
    r = check_trace("source", "trace")
    assert "dictionary_size" in r.detail
    assert r.detail["dictionary_size"] >= 0


# --- check_trace: the समुद्र तल regression case (real, documented finding) ---


def test_check_trace_catches_samudra_tal_mistranslation() -> None:
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = (
        "The question asks for the boiling temperature of water at the "
        "sea floor, so the boiling point is significantly above 100 C."
    )
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "misread_detected"
    assert r.passed is False
    assert r.detail["flagged_terms"][0]["term"] == "समुद्र तल"


def test_check_trace_passes_correct_samudra_tal_reading() -> None:
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = "The question asks about water boiling at sea level, which is 100 C."
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "no_misread_detected"
    assert r.passed is True
    assert r.detail["flagged_terms"] == []


def test_check_trace_does_not_flag_when_trace_mentions_both_readings() -> None:
    # A judge correctly reasoning through the ambiguity (mentioning
    # the wrong reading only to reject it) must not be flagged --
    # that's correct reasoning, not a misread.
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = (
        "समुद्र तल could be misread as sea floor, but it actually means "
        "sea level in this context, so 100 C is correct."
    )
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "no_misread_detected"


def test_check_trace_skips_term_not_present_in_source() -> None:
    source = "भारत की राजधानी क्या है?"
    trace = "The question asks for the boiling temperature at the sea floor."
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "no_misread_detected"
    assert r.detail["flagged_terms"] == []


# --- check_trace: the उत्तर (north vs answer) case from BUILD_PLAN.md ---


def test_check_trace_catches_uttar_mistranslation() -> None:
    source = "दिल्ली से उत्तर की ओर कौन सा राज्य है?"
    trace = "The question is asking for an answer, so I need to provide a response."
    r = check_trace(source, trace, trap_words=[_UTTAR])
    assert r.label == "misread_detected"
    assert r.detail["flagged_terms"][0]["reading_b"] == "answer"


# --- check_trace: multiple trap words, only matching ones flagged ---


def test_check_trace_flags_only_matching_terms() -> None:
    source = "समुद्र तल पर पानी उबलता है।"  # only समुद्र तल present, not उत्तर
    trace_bad_samudra = "boiling happens at the sea floor"
    r = check_trace(source, trace_bad_samudra, trap_words=[_SAMUDRA_TAL, _UTTAR])
    assert len(r.detail["flagged_terms"]) == 1
    assert r.detail["flagged_terms"][0]["term"] == "समुद्र तल"


def test_check_trace_is_deterministic() -> None:
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = "the sea floor"
    r1 = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    r2 = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r1.label == r2.label
    assert r1.score == r2.score


# --- check_trace_llm_fallback (Milestone 6.4): needs a real Groq call ---

pytest.importorskip("groq")
if not os.environ.get("GROQ_API_KEY"):
    pytest.skip(
        "GROQ_API_KEY not set; skipping check_trace_llm_fallback tests",
        allow_module_level=True,
    )

from vindex.judge_model import GroqJudge  # noqa: E402
from vindex.judge_trace_check import check_trace_llm_fallback  # noqa: E402


def test_llm_fallback_skips_call_on_exact_alignment() -> None:
    judge = GroqJudge()
    r = check_trace_llm_fallback("identical text here", "identical text here", judge)
    assert r.passed is True
    assert r.detail["llm_fallback_used"] is False
    assert r.detail["aligned"] is True


def test_llm_fallback_dictionary_hit_skips_llm_call() -> None:
    judge = GroqJudge()
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = "the sea floor is where this happens"
    r = check_trace_llm_fallback(source, trace, judge, trap_words=[_SAMUDRA_TAL])
    assert r.label == "misread_detected"
    assert "llm_fallback_used" not in r.detail  # dictionary result returned unchanged


def test_llm_fallback_catches_mistranslation_outside_dictionary() -> None:
    judge = GroqJudge()
    # "kal" (कल) is tense-ambiguous (yesterday/tomorrow) -- not a fixed
    # word-pair a dictionary entry can represent, exactly the category
    # 6.4 exists for.
    source = "कल मैं बाजार जाऊंगा।"
    trace = "The source says the speaker went to the market yesterday."
    r = check_trace_llm_fallback(source, trace, judge)
    assert r.passed is False
    assert r.detail["llm_fallback_used"] is True
