"""
Tests for vindex.judge_trace_check (Milestone 6). check_trace() (6.1)
is pure, deterministic, offline logic -- tested mostly with injected
TrapWord fixtures (समुद्र तल from this project's own documented Phase 0
finding, plus a synthetic उत्तर fixture -- see _UTTAR's own comment
below for why उत्तर is injected here but no longer in the real
dictionary) so the logic tests don't depend on the real dictionary's
exact size or content, plus a handful of tests against the real,
now-filled-in dictionary (see test_trap_words.py) to confirm the
loader and checker actually agree on real data.
check_trace_llm_fallback() (6.4) needs a real Groq call; skipped if
GROQ_API_KEY isn't set, matching test_judge.py's convention.
"""

from __future__ import annotations

import os

import pytest

from vindex.judge_trace_check import check_trace
from vindex.trap_words import TrapWord

_SAMUDRA_TAL = TrapWord(term="समुद्र तल", reading_a="sea level", reading_b="sea floor", source="own")
# Injected fixture ONLY -- उत्तर was REMOVED from the real dictionary
# (see test_trap_words.py's test_load_trap_words_does_not_include_uttar):
# this exact north/answer pairing was found to be backwards for this
# library's own primary use case (question-answering), since उत्तर
# meaning "answer" is correct far more often than "north" in that
# context. Kept here only as a fixture for tests that exercise
# check_trace's general MECHANISM (word-boundary matching, gloss
# suppression) using a real, illustrative ambiguous word -- not as a
# claim that this reading_a/reading_b assignment should ever ship.
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


def test_check_trace_false_positive_on_correct_paraphrase() -> None:
    # KNOWN LIMITATION (see _trace_says_wrong_reading's docstring): the
    # "mentions both readings" escape only works when the trace uses
    # the dictionary's exact reading_a gloss. A trace that correctly
    # rejects the wrong reading but paraphrases the right one ("the
    # surface", not the literal string "sea level") is misclassified
    # as a misread. Pinned here as a documented false positive, not
    # silently assumed away.
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = (
        "This does not refer to the sea floor at all -- it is about "
        "the surface, at standard atmospheric pressure, so 100 C is "
        "correct."
    )
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "misread_detected"  # wrong: this trace is actually correct


def test_check_trace_false_negative_on_synonym_for_wrong_reading() -> None:
    # KNOWN LIMITATION (see _trace_says_wrong_reading's docstring): a
    # trace that genuinely uses the wrong reading, but phrases it with
    # a synonym not in the dictionary ("ocean bottom" instead of "sea
    # floor"), is invisible to the substring check. Pinned here as a
    # documented false negative.
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = (
        "The question asks about the ocean bottom, where pressure is "
        "much higher, so the boiling point should be above 100 C."
    )
    r = check_trace(source, trace, trap_words=[_SAMUDRA_TAL])
    assert r.label == "no_misread_detected"  # wrong: this trace actually misread it


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


def test_check_trace_word_boundary_no_false_match_inside_longer_word() -> None:
    # Regression: substring matching used to flag "answer" inside
    # "unanswerable" -- an outside review's real finding. Word-boundary
    # matching fixes this specific accident (does NOT fix the broader,
    # documented limitation that an isolated, correct, unrelated use of
    # "answer" still flags -- see _trace_says_wrong_reading's docstring).
    source = "दिल्ली से उत्तर की ओर कौन सा राज्य है?"
    trace = "This question is fundamentally unanswerable given the data provided."
    r = check_trace(source, trace, trap_words=[_UTTAR])
    assert r.label == "no_misread_detected"


def test_check_trace_common_word_gloss_false_positive_known_limitation() -> None:
    # KNOWN LIMITATION (see _trace_says_wrong_reading's docstring's
    # "SHARPER VERSION" paragraph, found by an outside review): उत्तर's
    # reading_b, "answer", is an ordinary English word that any judge
    # trace evaluating correctness is likely to use regardless of
    # whether उत्तर (north) was ever misread. Word-boundary matching
    # does not fix this -- pinned here as a documented, not silently
    # assumed-away, false positive.
    source = "दिल्ली से उत्तर की ओर कौन सा राज्य है?"
    trace = "Checking the answer against the source, the reasoning holds up."
    r = check_trace(source, trace, trap_words=[_UTTAR])
    assert r.label == "misread_detected"  # wrong: "answer" here has nothing to do with उत्तर


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


def test_check_trace_matches_reading_with_parenthetical_gloss_stripped() -> None:
    # Regression: dictionary readings can carry a clarifying gloss in
    # parens (e.g. "ocean floor (seabed)") for a human filling in the
    # CSV, but a real judge trace usually only uses the primary term.
    # A trace saying "ocean floor" must still match a reading_b of
    # "ocean floor (seabed)" -- this broke on the first real filled-in
    # dictionary data until _primary_gloss() was added.
    word = TrapWord(
        term="समुद्र तल", reading_a="sea level", reading_b="ocean floor (seabed)", source="own"
    )
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = "The question asks about the ocean floor, so pressure matters."
    r = check_trace(source, trace, trap_words=[word])
    assert r.label == "misread_detected"


# --- check_trace: against the real, filled-in dictionary (not injected) ---


def test_check_trace_real_dictionary_catches_samudra_tal() -> None:
    source = "समुद्र तल पर पानी किस तापमान पर उबलता है?"
    trace = "The judge mistranslated this as sea floor, so the boiling point is above 100 C."
    r = check_trace(source, trace)  # uses the real, loaded dictionary
    assert r.label == "misread_detected"
    # 69, not 70: उत्तर (north/answer) was removed from the real
    # dictionary -- see test_trap_words.py's
    # test_load_trap_words_does_not_include_uttar for why.
    assert r.detail["dictionary_size"] == 69


def test_check_trace_real_dictionary_no_longer_flags_uttar_as_answer() -> None:
    # Regression: उत्तर (north/answer) was REMOVED from the real
    # dictionary (found by an independent outside review) -- a Hindi
    # question containing उत्तर, answered correctly in English using
    # the word "answer", must no longer be flagged. This is the
    # opposite assertion of what this test checked before the entry
    # was removed.
    source = "दिल्ली से उत्तर की ओर कौन सा राज्य है?"
    trace = "The question is asking for an answer, so I need to provide a response."
    r = check_trace(source, trace)
    assert r.label == "no_misread_detected"


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
    # Negation-scope drop -- structurally not a fixed word-pair a
    # dictionary entry can represent (the error isn't "word X misread
    # as word Y", it's the negation disappearing entirely), exactly
    # the category 6.4 exists for. कल (yesterday/tomorrow) was used
    # for this test originally, but is itself now in the real
    # dictionary (own_additions.csv), so it's a dictionary hit, not an
    # LLM-fallback case -- see test_check_trace_real_dictionary_* above.
    source = "मुझे नहीं लगता कि यह सही जवाब है।"  # "I don't think this is the right answer"
    trace = "The source confirms this is the right answer."  # drops the negation
    r = check_trace_llm_fallback(source, trace, judge)
    assert r.passed is False
    assert r.detail["llm_fallback_used"] is True
