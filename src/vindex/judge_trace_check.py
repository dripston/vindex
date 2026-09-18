"""
judge_trace_check: did the judge from indic_judge (Milestone 5) actually
understand what it read, or did it misread a source term inside its own
reasoning before producing any output? (Milestone 6)

THE PRODUCT STORY (why this pairs with indic_judge)

indic_judge gives you a judge built for Indic. judge_trace_check gives
you a way to verify that judge did not misread the source -- the error
this checks for happens INSIDE the judge's reasoning, before any score
or label exists, so nothing downstream can catch it: not
script_adherence (checks the OUTPUT's script, not the reasoning that
produced it), not a transliteration layer (operates on the answer
text, not the judge's internal reasoning), not WER (measures
transcription fidelity against a reference, a different problem
entirely). The समुद्र तल mistranslation this project found in Phase 0
(experiments/FINDINGS.md, and the reason vindex.judge_rubric exists in
Hindi at all) is exactly this class of error: the judge's own
reasoning silently substituted "sea floor" for "sea level" while
thinking, then scored a correct answer 0.0. Milestone 5's Hindi rubric
reduces how often this happens; this module catches it when it still
does.

Works on the reasoning trace from ANY judge, not only vindex.indic_judge
-- pass any text. Most teams evaluating LLM output in Indian languages
already have some judge in their pipeline (DeepEval, Ragas, a
hand-rolled one); this does not require replacing it.

MECHANISM (Milestone 6.1): dictionary first, not LLM. A fixed list of
known ambiguous Hindi terms (vindex.trap_words), each with a correct
reading and a plausible-mistranslation reading, both filled in by a
human (see trap_words.py's module docstring -- this is deliberately
not automated). The check is purely mechanical: does the source text
contain a trap word's term, AND does the reasoning trace contain that
trap word's wrong reading (in English) without also containing the
right reading? If so, flag it. No meta-judge, no second LLM call, no
kappa study -- deterministic, free, instant, reproducible.

HONEST LIMIT ON "without also containing the right reading": this is a
literal substring match on the dictionary's exact reading_a gloss, not
a paraphrase or negation check. A trace that correctly reasons through
the ambiguity IN DIFFERENT WORDS than the dictionary's exact gloss is
flagged as a misread anyway -- a false positive on genuinely correct
reasoning. A trace that uses a synonym for the wrong reading not in the
dictionary is invisible to the check -- a false negative. See
_trace_says_wrong_reading's docstring for both, pinned by tests rather
than silently assumed away.

THE HONEST CLAIM (Milestone 6.2): this module detects mistranslation
of the N known ambiguous terms in vindex.trap_words.load_trap_words()
-- not "detects mistranslation" in general. N is exactly the number of
human-reviewed trap-word entries that currently exist (see
trap_words.py) -- 69 as of this writing (was 70; उत्तर was removed
after an independent outside review found its reading_a/reading_b
assignment was backwards for this library's own QA-evaluation use
case, see trap_words.py's module docstring). Report N honestly
alongside any result -- see check_trace()'s detail payload, which
always includes dictionary_size. A judge_trace_check result that finds
nothing wrong on a trace containing a mistranslation OUTSIDE the
dictionary is not a false negative of this tool; it is outside this
tool's stated scope. Grow the dictionary from user reports, not from
trying to anticipate every possible ambiguous term up front.

REAL N IS SMALLER STILL FOR HINDI-LANGUAGE TRACES (found by an
independent outside review): every reading_a/reading_b gloss in the
dictionary is ASCII English (verified: zero entries have any
non-ASCII gloss). judge_rubric.py's own prompt instructs the judge, in
bold Hindi, to reason IN Hindi and not translate to English while
thinking -- so a trace produced by following that instruction can
never trigger this check at all, regardless of whether it actually
misread anything: _contains_gloss has nothing to match against in a
Hindi-language trace. `dictionary_size` in every result still reports
69 in this situation, which is honest about the dictionary's size but
not about this check's effective N of 0 on that trace's language.
check_trace() does not currently detect or flag this "trace language
made this check structurally unable to fire" case -- treat a
`no_misread_detected` result on a Hindi-language trace with real
skepticism, not as confirmation nothing was misread.

MILESTONE 6.3/6.5 STATUS -- READ BEFORE CITING ANY NUMBER FOR THIS
MODULE SPECIFICALLY: the 90.3%/93.5% agreement numbers reported in
README.md's Limitations section and experiments/README.md are
indic_judge's verdict agreement with human graders (Milestone 5's
study) -- NOT a precision/recall study of check_trace(). No such study
of check_trace() has been run; see the MILESTONE 6.3 paragraph below,
which remains true and unimplemented. If a passage anywhere appears to
attribute the 90.3% figure to check_trace, that is a documentation
error, not a second, contradicting result -- the 5.x study numbers are
the only real numbers that exist.

OPTIONAL LLM FALLBACK (Milestone 6.4): for mistranslation categories
the dictionary structurally cannot catch (a term not yet in the
dictionary, or an ambiguity that isn't a fixed word-pair at all), an
LLM mode is available via check_trace_llm_fallback() -- Sarvam's
align-then-judge shape: diff the source against the trace first
(vindex.judge_align), send only the mismatched segments to a model,
default to flagging when ambiguous rather than silently passing. This
is a clearly separate, opt-in mode -- the dictionary check
(check_trace()) never calls an LLM, and the two modes are never
silently mixed.

MILESTONE 6.3 (validate with ~60 traces) is NOT implemented here. It
needs a real annotation study with a strict bias protocol (labels
committed to git before the metric is written, a blinded sheet, no
cross-visibility between the two annotators, a frozen rubric, a 30%
held-out set never looked at while tuning, and disclosure of the
self-annotation) -- this is human labor this module cannot perform or
substitute for. See this module's docstring section at the bottom for
what MUST happen before any precision/recall number (6.5) is published.
"""

from __future__ import annotations

import re
import unicodedata

from vindex.judge_align import align
from vindex.judge_model import JudgeModel
from vindex.result import MetricResult, coerce_text
from vindex.trap_words import TrapWord, load_trap_words


def _is_devanagari_word_continuer(c: str) -> bool:
    """True if `c` is a Devanagari letter or combining mark -- i.e. it
    continues the same word as an adjacent Devanagari character, with
    no real word boundary between them. Used by _term_present_as_word
    to find real word boundaries in Devanagari text, where Python's
    regex `\\b` (built on `\\w`) does NOT work correctly:
      - `\\w` treats combining marks (matras, virama) as word
        characters with no boundary before them, so
        `re.search(r"\\bदर\\b", "चादर")` wrongly matches -- there is no
        `\\w`-boundary between "चा" and "दर", even though चादर is one
        word and दर is a different, unrelated word.
      - Consecutive Devanagari CONSONANT letters with no vowel sign
        between them are also still one word (Devanagari has no
        letter-to-letter space the way Latin does) -- कलम is one word,
        क+ल+म, not "क" + a boundary + "लम". A check that only excluded
        combining marks (an earlier draft of this fix) still wrongly
        matched "कल" inside "कलम", because "म" is an ordinary letter,
        not a combining mark, and was wrongly treated as a boundary.
    So this must exclude BOTH combining marks AND ordinary Devanagari
    letters immediately adjacent to a candidate match -- only a
    non-Devanagari character (space, punctuation, Latin, or the string
    boundary) counts as a real word boundary."""
    if not c:
        return False
    if not ("ऀ" <= c <= "ॿ"):
        return False
    category = unicodedata.category(c)
    return category in ("Lo", "Lm", "Mn", "Mc")  # letters + combining marks


def _term_present_as_word(term: str, source: str) -> bool:
    """True if `term` appears in `source` as a real word, not as a
    substring inside a longer word.

    FIXED (a real bug, found by an independent outside review): this
    used to be a plain `term in source` substring test with no
    boundary check at all -- so कल (term, "tomorrow"/"yesterday")
    matched inside कलम ("pen"), दर ("rate"/"door") matched inside
    चादर ("bedsheet"), and मूल ("root"/"principal") matched inside
    मूल्य ("price"). The gloss side of this check (see _contains_gloss)
    got a word-boundary fix already; the source side never did, and a
    plain `\\b` regex boundary does not work correctly for Devanagari
    (see _is_devanagari_word_continuer's docstring) -- so this checks
    the character immediately before and after each match manually
    instead."""
    idx = source.find(term)
    while idx != -1:
        before = source[idx - 1] if idx > 0 else ""
        after = source[idx + len(term)] if idx + len(term) < len(source) else ""
        if not _is_devanagari_word_continuer(before) and not _is_devanagari_word_continuer(after):
            return True
        idx = source.find(term, idx + 1)
    return False


def _primary_gloss(reading: str) -> str:
    """Strip a trailing parenthetical clarification, e.g. "ocean floor
    (seabed)" -> "ocean floor". Dictionary readings often carry a
    clarifying gloss in parens for a human filling in the CSV, but a
    real judge trace will typically only use the primary term -- match
    on that, not the full annotated string."""
    paren = reading.find("(")
    primary = reading[:paren] if paren != -1 else reading
    return primary.strip()


def _contains_gloss(trace_lower: str, gloss: str) -> bool:
    """Whole-word (not substring) match of `gloss` inside `trace_lower`.
    `gloss` may itself be multiple words (e.g. "sea floor") -- matched
    as a phrase with word boundaries at both ends, so "seafloor" or
    "sea floors" don't match but the exact phrase does regardless of
    surrounding punctuation/case.

    FIXED (was a real bug): this used to be a plain substring match, so
    a one-word gloss like "answer" matched inside "unanswerable" or
    "answers" -- word-boundary regex closes that specific hole."""
    pattern = r"\b" + re.escape(gloss) + r"\b"
    return re.search(pattern, trace_lower) is not None


def _trace_says_wrong_reading(trace: str, word: TrapWord) -> bool:
    """True if `trace` mentions the wrong reading (reading_b) without
    also mentioning the correct one (reading_a) -- case-insensitive,
    whole-word match on each reading's primary gloss (the part before
    any parenthetical clarification -- see _primary_gloss).

    KNOWN LIMITATION, narrower than this may first appear: "mentions
    reading_a"/"mentions reading_b" is a whole-word match on the
    dictionary's exact gloss, not a paraphrase check. A trace that
    mentions reading_b only to correctly rule it out (e.g. "this does
    NOT mean sea floor, it means sea level") is correctly NOT flagged,
    because it also contains the literal string "sea level"
    (reading_a). But a trace that does the same correct reasoning in
    different words (e.g. "... it's about the surface, not the
    depths") is flagged as a misread anyway, because reading_a's exact
    gloss never appears -- this is a false positive on a genuinely
    correct trace, not a scope limitation like the "N known terms"
    one. Symmetrically, a trace that DOES use the wrong reading but
    phrases it with a synonym never in the dictionary (e.g. "ocean
    bottom" instead of "sea floor") is invisible to this check -- a
    false negative. Both directions are inherent to word-level matching
    without any NLP; see test_check_trace_false_positive_on_correct_paraphrase
    and test_check_trace_false_negative_on_synonym_for_wrong_reading for
    both pinned as known, not silently assumed away.

    A SHARPER VERSION OF THE SAME LIMITATION (found by an independent
    outside review): 25 of the 70 shipped dictionary entries have a
    reading_b that is itself an ordinary, high-frequency English word
    with no connection to the source term when used in its own right
    -- e.g. उत्तर's reading_b is "answer", अंग's is "organ", फल's is
    "result". Because judge reasoning traces are themselves written in
    English about whether an answer is correct, a trace that legitimately
    uses the word "answer" (about the answer being evaluated, nothing to
    do with उत्तर meaning "north") is indistinguishable, at the word
    level, from a trace that misread उत्तर as "answer". There is no
    transliteration or context field in TrapWord to anchor the match
    more precisely against. The word-boundary fix above stops a
    same-word substring accident (e.g. "answer" inside "unanswerable")
    but does NOT stop this: an isolated, correct, unrelated use of
    "answer" in an English trace still flags if the source happens to
    contain उत्तर. On realistic Hindi-source/English-trace input this
    is not a rare edge case -- treat any check_trace flag on one of
    these 25 terms as needing a human look, not as confirmed evidence
    of a misread, until the dictionary gains a way to anchor reading_b
    to the source term itself rather than to bare English vocabulary."""
    trace_lower = trace.lower()
    says_b = _contains_gloss(trace_lower, _primary_gloss(word.reading_b).lower())
    says_a = _contains_gloss(trace_lower, _primary_gloss(word.reading_a).lower())
    return says_b and not says_a


def check_trace(
    source: str, trace: str, trap_words: list[TrapWord] | None = None
) -> MetricResult:
    """Mechanically check whether `trace` (a judge's reasoning text)
    misread any known ambiguous term present in `source`.

    trap_words defaults to vindex.trap_words.load_trap_words() -- pass
    your own list (e.g. for testing, or a project-specific dictionary
    grown from user reports per Milestone 6.2) to override.

    No LLM call. Deterministic: the same (source, trace, trap_words)
    always produces the same result.
    """
    source = coerce_text(source)
    trace = coerce_text(trace)
    words = trap_words if trap_words is not None else load_trap_words()
    for word in words:
        if not isinstance(word, TrapWord):
            # FIXED (a real bug, found by an independent outside
            # review): passing a list of plain strings/ints instead of
            # TrapWord objects (a caller typo, or an override built
            # from a differently-shaped source) used to fail deep
            # inside the matching loop with a bare
            # `AttributeError: 'str' object has no attribute 'term'`,
            # the one public entry point in this package with no
            # up-front validation of its own override parameter.
            raise TypeError(
                f"trap_words must contain only TrapWord instances, got {word!r} "
                f"({type(word).__name__})"
            )

    if source.strip() == "" or trace.strip() == "":
        return MetricResult(
            score=0.0,
            passed=False,
            label="empty",
            reason="source or trace is empty.",
            detail={"dictionary_size": len(words)},
        )

    flagged_terms = []
    for word in words:
        if not _term_present_as_word(word.term, source):
            continue
        if _trace_says_wrong_reading(trace, word):
            flagged_terms.append(
                {"term": word.term, "reading_a": word.reading_a, "reading_b": word.reading_b}
            )

    detail = {"dictionary_size": len(words), "flagged_terms": flagged_terms}

    if flagged_terms:
        terms_str = ", ".join(f["term"] for f in flagged_terms)
        return MetricResult(
            score=0.0,
            passed=False,
            label="misread_detected",
            reason=(
                f"trace appears to misread {len(flagged_terms)} known "
                f"ambiguous term(s) present in source: {terms_str}."
            ),
            detail=detail,
        )

    return MetricResult(
        score=1.0,
        passed=True,
        label="no_misread_detected",
        reason=(
            f"no known ambiguous term (of {len(words)} in the dictionary) "
            "was misread in the trace. This does not mean the trace is "
            "free of mistranslation -- only that none of the N known "
            "terms this tool checks for were misread (see module "
            "docstring's honest-claim section)."
        ),
        detail=detail,
    )


def check_trace_llm_fallback(
    source: str, trace: str, judge: JudgeModel, trap_words: list[TrapWord] | None = None
) -> MetricResult:
    """Optional LLM-backed fallback (Milestone 6.4) for mistranslation
    the dictionary structurally cannot catch: a term not yet in
    vindex.trap_words, or an ambiguity that isn't a fixed word-pair.

    Runs check_trace() first (free, instant). If it already found a
    dictionary-based misread, returns that result unchanged -- no LLM
    call needed. Otherwise, runs align-then-judge (Sarvam's shape,
    vindex.judge_align) between `source` and `trace`: if they align
    exactly, there is nothing further to check and no LLM call is
    made; if they diverge, only the mismatched segments are sent to
    `judge`, asking whether the trace's reading of that segment
    plausibly diverges from the source's meaning. Defaults to flagging
    (passed=False) when the judge's own response is ambiguous or
    unparseable, per this project's conservative-default convention
    (see indic_judge's Milestone 5.5).

    This is a clearly separate, opt-in mode: call check_trace() if you
    only want the free, deterministic dictionary check; call this only
    when you have accepted the added cost/latency/non-determinism of a
    second LLM call for the categories the dictionary cannot cover.
    """
    dict_result = check_trace(source, trace, trap_words)
    if dict_result.label == "misread_detected" or dict_result.label == "empty":
        return dict_result

    alignment = align(trace, source)
    if alignment.aligned:
        detail = dict(dict_result.detail)
        detail["llm_fallback_used"] = False
        detail["aligned"] = True
        return MetricResult(
            score=1.0,
            passed=True,
            label="no_misread_detected",
            reason=(
                "no dictionary misread found, and trace aligns exactly "
                "with source at the word level -- no LLM call made."
            ),
            detail=detail,
        )

    prompt = (
        "आप एक विशेषज्ञ हैं जो यह जांच रहे हैं कि क्या नीचे दिया गया "
        "तर्क (reasoning) स्रोत पाठ के किसी हिस्से को गलत समझ या गलत "
        "अनुवाद कर रहा है।\n\n"
        f"स्रोत का अंश: {alignment.mismatched_gold}\n\n"
        f"तर्क का अंश: {alignment.mismatched_response}\n\n"
        "क्या तर्क का यह अंश स्रोत के अर्थ को गलत समझता या गलत अनुवाद "
        "करता है? यदि आपको निश्चित नहीं है, तो इसे संभावित समस्या "
        '(flagged) मानें, चुप्पी से सही न मान लें। बिल्कुल इस JSON '
        'प्रारूप में उत्तर दें: {"misread": <true या false>, '
        '"reasoning": "<संक्षिप्त स्पष्टीकरण>", "confidence": <"high" '
        'या "low">}\nकेवल JSON ऑब्जेक्ट आउटपुट करें।'
    )
    raw = judge.call(prompt)

    import json

    from vindex.judge import _extract_json

    try:
        data = json.loads(_extract_json(raw))
        misread = bool(data.get("misread", True))
        confidence = str(data.get("confidence", "low")).lower()
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        misread, confidence, reasoning = True, "low", "judge response could not be parsed"

    passed = not misread and confidence == "high"
    detail = dict(dict_result.detail)
    detail.update(
        {
            "llm_fallback_used": True,
            "aligned": False,
            "judge_model_id": judge.model_id,
            "llm_confidence": confidence,
            "llm_reasoning": reasoning,
        }
    )

    return MetricResult(
        score=0.0 if not passed else 1.0,
        passed=passed,
        label="misread_detected" if not passed else "no_misread_detected",
        reason=(
            f"LLM fallback on mismatched segment: misread={misread} at "
            f"{confidence} confidence."
        ),
        detail=detail,
    )


# MILESTONE 6.3 (not implemented): validate with ~60 traces, human
# agreement study, bias protocol (commit labels before writing the
# metric, blind sheet, frozen rubric, 30% holdout, disclose
# self-annotation). This requires human labor -- see this module's
# docstring. Do not run 6.5 (publish precision/recall) until 6.3 has
# actually happened; do not fabricate a number here.
