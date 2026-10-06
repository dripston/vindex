"""Detect when a judge misread an ambiguous source term in its reasoning.

A judge can misunderstand a source term inside its own reasoning before
any score exists, which output-level checks (script, transliteration,
WER) cannot see. For example, a judge reasoning in English mistranslated
समुद्र तल ("sea level") as "sea floor" and scored a correct answer 0.
This module inspects the judge's reasoning trace for such misreadings.
It works with traces from any judge (indic_judge, DeepEval, Ragas, or
your own), not only :func:`vindex.indic_judge`.

:func:`check_trace` is deterministic and makes no LLM call. It uses the
human-reviewed trap-word dictionary in :mod:`vindex.trap_words`: for each
term that appears as a whole word in the source, the trace is flagged if
it contains the term's wrong reading (``reading_b``) but not its correct
reading (``reading_a``).

Scope and limitations:

- It detects misreadings of the known dictionary terms only (69 terms;
  every result reports ``dictionary_size``). Misreadings of other terms
  are out of scope, not false negatives.
- Glosses are matched literally (whole-word, case-insensitive). Correct
  reasoning phrased differently from ``reading_a`` is flagged (false
  positive), and a synonym for the wrong reading is missed (false
  negative).
- All glosses are English, so a trace written in Hindi (as the vindex
  rubric instructs) can never trigger the check. Treat
  ``"no_misread_detected"`` on a Hindi-language trace with skepticism.
- No precision/recall study of ``check_trace`` has been run. The
  published human-agreement figures (90.3% / 93.5%) are for
  :func:`vindex.indic_judge` verdicts, not for this module.

:func:`check_trace_llm_fallback` is a separate, opt-in mode for
ambiguities the dictionary cannot cover. It runs the dictionary check
first, then aligns source and trace and sends only the mismatched
segments to an LLM, flagging by default when the response is ambiguous.
"""

from __future__ import annotations

import re
import unicodedata

from vindex.judge_align import align
from vindex.judge_model import JudgeModel
from vindex.result import MetricResult, coerce_text
from vindex.trap_words import TrapWord, load_trap_words


def _is_devanagari_word_continuer(c: str) -> bool:
    """Return True if ``c`` continues an adjacent Devanagari word.

    Devanagari letters and combining marks (matras, virama) both
    continue a word; only a non-Devanagari character (space,
    punctuation, Latin, or the string boundary) is a word boundary.
    Python's regex ``\\b`` is unreliable here: it finds no boundary in
    ``"चादर"`` before ``"दर"``, so ``\\bदर\\b`` wrongly matches.
    """
    if not c:
        return False
    if not ("ऀ" <= c <= "ॿ"):
        return False
    category = unicodedata.category(c)
    return category in ("Lo", "Lm", "Mn", "Mc")  # letters + combining marks


def _term_present_as_word(term: str, source: str) -> bool:
    """Return True if ``term`` occurs in ``source`` as a whole word.

    Prevents substring matches such as कल inside कलम, दर inside चादर,
    or मूल inside मूल्य, by checking the characters on either side of
    each occurrence (see :func:`_is_devanagari_word_continuer`).
    """
    idx = source.find(term)
    while idx != -1:
        before = source[idx - 1] if idx > 0 else ""
        after = source[idx + len(term)] if idx + len(term) < len(source) else ""
        if not _is_devanagari_word_continuer(before) and not _is_devanagari_word_continuer(after):
            return True
        idx = source.find(term, idx + 1)
    return False


def _primary_gloss(reading: str) -> str:
    """Strip a trailing parenthetical clarification from a reading.

    For example ``"ocean floor (seabed)"`` becomes ``"ocean floor"``;
    traces typically use only the primary term.
    """
    paren = reading.find("(")
    primary = reading[:paren] if paren != -1 else reading
    return primary.strip()


def _contains_gloss(trace_lower: str, gloss: str) -> bool:
    """Return True if ``gloss`` occurs in ``trace_lower`` as a whole phrase.

    Multi-word glosses (e.g. ``"sea floor"``) are matched as a phrase with
    word boundaries at both ends, so ``"answer"`` does not match inside
    ``"unanswerable"``.
    """
    pattern = r"\b" + re.escape(gloss) + r"\b"
    return re.search(pattern, trace_lower) is not None


def _trace_says_wrong_reading(trace: str, word: TrapWord) -> bool:
    """Return True if the trace uses the wrong reading but not the right one.

    Matches each reading's primary gloss (see :func:`_primary_gloss`)
    as a case-insensitive whole word. A trace that names the wrong
    reading only to rule it out ("not sea floor, but sea level") is not
    flagged, because it also contains ``reading_a``.

    Known limitations of literal matching:

    - Correct reasoning in other words ("the surface, not the depths")
      is flagged (false positive).
    - A synonym for the wrong reading ("ocean bottom") is missed (false
      negative).
    - Many entries have a ``reading_b`` that is a common English word
      (e.g. अंग -> "organ", फल -> "result"). An English trace that uses
      such a word in its ordinary sense is flagged whenever the source
      contains the term. Treat flags on these entries as needing human
      review.
    """
    trace_lower = trace.lower()
    says_b = _contains_gloss(trace_lower, _primary_gloss(word.reading_b).lower())
    says_a = _contains_gloss(trace_lower, _primary_gloss(word.reading_a).lower())
    return says_b and not says_a


def check_trace(
    source: str, trace: str, trap_words: list[TrapWord] | None = None
) -> MetricResult:
    """Check whether a judge's reasoning misread a known ambiguous term.

    Deterministic and LLM-free: the same inputs always produce the same
    result. See the module docstring for scope and limitations.

    Args:
        source: The source text the judge was evaluating. ``None`` is
            treated as empty.
        trace: The judge's reasoning text. ``None`` is treated as empty.
        trap_words: Dictionary to check against. Defaults to
            :func:`vindex.trap_words.load_trap_words`; pass your own list to
            use a project-specific dictionary.

    Returns:
        A :class:`vindex.result.MetricResult` labelled
        ``"misread_detected"``, ``"no_misread_detected"``, or ``"empty"``.
        ``detail`` contains ``dictionary_size`` and, for non-empty input,
        ``flagged_terms`` (each with ``term``, ``reading_a``,
        ``reading_b``).

    Raises:
        TypeError: If ``trap_words`` contains anything other than
            :class:`vindex.trap_words.TrapWord` instances.
    """
    source = coerce_text(source)
    trace = coerce_text(trace)
    words = trap_words if trap_words is not None else load_trap_words()
    for word in words:
        if not isinstance(word, TrapWord):
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
            "terms this tool checks for were misread."
        ),
        detail=detail,
    )


def check_trace_llm_fallback(
    source: str, trace: str, judge: JudgeModel, trap_words: list[TrapWord] | None = None
) -> MetricResult:
    """Check a trace with the dictionary, then an LLM for anything it misses.

    Covers misreadings the dictionary cannot catch, such as terms not yet
    in :mod:`vindex.trap_words` or ambiguities that are not a fixed word
    pair. Steps:

    1. Run :func:`check_trace`. A ``"misread_detected"`` or ``"empty"``
       result is returned unchanged, with no LLM call.
    2. Align ``trace`` against ``source`` (:func:`vindex.judge_align.align`).
       An exact word-level match returns ``"no_misread_detected"`` with
       no LLM call.
    3. Otherwise send only the mismatched segments to ``judge``. The
       result passes only if the judge reports no misread at ``"high"``
       confidence; an ambiguous or unparseable response is flagged.

    Use :func:`check_trace` alone if you want only the free,
    deterministic check. See :mod:`vindex.judge_rubric` for a note on
    prompt injection: the trace is inserted into the prompt verbatim.

    Args:
        source: The source text.
        trace: The judge's reasoning text.
        judge: Judge model used for the fallback call.
        trap_words: Dictionary override, as in :func:`check_trace`.

    Returns:
        A :class:`vindex.result.MetricResult`. ``detail`` adds
        ``llm_fallback_used``, ``aligned``, and, when the LLM was called,
        ``judge_model_id``, ``llm_confidence``, and ``llm_reasoning``.
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

