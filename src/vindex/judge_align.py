"""
Align-then-judge: diff first, judge only what disagrees (Milestone 5.4).

Sarvam's shape, applied to reference-based judging: word-level diff
`response` against `gold` first. If they align completely (every
opcode is "equal"), there is nothing to judge -- return a pass with no
LLM call at all. If they diverge, extract only the non-equal spans
(with a small window of surrounding context so the judge isn't
evaluating a bare fragment) instead of sending the full texts.

This bounds cost, latency, and variance to the subset of cases that
actually disagree: a response that is character-for-character (or
near enough) the gold answer never touches the judge model at all, and
a response that differs in one clause doesn't force the judge to
re-read and re-reason about the parts that already match.

Only applies to reference-based judging (Milestone 5.3's non-default
mode) -- reference-free judging has nothing to diff against by
definition, since there is no second text.

Uses difflib.SequenceMatcher (stdlib), same choice as vindex.match
(Milestone 2.3) for the same reason: no new third-party dependency for
a diffing primitive stdlib already provides adequately.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

_CONTEXT_WORDS = 3


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """aligned      : True if response and gold matched completely at
                       the word level -- no LLM call needed.
    mismatched_response : the response's non-matching spans, joined
                       with a small window of surrounding context for
                       readability, or "" if aligned is True.
    mismatched_gold : the gold's non-matching spans, same shape.
    similarity_ratio : SequenceMatcher's own ratio() over the two word
                       sequences, for logging/debugging -- not used to
                       decide aligned (aligned is exact: every opcode
                       must be "equal", not "close enough").
    """

    aligned: bool
    mismatched_response: str
    mismatched_gold: str
    similarity_ratio: float


def align(response: str, gold: str) -> AlignmentResult:
    """Word-level diff of `response` against `gold`.

    Splits on whitespace only (script-agnostic: does not require
    word-boundary logic beyond that, since Devanagari and other Indic
    scripts already use spaces between words the same as Latin text).
    """
    response_words = (response or "").split()
    gold_words = (gold or "").split()

    matcher = SequenceMatcher(None, response_words, gold_words)
    opcodes = matcher.get_opcodes()

    if all(tag == "equal" for tag, *_rest in opcodes):
        return AlignmentResult(
            aligned=True, mismatched_response="", mismatched_gold="", similarity_ratio=1.0
        )

    response_spans: list[str] = []
    gold_spans: list[str] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        r_start = max(0, i1 - _CONTEXT_WORDS)
        r_end = min(len(response_words), i2 + _CONTEXT_WORDS)
        g_start = max(0, j1 - _CONTEXT_WORDS)
        g_end = min(len(gold_words), j2 + _CONTEXT_WORDS)
        response_spans.append(" ".join(response_words[r_start:r_end]))
        gold_spans.append(" ".join(gold_words[g_start:g_end]))

    return AlignmentResult(
        aligned=False,
        mismatched_response=" ... ".join(response_spans),
        mismatched_gold=" ... ".join(gold_spans),
        similarity_ratio=matcher.ratio(),
    )
