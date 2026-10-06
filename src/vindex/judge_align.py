"""Align-then-judge: diff an answer against gold, judge only what differs.

Used by :func:`vindex.indic_judge` in reference-based mode. The response
is diffed against the gold answer word by word (``difflib``). If every
word matches, no LLM call is needed. Otherwise only the differing spans,
each with a few words of surrounding context, are sent to the judge.
This bounds cost, latency, and variance to the parts that actually
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

_CONTEXT_WORDS = 3


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Result of aligning a response against a gold answer.

    Attributes:
        aligned: True if response and gold match exactly at the word level.
        mismatched_response: The response's differing spans with surrounding
            context, joined by ``" ... "``; empty if ``aligned``.
        mismatched_gold: The gold answer's differing spans, same shape.
        similarity_ratio: ``SequenceMatcher.ratio()`` over the two word
            sequences, for logging. Not used to decide ``aligned``.
    """

    aligned: bool
    mismatched_response: str
    mismatched_gold: str
    similarity_ratio: float


def align(response: str, gold: str) -> AlignmentResult:
    """Diff a response against a gold answer at the word level.

    Words are split on whitespace only, which works for Latin and Indic
    scripts alike; whitespace differences are therefore ignored. The
    comparison is case-sensitive: ``"Same Answer"`` and ``"same answer"``
    do not align and are sent to the judge, since whether case matters
    depends on the domain.

    Args:
        response: The candidate answer. ``None`` is treated as empty.
        gold: The reference answer. ``None`` is treated as empty.

    Returns:
        An :class:`AlignmentResult`.
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
