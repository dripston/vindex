"""
Three match modes for script_normalized_match (Milestone 2.3).

Each function normalizes both inputs via vindex.normalize.normalize()
before comparing, so callers pass raw text (in any script/spelling) and
get a comparison that already accounts for script, case, diacritics,
punctuation, and numeral differences -- see normalize.py for exactly
what that does and does not fix (code-mixed transliteration is its one
documented known limitation).

  exact_match_score      : 1.0 if normalized strings are identical,
                            else 0.0. Strictest mode -- a single
                            mismatched token fails it.
  token_f1_score          : whitespace-token-level F1 (precision and
                            recall over token multisets, harmonic mean).
                            Standard QA-evaluation style (SQuAD-style
                            token F1). Tolerant of reordering, partial
                            overlap, extra/missing words.
  char_similarity_score    : difflib.SequenceMatcher ratio over the
                            normalized strings. Character-level, so
                            tolerant of small spelling differences (e.g.
                            the vowel-length noise transliterate.py's
                            docstring documents) that would fail
                            exact_match_score and can under- or
                            over-count in token_f1_score depending on
                            which token they land in.

None of these three functions returns a MetricResult -- they are
comparison primitives. script_normalized_match (not yet built) is the
public MetricResult-returning metric that will pick among them.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from vindex.normalize import normalize


def exact_match_score(a: str | None, b: str | None) -> float:
    """1.0 if normalize(a) == normalize(b), else 0.0.

    Two empty/None inputs are treated as a match (both normalize to
    ""); one empty and one non-empty is not a match.
    """
    na, nb = normalize(a), normalize(b)
    if na == "" and nb == "":
        return 1.0
    return 1.0 if na == nb else 0.0


def _tokens(text: str) -> list[str]:
    return text.split(" ") if text else []


def token_f1_score(a: str | None, b: str | None) -> float:
    """Whitespace-token F1 between normalize(a) and normalize(b).

    Precision and recall are computed over token multisets (a repeated
    token counts once per occurrence, in both the numerator via min()
    and each side's own total). Two empty inputs score 1.0; one empty
    and one non-empty scores 0.0.
    """
    tokens_a = _tokens(normalize(a))
    tokens_b = _tokens(normalize(b))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    counts_a: dict[str, int] = {}
    for tok in tokens_a:
        counts_a[tok] = counts_a.get(tok, 0) + 1
    counts_b: dict[str, int] = {}
    for tok in tokens_b:
        counts_b[tok] = counts_b.get(tok, 0) + 1

    overlap = sum(min(n, counts_b.get(tok, 0)) for tok, n in counts_a.items())
    if overlap == 0:
        return 0.0

    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def char_similarity_score(a: str | None, b: str | None) -> float:
    """difflib.SequenceMatcher ratio between normalize(a) and normalize(b).

    Ratcliff/Obershelp similarity, not true Levenshtein edit distance --
    picked to avoid a second third-party dependency (see match.py's
    module docstring). Two empty inputs score 1.0; one empty and one
    non-empty scores 0.0 (SequenceMatcher's own ratio() already returns
    1.0 for two empty strings, so this is handled without a special
    case, but is stated here for clarity).
    """
    na, nb = normalize(a), normalize(b)
    return SequenceMatcher(None, na, nb).ratio()
