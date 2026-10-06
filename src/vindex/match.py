"""String-comparison scores that are robust to script and spelling.

Each function normalizes both inputs with :func:`vindex.normalize.normalize`
before comparing, so raw text in any supported script can be passed
directly; script, case, diacritic, punctuation, and numeral differences
are already reconciled. Code-mixed transliteration remains a known
limitation of normalization (see :mod:`vindex.normalize`).

- :func:`exact_match_score`: 1.0 if the normalized strings are identical,
  else 0.0.
- :func:`token_f1_score`: SQuAD-style whitespace-token F1; tolerant of
  reordering and partial overlap.
- :func:`char_similarity_score`: character-level similarity; tolerant of
  small spelling differences such as transliteration vowel-length noise.

These return plain floats in [0, 1], not :class:`vindex.result.MetricResult`.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from vindex.normalize import normalize


def exact_match_score(a: str | None, b: str | None) -> float:
    """Return 1.0 if both inputs normalize to the same string, else 0.0.

    Two empty or ``None`` inputs match; one empty and one non-empty do not.
    """
    na, nb = normalize(a), normalize(b)
    if na == "" and nb == "":
        return 1.0
    return 1.0 if na == nb else 0.0


def _tokens(text: str) -> list[str]:
    return text.split(" ") if text else []


def token_f1_score(a: str | None, b: str | None) -> float:
    """Return whitespace-token F1 between the normalized inputs.

    Precision and recall are computed over token multisets, so repeated
    tokens count once per occurrence. Two empty inputs score 1.0; one
    empty and one non-empty score 0.0.
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
    """Return the character-level similarity of the normalized inputs.

    Uses ``difflib.SequenceMatcher.ratio()`` (Ratcliff/Obershelp), not
    Levenshtein distance, to avoid a third-party dependency. Two empty
    inputs score 1.0; one empty and one non-empty score 0.0.
    """
    na, nb = normalize(a), normalize(b)
    return SequenceMatcher(None, na, nb).ratio()
