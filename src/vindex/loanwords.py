"""English-loanword lookup for Hindi transliteration.

Common English loanwords in Hinglish ("vah doctor hai") have an
established Devanagari spelling (डॉक्टर) that letter-by-letter ITRANS
transliteration does not produce (it gives दोच्तोर्).
:func:`vindex.transliterate.transliterate` consults this table before
falling back to ITRANS, which handles these cases deterministically and
without an LLM call.

Limitations:

- Fixed list of 10 words; anything else falls through to ITRANS, which
  is often wrong for English loanwords.
- Whole-word, case-insensitive matching only; no inflections
  ("doctors" is not recognized).
- Devanagari only.
"""

from __future__ import annotations

import re

LOANWORDS_DEVANAGARI: dict[str, str] = {
    "doctor": "डॉक्टर",
    "hospital": "अस्पताल",
    "school": "स्कूल",
    "college": "कॉलेज",
    "computer": "कंप्यूटर",
    "internet": "इंटरनेट",
    "mobile": "मोबाइल",
    "office": "ऑफिस",
    "station": "स्टेशन",
    "manager": "मैनेजर",
}

_WORD_RE = re.compile(r"[A-Za-z]+")


def lookup_loanword(word: str) -> str | None:
    """Return the conventional Devanagari spelling of a known loanword.

    Args:
        word: A single word, matched case-insensitively.

    Returns:
        The Devanagari spelling, or ``None`` if ``word`` is not in
        :data:`LOANWORDS_DEVANAGARI`.
    """
    return LOANWORDS_DEVANAGARI.get(word.lower())


def substitute_known_loanwords(text: str) -> str:
    """Replace every known loanword in ``text`` with its Devanagari spelling.

    Unknown words are left unchanged.

    Args:
        text: Latin-script text.

    Returns:
        ``text`` with known loanwords substituted.
    """

    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        devanagari = lookup_loanword(word)
        return devanagari if devanagari is not None else word

    return _WORD_RE.sub(_replace, text)
