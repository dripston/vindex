"""
v0 English-loanword lookup for Hindi transliteration (Milestone 2.5).

Common English loanwords used inside Hindi/Hinglish sentences ("vah
doctor hai") have one established, conventional Devanagari spelling
("डॉक्टर") that is NOT what letter-by-letter ITRANS phonetic
transliteration produces. transliterate.py's ITRANS backend renders
"doctor" as "दोच्तोर्" -- phonetically plausible, but not the spelling
any Hindi speaker or dictionary actually uses. This is a real gap
transliterate.py's own docstring does not cover, because ITRANS is a
Sanskrit/formal-Hindi romanization scheme, not a loanword dictionary.

This module is a small, hand-picked word-level lookup table, consulted
BEFORE falling back to ITRANS. Same v0 shape and same honesty as
language.py's fixed function-word list: a short dictionary, not a
model, checked before assuming ITRANS output is usable.

WHY THIS MATTERS (see BUILD_PLAN.md / experiments docs): Sarvam's
published work solves this same loanword problem ("वह doctor" versus
"वह डॉक्टर") with an LLM call per case. A lookup-table transliteration
step solves the SAME cases deterministically, for free, and
reproducibly -- no API call, no latency, no temperature/model-version
sensitivity. This is a genuine advantage over an LLM-based normalizer
for exactly this narrow class of input (a fixed loanword vocabulary),
not a general replacement for one.

v0 LIMITATIONS:
  - Fixed, hand-picked word list (10 entries). No coverage beyond what
    is listed below.
  - Whole-word, case-insensitive match only. No inflection handling
    ("doctors", "hospitalized" are not recognized).
  - Devanagari only. No lookup table for the other 8 Indic scripts.
  - A loanword not in this list silently falls through to ITRANS
    phonetic transliteration, which -- per transliterate.py -- is
    frequently wrong for English loanwords specifically. This module
    narrows that gap, it does not close it.
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
    """Return the conventional Devanagari spelling for `word` if it is
    a known loanword (case-insensitive, whole word only), else None."""
    return LOANWORDS_DEVANAGARI.get(word.lower())


def substitute_known_loanwords(text: str) -> str:
    """Replace any whole-word match of a known loanword in `text` with
    its conventional Devanagari spelling, leaving everything else
    (including unknown words, left for ITRANS) unchanged."""

    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        devanagari = lookup_loanword(word)
        return devanagari if devanagari is not None else word

    return _WORD_RE.sub(_replace, text)
