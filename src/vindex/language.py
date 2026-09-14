"""
v0 Latin-script language detector: is this Roman text English, or Romanized
("Hinglish") Hindi?

script.py's classify() can tell Devanagari from Latin, but it cannot tell
English from Romanized Hindi -- both are Latin script. This module adds a
second, narrower signal on top of Latin-script text: does it contain Hindi
function words spelled in Roman letters?

v0 LIMITATIONS (read before trusting this for anything load-bearing):
  - Fixed, hand-picked word list. No coverage of Hindi verb conjugations,
    postpositions, or vocabulary beyond the ~16 function words below.
  - No handling of transliteration variants (e.g. "nahin" vs "nahi", "hen"
    vs "hain"). A word not spelled exactly as listed is invisible to this
    detector.
  - No support for other Romanized Indic languages (Romanized Tamil,
    Bengali, etc.) -- Hindi/Hinglish only.
  - Word-boundary regex matching only; no real tokenization, no stemming,
    no part-of-speech awareness. "hai" inside "chai" will not match
    (boundary-safe), but this is still a heuristic, not a parser.
  - Case-insensitive substring-of-words matching can still false-positive
    on English text that happens to contain one of these tokens as a
    stray word (e.g. proper nouns). One match is treated as a signal, not
    proof.
  - Not ML-based, not a language-ID library (e.g. langdetect, fasttext).
    Those would do better on long, ambiguous, or low-function-word text.
    This exists to be dependency-free and auditable, not maximally
    accurate.

Do not present this module's output as a confident language label. It is a
cheap first-pass heuristic, intended to be replaced or backed by a real
language-ID model later.
"""

from __future__ import annotations

import re

HINDI_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "hai",
        "hain",
        "kya",
        "nahi",
        "mera",
        "aap",
        "ka",
        "ki",
        "ke",
        "se",
        "mein",
        "tha",
        "hoga",
        "raha",
    }
)

_WORD_RE = re.compile(r"[A-Za-z]+")


def find_hindi_function_words(text: str | None) -> set[str]:
    """Return the set of Hindi function words (from HINDI_FUNCTION_WORDS)
    found in text, matched case-insensitively on whole words only."""
    text = text or ""
    words = (w.lower() for w in _WORD_RE.findall(text))
    return {w for w in words if w in HINDI_FUNCTION_WORDS}


def looks_like_hinglish(text: str | None) -> bool:
    """v0 heuristic: True if text (assumed Latin-script) contains at least
    one Hindi function word from HINDI_FUNCTION_WORDS.

    This says nothing about whether text is *also* valid English -- code
    mixing is the normal case for Hinglish. It only says "Hindi function
    words are present." See the module docstring for limitations.
    """
    return len(find_hindi_function_words(text)) > 0
