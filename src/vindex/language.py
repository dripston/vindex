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
    proof -- but in metric.py's script_adherence, one match on the PROMPT
    is enough to flip the whole prompt's bucket to code-mixed and then
    hard-fail a genuinely correct English response as language_mismatch.
    Verified real cases: "Who directed Se7en?" contains "se" only because
    the digit splits the word into two alphabetic runs; "What is the ka
    in Egyptian belief?" contains "ka" as an ordinary English word. A
    stricter "require 2+ matches" rule was tried and reverted -- it also
    breaks short, genuine Hinglish questions, including this package's
    own README example ("Mumbai kahan hai?" has exactly one function
    word). This is a real, unresolved trade-off of the v0 approach, not
    a solved edge case.
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


# A stricter "2+ words" variant was tried and reverted (see
# metric.py's language_mismatch docstring): it fixes the "Se7en"/"ka"
# single-token false positive, but also breaks this package's own
# canonical example ("Mumbai kahan hai?" has exactly one function
# word, "hai") and any other short, genuine Hinglish question. The
# false positive is a real, honest v0 heuristic limitation, not
# something a simple threshold change can fix without breaking
# legitimate short-sentence detection.
