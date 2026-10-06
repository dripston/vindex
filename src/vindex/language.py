"""Heuristic detector for Romanized Hindi (Hinglish) in Latin-script text.

:func:`vindex.script.classify` distinguishes Devanagari from Latin but
cannot tell English from Romanized Hindi, since both are Latin script.
This module adds a narrow, dependency-free signal: whether the text
contains any of a fixed set of Hindi function words spelled in Roman
letters.

Limitations:

- Fixed list of 14 function words; no verb conjugations, postpositions,
  or other vocabulary, and no spelling variants ("nahin" vs "nahi").
- Hindi only; other Romanized Indic languages are not detected.
- Whole-word, case-insensitive matching only ("hai" does not match
  inside "chai"), with no tokenization or part-of-speech awareness.
- A single match counts, so English text containing one of these
  tokens can be misdetected (e.g. "Who directed Se7en?" yields "se";
  "the ka in Egyptian belief" yields "ka"). In
  :func:`vindex.script_adherence` this can flip a prompt to code-mixed
  and fail a correct English response as ``language_mismatch``.
  Requiring two or more matches would instead miss short genuine
  Hinglish such as "Mumbai kahan hai?".

This is a cheap first-pass heuristic, not a language-ID model; do not
treat its output as a confident language label.
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
    """Return the Hindi function words found in ``text``.

    Args:
        text: Text to scan. ``None`` is treated as empty.

    Returns:
        The words from :data:`HINDI_FUNCTION_WORDS` present in ``text``,
        matched case-insensitively on whole words.
    """
    text = text or ""
    words = (w.lower() for w in _WORD_RE.findall(text))
    return {w for w in words if w in HINDI_FUNCTION_WORDS}


def looks_like_hinglish(text: str | None) -> bool:
    """Return True if ``text`` contains at least one Hindi function word.

    This only signals that Hindi function words are present; it says
    nothing about whether the text is also valid English, since code
    mixing is normal for Hinglish. See the module docstring for
    limitations.

    Args:
        text: Latin-script text. ``None`` is treated as empty.
    """
    return len(find_hindi_function_words(text)) > 0

