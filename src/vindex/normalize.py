"""Text normalization for script- and transliteration-tolerant comparison.

:func:`normalize` produces a comparable form of a string; the scores in
:mod:`vindex.match` build on it. Steps, in order:

1. **Transliterate.** Latin-script text that contains Hindi function
   words (:func:`vindex.language.looks_like_hinglish`) is transliterated
   to the target script (Devanagari by default). Plain English, native
   script, and empty text pass through unchanged.
2. **Lowercase.** Affects Latin letters only.
3. **Strip Latin diacritics.** NFKD-decompose and drop combining marks
   on ASCII base letters only; Indic vowel signs and virama are never
   removed, since that would change the word (नमस्कार -> नमसकार).
4. **Strip punctuation.** Drop all Unicode punctuation, including the
   danda (।॥), except a ``-`` used as a numeric sign or a ``.``/``,``
   between two digits (normalized to ``.``), which would otherwise
   change a number's value.
5. **Reconcile numerals.** Map decimal digits from all 9 supported Indic
   scripts to ASCII.
6. **Collapse whitespace.**

Limitations:

- Transliteration is decided once per string, not per word, so English
  proper nouns in Hinglish text are transliterated too ("Mumbai kahan
  hai?" also converts "Mumbai", which ITRANS handles poorly). Normalization
  is least reliable on code-mixed sentences.
- Thousands separators and decimal points are indistinguishable:
  ``"1,200"`` and ``"1.200"`` both normalize to ``"1.200"``.
"""

from __future__ import annotations

import re
import unicodedata

from vindex.language import looks_like_hinglish
from vindex.script import classify
from vindex.transliterate import DEVANAGARI, transliterate

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_latin_diacritics(text: str) -> str:
    """Drop combining marks (category Mn) attached to ASCII base letters.

    Marks attached to Indic letters (vowel signs, virama) are kept.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    out = []
    last_base_is_ascii = False
    for c in nfkd:
        if unicodedata.category(c) == "Mn":
            if not last_base_is_ascii:
                out.append(c)
            continue
        out.append(c)
        last_base_is_ascii = ord(c) < 128
    return "".join(out)


def _reconcile_numerals(text: str) -> str:
    out = []
    for c in text:
        if c.isdigit():
            digit = unicodedata.digit(c, None)
            out.append(str(digit) if digit is not None else c)
        else:
            out.append(c)
    return "".join(out)


def _is_protected_numeric_punctuation(text: str, i: int) -> bool:
    """Return True if ``text[i]`` is numeric punctuation to keep.

    That is a ``-`` acting as a sign (followed by a digit and not preceded
    by an alphanumeric character), or a ``.``/``,`` between two digits.
    """
    c = text[i]
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if c == "-":
        return nxt.isdigit() and not prev.isalnum()
    if c in ".,":
        return prev.isdigit() and nxt.isdigit()
    return False


def _strip_punctuation(text: str) -> str:
    """Drop punctuation, except numeric signs and digit separators.

    Keeps ``-5`` and ``100.5`` distinct from ``5`` and ``1005``. Hyphens in
    words (``"e-mail"``) and ranges (``"5-6"``) are stripped as ordinary
    punctuation, as is a sentence-ending period after a digit. A ``,``
    between digits is normalized to ``.``.
    """
    out = []
    for i, c in enumerate(text):
        if unicodedata.category(c).startswith("P") and not _is_protected_numeric_punctuation(
            text, i
        ):
            continue
        out.append("." if c == "," and _is_protected_numeric_punctuation(text, i) else c)
    return "".join(out)


def normalize(text: str | None, to_script: str = DEVANAGARI) -> str:
    """Normalize text for script- and transliteration-tolerant comparison.

    Args:
        text: Text to normalize. ``None`` is treated as empty.
        to_script: Target script for transliterating Romanized Hindi.

    Returns:
        The normalized string, or ``""`` for empty input. See the module
        docstring for the steps applied.
    """
    text = text or ""
    if text.strip() == "":
        return ""

    if classify(text) == "roman" and looks_like_hinglish(text):
        text = transliterate(text, to_script)

    text = text.lower()
    text = _strip_latin_diacritics(text)
    text = _strip_punctuation(text)
    text = _reconcile_numerals(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
