"""
Normalization pipeline for script_normalized_match (Milestone 2.2).

Pipeline order (each step operates on the previous step's output):

  1. transliterate  -- romanized text is converted to a target native
                        script via vindex.transliterate, but ONLY if
                        vindex.language.looks_like_hinglish() also says
                        it contains Hindi function words. classify()
                        alone cannot tell English from Romanized Hindi --
                        both are Latin script (see script.py) -- so
                        gating on classify() == "roman" alone would
                        transliterate plain English ("Mumbai kahan hai?"
                        would garble "Mumbai" itself). Text already in a
                        native script, plain English, or empty passes
                        through unchanged at this step.
  2. lowercase       -- str.lower(). Only affects Latin letters; Indic
                        scripts have no case.
  3. strip diacritics -- NFKD-decompose, drop combining marks (Unicode
                        category Mn). This is a LATIN-ONLY operation --
                        see KNOWN LIMITATION below for why it cannot run
                        on Indic text.
  4. normalize whitespace and punctuation -- collapse all whitespace
                        (category Zs plus \\t\\n\\r) to a single space,
                        strip leading/trailing space, and drop
                        punctuation (any Unicode category starting with
                        "P" -- this includes the Devanagari danda ।/॥,
                        deliberately, per script.py's note that danda is
                        shared cross-script sentence punctuation, not
                        content) EXCEPT a "-" directly before a digit
                        (a numeric sign) or a "."/"," directly between
                        two digits (a decimal point or thousands
                        separator, both normalized to "."), which are
                        preserved because dropping them silently
                        changes a number's VALUE -- see KNOWN LIMITATION
                        below and _strip_punctuation's docstring.
  5. reconcile numerals -- every Unicode decimal digit, in any of the 9
                        Indic scripts or ASCII, is mapped to its ASCII
                        digit via unicodedata.digit(). Verified against
                        all 9 scripts' native digit ranges (Devanagari,
                        Kannada, Tamil, Telugu, Bengali, Gujarati,
                        Gurmukhi, Malayalam, Odia): each maps 0-9
                        correctly with no table needed, since Unicode
                        assigns every decimal-digit codepoint its numeric
                        value directly.

KNOWN LIMITATION: code-mixed text gets ONE transliteration decision for
the whole string, not per-word. "Mumbai kahan hai?" contains the Hindi
function word "hai", so looks_like_hinglish() correctly calls the whole
sentence Hinglish -- but transliterate() then converts "Mumbai" too,
which ITRANS mangles ("Mumbai" is not valid ITRANS input). There is no
per-word language tagging here; language.py is a whole-string v0
heuristic by design (see its own module docstring), and word-level
language identification is out of scope for this pipeline. This means
normalize() is least reliable exactly on genuinely code-mixed sentences
that mix English proper nouns with Hindi function words -- a real gap,
not a rare one, given how common that pattern is in Hinglish chat.

KNOWN LIMITATION: diacritic-stripping (step 3) must never run on Indic
script text. Indic vowel signs and the virama/halant are represented as
Unicode combining marks (category Mn/Mc) that are structurally part of
the letter, not decoration on it -- running NFKD+strip-Mn on Devanagari
turns "नमस्कार" into "नमसकार" (the halant, which marks a consonant
conjunct, is deleted, silently changing the word). normalize() therefore
only applies step 3 to the portion of text classified as Latin-script
(via vindex.script.classify), never to native-script text. This means a
mixed-script string's Indic portion is normalized only by steps 4-5, not
step 3 -- accepted as correct behavior, not a gap, since step 3 does not
apply to Indic text at all.

KNOWN LIMITATION: a thousands separator and a decimal point are
indistinguishable by this module once digit-adjacency is the only
signal -- "1,200" (one thousand two hundred) and "1.200" (a decimal,
one point two) both normalize to "1.200" here. This module cannot
recover which one was meant from the string alone (that needs a
locale, which is out of scope); it only fixes the strictly worse prior
behavior of silently dropping the sign or separator entirely and
reporting "1200" and "12" as an exact match against "1,200"/"1.200"
respectively.

This module does not decide whether two normalized strings "match" --
that is script_normalized_match's job (Milestone 2.3, not yet built).
normalize() only produces the comparable form.
"""

from __future__ import annotations

import re
import unicodedata

from vindex.language import looks_like_hinglish
from vindex.script import classify
from vindex.transliterate import DEVANAGARI, transliterate

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_latin_diacritics(text: str) -> str:
    """Drop combining marks (category Mn) that decorate an ASCII base
    letter (e.g. NFKD-decomposed "a" + combining macron from "ā").

    Must not drop Mn marks attached to Indic base letters -- those are
    vowel signs and the virama, structurally part of the letter, not
    decoration. NFKD is a no-op on Indic text (verified empirically), so
    scoping the drop to "preceding base char is ASCII" is sufficient.
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
    """True if text[i] is a "-" acting as a numeric sign, or a "."/","
    acting as a decimal point/thousands separator -- i.e. attached
    directly to digits, not a hyphen in a word or sentence punctuation.
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
    """Drop punctuation, except a numeric sign or decimal/thousands
    separator directly attached to digits.

    A blanket "drop every Unicode category P* character" also drops the
    minus sign (category Pd) and the decimal point/comma (category Po)
    -- both of which change a NUMBER'S VALUE, not just its surface form.
    Without this guard, normalize("-5") == normalize("5") and
    normalize("100.5") == normalize("1005"), so exact_match_score would
    report a sign flip or a 10x magnitude error as a perfect match. Loan
    words with an internal hyphen ("e-mail") are unaffected: a "-" only
    counts as a sign when the character before it is not alphanumeric
    (so "e-mail" and "5-6" both keep their hyphen as ordinary
    punctuation, since "e" and "5" are alnum, while "-5" and
    "temperature is -5" keep theirs as a sign). A "."/"," only counts as
    a numeric separator when digits sit on both sides, so a sentence-
    ending period after a digit ("...costs 5.") is still stripped.
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
    """Normalize text for script-and-transliteration-tolerant comparison.

    Romanized input is transliterated to `to_script` (default Devanagari)
    before the remaining steps run. Returns "" for None or empty input.
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
