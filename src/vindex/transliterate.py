"""
Transliteration backend for script_normalized_match (Milestone 2).

Converts romanized text to a native script so answers spelled differently
("namaskara" vs "namaskar" vs a native-script "नमस्कार") can be compared
after normalization instead of failing exact-match outright.

BACKEND DECISION (Milestone 2.1)

Two backends were evaluated on 30 hand-picked cases (romanized Hindi,
Kannada, Tamil, Telugu, using casual spelling as people actually type it,
not strict diacritic-marked romanization) -- see
experiments/scripts/transliteration_backend_eval.py.

- IndicXlit (ai4bharat-transliteration): could not be evaluated. It
  depends on fairseq, which fails at metadata-generation
  (FileNotFoundError: fairseq/version.txt) on Python 3.14, with or
  without --no-build-isolation. fairseq is unmaintained; this is not a
  local config problem. An install-time failure with no workaround found
  is treated as a disqualifying result in its own right.
- indic_transliteration (this package's pick): installs cleanly, no
  native build step. Scored 3/30 exact matches, 4/30 after stripping a
  trailing virama (्) artifact of ITRANS rendering a final consonant
  literally. The real gap: its ITRANS input scheme requires strict
  long/short vowel marking ("namaskAra", "pAnI") that casual romanized
  input never provides ("namaskara", "pani"), so vowel length is wrong
  on most casual-spelling input. This is a known, honest limitation, not
  a bug -- see KNOWN LIMITATION below.

Picked indic_transliteration because it is the only one of the two that
installs at all in this environment, and because a swappable interface
(transliterate() below) means the backend can change later without
touching call sites, per BUILD_PLAN.md 2.1 ("wrap it behind an interface
so it can be swapped").

KNOWN LIMITATION: transliterate() will mis-render vowel length on casual
romanized input with no diacritics (most real Hinglish text). This
degrades script_normalized_match's ability to match "pani" against a
native-script "पानी" without an additional normalization step (e.g.
stripping vowel-length distinctions before comparing). That
normalization is not implemented here -- this module only wraps the
backend, per 2.1's scope.
"""

from __future__ import annotations

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _sanscript_transliterate

DEVANAGARI = sanscript.DEVANAGARI
KANNADA = sanscript.KANNADA
TAMIL = sanscript.TAMIL
TELUGU = sanscript.TELUGU
MALAYALAM = sanscript.MALAYALAM
BENGALI = sanscript.BENGALI
GUJARATI = sanscript.GUJARATI
GURMUKHI = sanscript.GURMUKHI
ORIYA = sanscript.ORIYA


def transliterate(text: str, to_script: str) -> str:
    """Convert romanized text to `to_script` using the ITRANS input scheme.

    `to_script` is one of the script constants above. Returns `text`
    unchanged if it is empty.
    """
    if not text:
        return text
    result: str = _sanscript_transliterate(text, sanscript.ITRANS, to_script)
    return result
