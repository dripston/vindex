"""Romanized-to-native-script transliteration.

Converts Romanized text to a native Indic script so that differently
spelled answers ("namaskar" vs "नमस्कार") can be compared after
normalization. Backed by ``indic_transliteration`` using the ITRANS input
scheme, behind the :func:`transliterate` function so the backend can be
replaced without changing callers.

When the target is Devanagari, known English loanwords are first
replaced with their conventional spelling (see :mod:`vindex.loanwords`),
since ITRANS renders "doctor" as दोच्तोर् rather than डॉक्टर.

Limitation: ITRANS expects explicit long/short vowel marking
("namaskAra", "pAnI"), which casual Romanized input never has, so vowel
length is frequently wrong (e.g. "pani" does not become पानी) and a
trailing virama may appear. Exact matches against native-script text
are therefore unreliable for casual Romanized input; prefer the fuzzier
scores in :mod:`vindex.match`.
"""

from __future__ import annotations

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _sanscript_transliterate

from vindex.loanwords import substitute_known_loanwords

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
    """Transliterate Romanized text to a native script.

    Args:
        text: Romanized input (ITRANS scheme).
        to_script: Target script, one of this module's script constants
            (e.g. :data:`DEVANAGARI`, :data:`TAMIL`).

    Returns:
        The transliterated text, or ``text`` unchanged if it is empty.
    """
    if not text:
        return text
    if to_script == DEVANAGARI:
        text = substitute_known_loanwords(text)
    result: str = _sanscript_transliterate(text, sanscript.ITRANS, to_script)
    return result
