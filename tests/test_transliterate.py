"""
Tests for vindex.transliterate. Covers the ITRANS wrapper itself and
its Milestone 2.5 loanword-substitution integration (see
src/vindex/loanwords.py for the lookup table and its rationale).
"""

from vindex.transliterate import DEVANAGARI, KANNADA, transliterate

# --- basic wrapper behavior ---


def test_transliterate_empty_string_passes_through() -> None:
    assert transliterate("", DEVANAGARI) == ""


def test_transliterate_converts_to_devanagari() -> None:
    result = transliterate("kaise", DEVANAGARI)
    assert result == "कैसे"


def test_transliterate_converts_to_kannada() -> None:
    result = transliterate("dhanyavaada", KANNADA)
    assert result == "ಧನ್ಯವಾದ"


# --- loanword substitution (Milestone 2.5) ---


def test_transliterate_substitutes_known_loanword_before_itrans() -> None:
    # "doctor" via plain ITRANS is "दोच्तोर्" (phonetic, not the real
    # spelling). Loanword substitution must produce the conventional
    # spelling "डॉक्टर" instead.
    result = transliterate("vah doctor hai", DEVANAGARI)
    assert "डॉक्टर" in result
    assert "दोच्तोर्" not in result


def test_transliterate_loanword_substitution_only_applies_to_devanagari() -> None:
    # The lookup table is Devanagari-only (see loanwords.py); a
    # non-Devanagari target script must fall through to plain ITRANS
    # phonetic transliteration for "doctor", unchanged.
    result = transliterate("doctor", KANNADA)
    assert "ಡಾಕ್ಟರ್" not in result  # the real Kannada loanword spelling
    assert result != ""


def test_transliterate_unknown_word_unaffected_by_loanword_table() -> None:
    # "tune" is a genuine many-to-many ambiguity (loanword "tune" vs
    # pronoun+postposition "तूने") not in the loanword table -- must be
    # left to plain ITRANS, unchanged by this milestone.
    assert transliterate("tune", DEVANAGARI) == "तुने"
