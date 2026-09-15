"""
Tests for vindex.loanwords' v0 English-loanword lookup table. See the
module docstring in src/vindex/loanwords.py for the documented
limitations (10-word fixed list, Devanagari only, no inflections)
before extending these tests to cases it isn't meant to handle.
"""

from vindex.loanwords import lookup_loanword, substitute_known_loanwords

# --- lookup_loanword ---


def test_lookup_loanword_known_word() -> None:
    assert lookup_loanword("doctor") == "डॉक्टर"


def test_lookup_loanword_is_case_insensitive() -> None:
    assert lookup_loanword("DOCTOR") == "डॉक्टर"
    assert lookup_loanword("Doctor") == "डॉक्टर"


def test_lookup_loanword_unknown_word_returns_none() -> None:
    assert lookup_loanword("namaskara") is None
    assert lookup_loanword("tune") is None


def test_lookup_loanword_inflected_form_not_recognized() -> None:
    # v0 limitation: whole-word match only, no inflection handling.
    assert lookup_loanword("doctors") is None


# --- substitute_known_loanwords ---


def test_substitute_known_loanwords_replaces_known_word() -> None:
    assert substitute_known_loanwords("vah doctor hai") == "vah डॉक्टर hai"


def test_substitute_known_loanwords_leaves_unknown_words_unchanged() -> None:
    assert substitute_known_loanwords("vah tune kiya") == "vah tune kiya"


def test_substitute_known_loanwords_multiple_in_one_sentence() -> None:
    result = substitute_known_loanwords("mujhe hospital aur office jana hai")
    assert result == "mujhe अस्पताल aur ऑफिस jana hai"


def test_substitute_known_loanwords_empty_string() -> None:
    assert substitute_known_loanwords("") == ""
