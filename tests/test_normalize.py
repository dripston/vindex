"""
Tests for vindex.normalize's normalization pipeline. See the module
docstring in src/vindex/normalize.py for the documented limitations --
in particular, code-mixed English-proper-noun-plus-Hindi-function-word
sentences get one whole-string transliteration decision, not per-word,
and can garble the English portion. Tests below cover that limitation
directly rather than pretending it doesn't exist.
"""

from vindex.normalize import normalize
from vindex.transliterate import KANNADA

# --- empty / whitespace ---


def test_normalize_none_is_empty_string() -> None:
    assert normalize(None) == ""


def test_normalize_empty_string_is_empty_string() -> None:
    assert normalize("") == ""


def test_normalize_whitespace_only_is_empty_string() -> None:
    assert normalize("   \t\n  ") == ""


# --- lowercase ---


def test_normalize_lowercases_latin() -> None:
    assert normalize("HELLO World") == "hello world"


def test_normalize_does_not_case_fold_devanagari() -> None:
    # Devanagari has no case; must pass through unchanged by this step.
    assert normalize("नमस्ते") == "नमस्ते"


# --- diacritic stripping (Latin-only) ---


def test_normalize_strips_latin_diacritics() -> None:
    assert normalize("café") == "cafe"
    assert normalize("résumé naïve") == "resume naive"


def test_normalize_does_not_corrupt_devanagari_combining_marks() -> None:
    # Regression: NFKD+blanket-strip-Mn deletes the halant (्, U+094D),
    # which marks a consonant conjunct and is not decoration.
    assert normalize("नमस्कार") == "नमस्कार"


def test_normalize_does_not_corrupt_kannada_vowel_signs() -> None:
    # Regression: Kannada vowel signs (e.g. ೆ, U+0CC6) are category Mn,
    # same as a Latin combining diacritic, but are structural, not
    # decorative -- must survive normalization on Kannada base letters.
    assert normalize("ಬೆಂಗಳೂರು", to_script=KANNADA) == "ಬೆಂಗಳೂರು"


# --- whitespace and punctuation normalization ---


def test_normalize_collapses_internal_whitespace() -> None:
    assert normalize("hello    world\t\nfoo") == "hello world foo"


def test_normalize_strips_leading_trailing_whitespace() -> None:
    assert normalize("  hello world  ") == "hello world"


def test_normalize_strips_latin_punctuation() -> None:
    assert normalize("Hello, world!") == "hello world"


def test_normalize_strips_devanagari_danda() -> None:
    assert normalize("नमस्ते।") == "नमस्ते"
    assert normalize("नमस्ते॥") == "नमस्ते"


# --- numeral reconciliation ---


def test_normalize_reconciles_devanagari_numerals_to_ascii() -> None:
    assert normalize("मेरे पास १२३ रुपये हैं") == normalize("मेरे पास 123 रुपये हैं")


def test_normalize_reconciles_kannada_numerals_to_ascii() -> None:
    assert normalize("೧೨೩", to_script=KANNADA) == "123"


def test_normalize_reconciles_tamil_numerals_to_ascii() -> None:
    assert normalize("௧௨௩") == "123"


def test_normalize_leaves_ascii_numerals_unchanged() -> None:
    assert normalize("123") == "123"


# --- transliteration gating: romanized Hindi vs plain English ---


def test_normalize_transliterates_romanized_hindi() -> None:
    # "namaskar" alone has no Hindi function word from language.py's
    # fixed list, so looks_like_hinglish() is False for it -- use a
    # sentence that actually contains one ("hai").
    text = "yeh mera ghar hai"
    assert normalize(text) != text
    assert normalize(text).strip() != ""


def test_normalize_does_not_transliterate_plain_english() -> None:
    # "Mumbai is the capital of Maharashtra." has no Hindi function word,
    # so looks_like_hinglish() is False and it must stay in Latin script.
    assert normalize("Mumbai is the capital of Maharashtra.") == (
        "mumbai is the capital of maharashtra"
    )


def test_normalize_code_mixed_transliterates_whole_string_known_limitation() -> None:
    # Documented limitation: "hai" triggers looks_like_hinglish() for the
    # whole sentence, so "Mumbai" gets transliterated too, and ITRANS
    # mangles it since "Mumbai" is not valid ITRANS input. This test
    # pins the current (imperfect) behavior so a future change to it is
    # a deliberate decision, not a silent regression.
    result = normalize("Mumbai kahan hai?")
    assert result != "mumbai kahan hai"
    assert result != ""


# --- idempotence ---


def test_normalize_is_idempotent_on_native_script() -> None:
    once = normalize("नमस्ते, आप कैसे हैं?")
    assert normalize(once) == once


def test_normalize_is_idempotent_on_plain_english() -> None:
    once = normalize("Hello, World!")
    assert normalize(once) == once
