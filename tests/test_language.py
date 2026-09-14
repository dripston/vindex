"""
Tests for vindex.language's v0 Hindi-function-word heuristic. See the
module docstring in src/vindex/language.py for the documented limitations
of this approach before extending these tests to cases it isn't meant to
handle (e.g. transliteration variants, non-Hindi Romanized languages).
"""

from vindex.language import find_hindi_function_words, looks_like_hinglish

# --- find_hindi_function_words ---


def test_find_hindi_function_words_pure_english_finds_none() -> None:
    assert find_hindi_function_words("Mumbai is the capital of Maharashtra.") == set()


def test_find_hindi_function_words_pure_hinglish() -> None:
    text = "Maharashtra ki rajdhani Mumbai hai."
    assert find_hindi_function_words(text) == {"ki", "hai"}


def test_find_hindi_function_words_real_hinglish_example() -> None:
    # Same sentence used in test_script.py's contamination regression test.
    text = "Hamare saur mandal mein aath grah hain."
    assert find_hindi_function_words(text) == {"mein", "hain"}


def test_find_hindi_function_words_is_case_insensitive() -> None:
    assert find_hindi_function_words("KYA yeh sahi HAI") == {"kya", "hai"}


def test_find_hindi_function_words_empty_string() -> None:
    assert find_hindi_function_words("") == set()


def test_find_hindi_function_words_none() -> None:
    assert find_hindi_function_words(None) == set()


def test_find_hindi_function_words_word_boundary_not_substring() -> None:
    # "hai" must not match inside "chai" -- whole-word matching only.
    assert find_hindi_function_words("Let's have some chai.") == set()


def test_find_hindi_function_words_ka_not_substring_of_karma() -> None:
    # "ka" must not match inside "karma".
    assert find_hindi_function_words("Karma is a concept, not a person.") == set()


# --- looks_like_hinglish ---


def test_looks_like_hinglish_true_for_hinglish_text() -> None:
    assert looks_like_hinglish("Maharashtra ki rajdhani Mumbai hai.") is True


def test_looks_like_hinglish_false_for_pure_english() -> None:
    assert looks_like_hinglish("Mumbai is the capital of Maharashtra.") is False


def test_looks_like_hinglish_false_for_empty() -> None:
    assert looks_like_hinglish("") is False


def test_looks_like_hinglish_false_for_none() -> None:
    assert looks_like_hinglish(None) is False


def test_looks_like_hinglish_single_function_word_is_enough() -> None:
    # v0 is deliberately permissive: one match is a signal, not proof.
    assert looks_like_hinglish("I think that is correct, kya?") is True
