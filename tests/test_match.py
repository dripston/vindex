"""
Tests for vindex.match's three match modes (Milestone 2.3). Each function
normalizes internally via vindex.normalize -- see test_normalize.py for
normalization-specific coverage (diacritics, numerals, script). These
tests focus on the comparison logic itself: exact/F1/char-similarity
scoring behavior, not normalization correctness.
"""

from vindex.match import char_similarity_score, exact_match_score, token_f1_score

# --- exact_match_score ---


def test_exact_match_identical_strings() -> None:
    assert exact_match_score("hello world", "hello world") == 1.0


def test_exact_match_case_and_punctuation_insensitive_via_normalize() -> None:
    assert exact_match_score("Hello, World!", "hello world") == 1.0


def test_exact_match_different_strings_scores_zero() -> None:
    assert exact_match_score("hello world", "goodbye world") == 0.0


def test_exact_match_single_word_difference_scores_zero() -> None:
    assert exact_match_score("namaskara", "namaskar") == 0.0


def test_exact_match_both_empty_scores_one() -> None:
    assert exact_match_score("", "") == 1.0
    assert exact_match_score(None, None) == 1.0


def test_exact_match_one_empty_scores_zero() -> None:
    assert exact_match_score("hello", "") == 0.0
    assert exact_match_score("", "hello") == 0.0
    assert exact_match_score(None, "hello") == 0.0


# --- token_f1_score ---


def test_token_f1_identical_strings_scores_one() -> None:
    assert token_f1_score("hello world", "hello world") == 1.0


def test_token_f1_partial_overlap() -> None:
    # "quick brown fox" overlaps 3 of 4 tokens each side.
    score = token_f1_score("the quick brown fox", "quick brown fox jumps")
    assert 0.7 < score < 0.8


def test_token_f1_no_overlap_scores_zero() -> None:
    assert token_f1_score("apple banana", "car truck") == 0.0


def test_token_f1_repeated_tokens_counted_as_multiset() -> None:
    # "a a a" vs "a a": overlap=2, precision=2/3, recall=2/2=1,
    # F1 = 2*(2/3*1)/(2/3+1) = 0.8
    score = token_f1_score("a a a", "a a")
    assert round(score, 3) == 0.8


def test_token_f1_both_empty_scores_one() -> None:
    assert token_f1_score("", "") == 1.0
    assert token_f1_score(None, None) == 1.0


def test_token_f1_one_empty_scores_zero() -> None:
    assert token_f1_score("hello world", "") == 0.0
    assert token_f1_score("", "hello world") == 0.0


def test_token_f1_ignores_word_order() -> None:
    assert token_f1_score("red blue green", "green red blue") == 1.0


# --- char_similarity_score ---


def test_char_similarity_identical_strings_scores_one() -> None:
    assert char_similarity_score("hello world", "hello world") == 1.0


def test_char_similarity_near_miss_spelling_scores_high() -> None:
    # Single-token mismatch that exact_match and token_f1 both fail
    # outright -- this is the case char similarity exists to catch.
    score = char_similarity_score("namaskara", "namaskar")
    assert score > 0.9


def test_char_similarity_totally_different_scores_low() -> None:
    score = char_similarity_score("hello", "xyzabc")
    assert score < 0.3


def test_char_similarity_both_empty_scores_one() -> None:
    assert char_similarity_score("", "") == 1.0
    assert char_similarity_score(None, None) == 1.0


def test_char_similarity_one_empty_scores_zero() -> None:
    assert char_similarity_score("hello", "") == 0.0
    assert char_similarity_score("", "hello") == 0.0


def test_char_similarity_is_symmetric() -> None:
    a, b = "namaskara", "namaskar"
    assert char_similarity_score(a, b) == char_similarity_score(b, a)


# --- cross-mode: same pair, different signal (motivates having all three) ---


def test_modes_disagree_on_single_token_spelling_variant() -> None:
    a, b = "namaskara", "namaskar"
    assert exact_match_score(a, b) == 0.0
    assert token_f1_score(a, b) == 0.0
    assert char_similarity_score(a, b) > 0.9
