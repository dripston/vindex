"""Tests for vindex.metric.script_adherence. See metric.py's module
docstring for the prompt-bucket rules and label definitions these tests
check against."""

from vindex.metric import script_adherence
from vindex.result import MetricResult

# --- empty ---


def test_empty_prompt() -> None:
    r = script_adherence("", "Mumbai is the capital.")
    assert r.label == "empty"
    assert r.passed is False


def test_empty_response() -> None:
    r = script_adherence("Mumbai kahan hai?", "")
    assert r.label == "empty"
    assert r.passed is False


def test_both_none() -> None:
    r = script_adherence(None, None)
    assert r.label == "empty"
    assert r.passed is False


def test_returns_metric_result() -> None:
    r = script_adherence("Mumbai kahan hai?", "Mumbai Maharashtra mein hai.")
    assert isinstance(r, MetricResult)


# --- native-script prompt ---


def test_native_script_same_script_matched() -> None:
    r = script_adherence("मुंबई कहाँ है?", "मुंबई महाराष्ट्र में है।")
    assert r.label == "matched"
    assert r.passed is True
    assert r.score == 1.0


def test_native_script_response_mixed_passes_as_mixed() -> None:
    r = script_adherence("मुंबई कहाँ है?", "Mumbai matlab राजधानी शहर है city")
    assert r.label == "mixed"
    assert r.passed is True


def test_native_script_response_roman_is_script_mismatch() -> None:
    r = script_adherence("मुंबई कहाँ है?", "Mumbai is in Maharashtra.")
    assert r.label == "script_mismatch"
    assert r.passed is False


def test_native_script_response_different_indic_script_is_script_mismatch() -> None:
    # Hindi prompt, Tamil response -- wrong script, still Indic.
    hindi_prompt = "मुंबई कहाँ है?"
    tamil_response = "சென்னை தமிழ்நாட்டின் தலைநகரமும் ஆகும்."
    r = script_adherence(hindi_prompt, tamil_response)
    assert r.label == "script_mismatch"
    assert r.passed is False


# --- romanized prompt (Latin script, no Hindi function words) ---


def test_romanized_prompt_roman_response_matched() -> None:
    r = script_adherence("Where is Mumbai?", "Mumbai is in Maharashtra.")
    assert r.label == "matched"
    assert r.passed is True


def test_romanized_prompt_mixed_response_passes_as_mixed() -> None:
    r = script_adherence("Where is Mumbai?", "Mumbai matlab राजधानी शहर है city")
    assert r.label == "mixed"
    assert r.passed is True


def test_romanized_prompt_devanagari_response_is_script_mismatch() -> None:
    r = script_adherence("Where is Mumbai?", "मुंबई महाराष्ट्र में है।")
    assert r.label == "script_mismatch"
    assert r.passed is False


# --- code-mixed prompt (mixed script, or Romanized Hindi/Hinglish) ---


def test_code_mixed_script_prompt_roman_response_matched() -> None:
    r = script_adherence("Mumbai matlab kya hai city mein", "Mumbai matlab bada shahar hai.")
    assert r.label == "matched"
    assert r.passed is True


def test_hinglish_prompt_roman_hinglish_response_matched() -> None:
    r = script_adherence("Mumbai kahan hai?", "Mumbai Maharashtra mein hai.")
    assert r.label == "matched"
    assert r.passed is True


def test_hinglish_prompt_roman_english_response_is_language_mismatch() -> None:
    # Prompt is Romanized Hindi (has function words); response is Roman
    # script but plain English with zero Hindi function words.
    r = script_adherence("Mumbai kahan hai?", "Mumbai is the capital of Maharashtra.")
    assert r.label == "language_mismatch"
    assert r.passed is False


def test_code_mixed_prompt_devanagari_response_is_script_mismatch() -> None:
    r = script_adherence("Mumbai matlab kya hai city mein", "मुंबई महाराष्ट्र में है।")
    assert r.label == "script_mismatch"
    assert r.passed is False


# --- detail payload sanity ---


def test_detail_contains_prompt_and_response_labels() -> None:
    r = script_adherence("Mumbai kahan hai?", "Mumbai Maharashtra mein hai.")
    assert r.detail["prompt_label"] == "roman"
    assert r.detail["response_label"] == "roman"
    assert r.detail["prompt_bucket"] == "code-mixed"
