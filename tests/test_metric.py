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


# ---------------------------------------------------------------------------
# Milestone 1.5: every script, every label, and edge-case inputs (empty,
# whitespace, numerals, emoji, punctuation, mixed-script response, and a
# third-language response classify() has no bucket for).
#
# Sentences reused from tests/test_script.py's Milestone 1.2 real-sentence
# set (each sourced from that language's own Wikipedia; see test_script.py
# for source URLs) -- not rewritten here, same prompt used as response to
# get an exact same-script "matched" case per script.
# ---------------------------------------------------------------------------

_SCRIPT_SENTENCES: dict[str, str] = {
    "devanagari": "महाराष्ट्र की राजधानी मुंबई है।",
    "kannada": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅತಿ ದೊಡ್ಡ ನಗರ ಮತ್ತು ರಾಜಧಾನಿ ಕೇಂದ್ರ",
    "tamil": "சென்னை தமிழ்நாட்டின் தலைநகரமும், இந்தியாவின் நான்காவது பெரிய நகரமும் ஆகும்.",
    "telugu": (
        "హైదరాబాద్ రాజ్యం ఒకప్పటి భారత సామ్రాజ్యంలో నిజాముల ఆధ్వర్యంలో "
        "ఉన్న అతిపెద్ద రాచరిక రాష్ట్రం."
    ),
    "bengali": "ঢাকা বাংলাদেশের রাজধানী ও মহানগর বা বৃহত্তম শহর।",
    "gujarati": (
        "અમદાવાદ ગુજરાત રાજ્યનું સૌથી મોટુંં અને વસ્તી પ્રમાણે ભારતનું "
        "પાંચમા અને શહેરી વસ્તી પ્રમાણે સાતમે ક્રમનું શહેર છે."
    ),
    "malayalam": "ഇന്ത്യയിലെ തീരദേശ കേരള സംസ്ഥാനത്തിലെ ഒരു വലിയ നഗരമാണ് കൊച്ചി.",
    "odia": "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ ।",
    "gurmukhi": "ਪੰਜਾਬ ਉੱਤਰ-ਦੱਖਣੀ ਏਸ਼ੀਆ ਵਿੱਚ ਇੱਕ ਭੂਗੋਲਿਕ, ਸੱਭਿਆਚਾਰਕ ਅਤੇ ਇਤਿਹਾਸਕ ਖਿੱਤਾ ਹੈ।",
}


def test_every_script_same_script_prompt_and_response_matched() -> None:
    for script, sentence in _SCRIPT_SENTENCES.items():
        r = script_adherence(sentence, sentence)
        assert r.label == "matched", f"{script}: expected matched, got {r.label}"
        assert r.passed is True, f"{script}: expected passed"
        assert r.detail["prompt_label"] == script


def test_every_script_cross_script_response_is_script_mismatch() -> None:
    # Each Indic-script prompt answered in a *different* Indic script.
    scripts = list(_SCRIPT_SENTENCES)
    for i, prompt_script in enumerate(scripts):
        response_script = scripts[(i + 1) % len(scripts)]
        r = script_adherence(_SCRIPT_SENTENCES[prompt_script], _SCRIPT_SENTENCES[response_script])
        msg = f"{prompt_script} prompt / {response_script} response: expected script_mismatch"
        assert r.label == "script_mismatch", msg
        assert r.passed is False


# --- whitespace-only ---


def test_whitespace_only_prompt_is_empty() -> None:
    r = script_adherence("   \t\n  ", "Mumbai is the capital.")
    assert r.label == "empty"
    assert r.passed is False


def test_whitespace_only_response_is_empty() -> None:
    r = script_adherence("Where is Mumbai?", "   \t\n  ")
    assert r.label == "empty"
    assert r.passed is False


# --- numerals-only ---


def test_numerals_only_response_to_romanized_prompt() -> None:
    # classify() has no digit bucket: an all-numeral string has zero
    # counted chars in every bucket, so dominant_n == l == 0, which falls
    # through classify()'s boundary logic to "mixed" (not "empty" -- it
    # isn't blank -- and not "roman", since l is not > 0). script_adherence
    # then distinguishes this from genuine code-mixing: a response with NO
    # alphabetic characters in any script is not a real answer in any
    # language, so it must not silently pass as "mixed" -- see
    # metric.py's no_script_signal label.
    r = script_adherence("Where is Mumbai?", "12345 100 42")
    assert r.detail["response_label"] == "mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


def test_numerals_only_prompt_is_not_empty_and_buckets_as_mixed() -> None:
    # The PROMPT bucketing (_prompt_bucket) is unaffected by the
    # no_script_signal fix -- it still buckets a numerals-only prompt as
    # code-mixed. But the RESPONSE here is also numerals-only, so it
    # trips the same no_script_signal check as the response-only tests
    # above: a numerals-only response is not a real answer regardless of
    # what bucket the prompt fell into.
    r = script_adherence("12345 100 42", "12345 100 42")
    assert r.detail["prompt_label"] == "mixed"
    assert r.detail["prompt_bucket"] == "code-mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


# --- emoji ---


def test_emoji_only_response_to_romanized_prompt() -> None:
    # Same zero-count boundary case as numerals: emoji aren't Latin
    # alphabetic or any Indic script, so classify() falls to "mixed" --
    # and script_adherence correctly refuses to count that as a pass
    # (no_script_signal), since an emoji-only response answers nothing.
    r = script_adherence("Where is Mumbai?", "🎉🎊😀👍")
    assert r.detail["response_label"] == "mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


def test_emoji_mixed_into_native_script_response_passes_as_mixed() -> None:
    r = script_adherence("मुंबई कहाँ है?", "मुंबई 🎉 महाराष्ट्र में है 👍।")
    assert r.detail["response_label"] == "devanagari"
    assert r.label == "matched"


# --- punctuation-only ---


def test_punctuation_only_response_to_romanized_prompt() -> None:
    # A punctuation-only response (e.g. a truncated/broken generation)
    # has no alphabetic content in any script -- must not silently pass
    # as "mixed" the way genuine code-mixing does.
    r = script_adherence("Where is Mumbai?", "... !!! ???")
    assert r.detail["response_label"] == "mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


def test_punctuation_only_prompt_buckets_as_code_mixed() -> None:
    r = script_adherence("... !!! ???", "Mumbai is the capital.")
    assert r.detail["prompt_label"] == "mixed"
    assert r.detail["prompt_bucket"] == "code-mixed"


# --- mixed-script response (explicit code-mixing across multiple buckets) ---


def test_mixed_script_response_to_native_script_prompt() -> None:
    r = script_adherence("मुंबई कहाँ है?", "Mumbai matlab राजधानी शहर है city")
    assert r.detail["response_label"] == "mixed"
    assert r.label == "mixed"
    assert r.passed is True


def test_mixed_script_response_to_romanized_prompt() -> None:
    r = script_adherence("Where is Mumbai?", "Mumbai matlab राजधानी शहर है city")
    assert r.detail["response_label"] == "mixed"
    assert r.label == "mixed"
    assert r.passed is True


# --- third-language response (not English, not Hindi, not any of the 9
# Indic scripts classify() knows about -- e.g. Russian/Cyrillic) ---


def test_third_language_cyrillic_response_to_romanized_prompt() -> None:
    # Cyrillic has zero chars in every counted bucket (same boundary case
    # as numerals/emoji/punctuation above) -- classify() cannot tell
    # "unrecognized script" from "no script-bearing content", both land on
    # "mixed". vindex only knows Latin + the 9 Indic scripts in
    # SCRIPT_RANGES, nothing else -- but a response in a genuinely
    # unrecognized script (not code-mixed with anything vindex DOES
    # recognize) must not be silently scored as a passing "mixed" answer
    # to an English/Hinglish prompt.
    russian = "Москва является столицей России."
    r = script_adherence("Where is Moscow?", russian)
    assert r.detail["response_label"] == "mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


def test_third_language_cyrillic_response_to_native_script_prompt() -> None:
    russian = "Москва является столицей России."
    r = script_adherence("मुंबई कहाँ है?", russian)
    assert r.detail["response_label"] == "mixed"
    assert r.label == "no_script_signal"
    assert r.passed is False


# --- lone danda response (degenerate/truncated input) ---


def test_lone_danda_response_to_native_script_prompt_is_not_a_confident_match() -> None:
    # Regression: classify("।") == "devanagari" (dominant_n=1 > l=0), so
    # a lone danda used to be scored a confident script "match" against
    # a Devanagari prompt, even though it is sentence-ending punctuation
    # with no actual Devanagari letter in it -- see
    # script.is_only_danda_punctuation.
    r = script_adherence("मुंबई कहाँ है?", "।")
    assert r.label == "no_script_signal"
    assert r.passed is False


def test_real_devanagari_response_with_trailing_danda_is_unaffected() -> None:
    # The danda fix must not affect a real Devanagari response that
    # simply ends with a danda, as any real sentence does.
    r = script_adherence("राजधानी क्या है?", "नई दिल्ली है।")
    assert r.label == "matched"
    assert r.passed is True
