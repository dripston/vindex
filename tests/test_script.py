"""
Ported from script_check.py's _run_tests() (Milestone 1.1). Every case
below existed in the original inline test runner; this file just moves
them into pytest, one assertion per test, with no logic changes.
"""

from vindex.script import classify, count_scripts, expected_script, is_script_adherent

# --- count_scripts ---


def test_count_scripts_pure_latin() -> None:
    c = count_scripts("Mumbai")
    assert c["devanagari_chars"] == 0
    assert c["latin_alpha_chars"] == 6
    assert c["other_chars"] == 0


def test_count_scripts_pure_devanagari() -> None:
    c = count_scripts("मुंबई")
    assert c["devanagari_chars"] == 5
    assert c["latin_alpha_chars"] == 0


def test_count_scripts_mixed_has_other_chars() -> None:
    c = count_scripts("Mumbai 100 °C!")
    assert c["other_chars"] > 0


def test_count_scripts_empty_string_all_zero() -> None:
    c = count_scripts("")
    assert (c["devanagari_chars"], c["latin_alpha_chars"], c["other_chars"]) == (0, 0, 0)


# --- classify: empty ---


def test_classify_empty_string() -> None:
    assert classify("") == "empty"


def test_classify_whitespace_only() -> None:
    assert classify("   ") == "empty"


def test_classify_none() -> None:
    assert classify(None) == "empty"


# --- classify: devanagari ---


def test_classify_pure_hindi() -> None:
    assert classify("महाराष्ट्र की राजधानी मुंबई है।") == "devanagari"


def test_classify_hindi_with_some_latin() -> None:
    assert classify("महाराष्ट्र की राजधानी Mumbai है और भारत का प्रमुख शहर है।") == "devanagari"


# --- classify: roman ---


def test_classify_pure_english() -> None:
    assert classify("Mumbai is the capital of Maharashtra.") == "roman"


def test_classify_pure_hinglish_no_devanagari() -> None:
    assert classify("Maharashtra ki rajdhani Mumbai hai.") == "roman"


def test_classify_roman_with_a_couple_devanagari_chars() -> None:
    assert classify("Maharashtra ki rajdhani Mumbai hai, matlab राजधानी.") == "roman"


# --- classify: mixed ---


def test_classify_balanced_code_mix() -> None:
    # devanagari_chars <= latin_alpha_chars <= devanagari_chars*2
    assert classify("Mumbai matlab राजधानी शहर है city") == "mixed"


def test_classify_boundary_latin_equals_devanagari_times_two() -> None:
    # exact boundary: latin == devanagari*2 -> NOT roman (needs strictly >), falls to mixed
    dev = "राजधानी"  # 7 devanagari chars (approx, doesn't matter for the boundary logic)
    dcount = count_scripts(dev)["devanagari_chars"]
    boundary_latin = "a" * (dcount * 2)  # latin_alpha_chars == devanagari_chars * 2 exactly
    boundary_text = dev + boundary_latin

    bc = count_scripts(boundary_text)
    assert bc["latin_alpha_chars"] == dcount * 2
    assert classify(boundary_text) == "mixed"


def test_classify_boundary_plus_one_latin_char_is_roman() -> None:
    dev = "राजधानी"
    dcount = count_scripts(dev)["devanagari_chars"]
    over_boundary_text = dev + "a" * (dcount * 2 + 1)  # one more latin char -> roman
    assert classify(over_boundary_text) == "roman"


# --- expected_script ---


def test_expected_script_en() -> None:
    assert expected_script("en") == {"roman"}


def test_expected_script_hi() -> None:
    assert expected_script("hi") == {"devanagari"}


def test_expected_script_hinglish() -> None:
    assert expected_script("hinglish") == {"roman", "mixed"}


def test_expected_script_unknown_variant_raises() -> None:
    try:
        expected_script("klingon")
        raise AssertionError("expected_script('klingon') should have raised ValueError")
    except ValueError:
        pass


# --- is_script_adherent ---


def test_is_script_adherent_english_text_en() -> None:
    assert is_script_adherent("Mumbai is the capital.", "en") is True


def test_is_script_adherent_hindi_text_en_is_false() -> None:
    assert is_script_adherent("मुंबई राजधानी है।", "en") is False


def test_is_script_adherent_hindi_text_hi() -> None:
    assert is_script_adherent("मुंबई राजधानी है।", "hi") is True


def test_is_script_adherent_roman_hinglish() -> None:
    assert is_script_adherent("Maharashtra ki rajdhani Mumbai hai.", "hinglish") is True


def test_is_script_adherent_devanagari_answer_to_hinglish_prompt_is_false() -> None:
    assert is_script_adherent("महाराष्ट्र की राजधानी मुंबई है।", "hinglish") is False


def test_is_script_adherent_empty_any_variant_is_false() -> None:
    assert is_script_adherent("", "hinglish") is False


def test_is_script_adherent_whitespace_only_hi_is_false() -> None:
    assert is_script_adherent("   ", "hi") is False


# --- real contamination example from results.json (the actual bug) ---


def test_real_contaminated_hinglish_answer_classifies_as_devanagari() -> None:
    contaminated = "हमारे सौर मंडल में 8 ग्रह हैं: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, यूरेनस और नेपच्यून।"
    assert classify(contaminated) == "devanagari"


def test_real_contaminated_hinglish_answer_not_adherent_for_hinglish() -> None:
    contaminated = "हमारे सौर मंडल में 8 ग्रह हैं: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, यूरेनस और नेपच्यून।"
    assert is_script_adherent(contaminated, "hinglish") is False


def test_real_roman_hinglish_answer_is_adherent() -> None:
    good_hinglish = "Hamare saur mandal mein aath grah hain."
    assert is_script_adherent(good_hinglish, "hinglish") is True
