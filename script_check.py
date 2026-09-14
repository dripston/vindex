"""
Deterministic script (writing-system) classifier. No LLM involved.

Used to verify that a model's response is actually written in the script
the experiment asked for -- e.g. that a "Romanized Hinglish" answer is
actually in Roman letters, not Devanagari.
"""
import re

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")


def count_scripts(text):
    """Count characters by script bucket.

    devanagari_chars : Unicode U+0900-U+097F (Devanagari block)
    latin_alpha_chars: ASCII alphabetic (A-Z, a-z)
    other_chars      : everything else (digits, punctuation, whitespace,
                        symbols, non-Devanagari/non-Latin scripts)
    """
    text = text or ""
    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    latin_alpha_chars = len(LATIN_ALPHA_RE.findall(text))
    other_chars = len(text) - devanagari_chars - latin_alpha_chars
    return {
        "devanagari_chars": devanagari_chars,
        "latin_alpha_chars": latin_alpha_chars,
        "other_chars": other_chars,
    }


def classify(text):
    """Classify a string's dominant script.

    empty      : blank or whitespace only
    devanagari : devanagari_chars > latin_alpha_chars
    roman      : latin_alpha_chars > devanagari_chars * 2
    mixed      : otherwise (includes ties, and cases where devanagari_chars
                 <= latin_alpha_chars <= devanagari_chars * 2)
    """
    if text is None or text.strip() == "":
        return "empty"
    counts = count_scripts(text)
    d = counts["devanagari_chars"]
    l = counts["latin_alpha_chars"]
    if d > l:
        return "devanagari"
    if l > d * 2:
        return "roman"
    return "mixed"


def expected_script(variant):
    """Required output classification for a given question variant.

    en       -> roman        (English must be written in Latin letters)
    hi       -> devanagari   (Hindi must be written in Devanagari)
    hinglish -> roman or mixed (Romanized Hinglish; code-mixing with
                English words/numbers is normal and acceptable, but
                Devanagari script is not)
    """
    if variant == "en":
        return {"roman"}
    if variant == "hi":
        return {"devanagari"}
    if variant == "hinglish":
        return {"roman", "mixed"}
    raise ValueError(f"unknown variant: {variant!r}")


def is_script_adherent(text, variant):
    """True if classify(text) is in expected_script(variant), and text is
    non-empty."""
    label = classify(text)
    if label == "empty":
        return False
    return label in expected_script(variant)


# ---------------------------------------------------------------------------
# Unit tests -- run directly: python script_check.py
# ---------------------------------------------------------------------------
def _run_tests():
    failures = []

    def check(desc, actual, expected):
        if actual != expected:
            failures.append(f"FAIL: {desc}: got {actual!r}, expected {expected!r}")
        else:
            print(f"ok:   {desc}: {actual!r}")

    # --- count_scripts ---
    c = count_scripts("Mumbai")
    check("count_scripts('Mumbai') devanagari", c["devanagari_chars"], 0)
    check("count_scripts('Mumbai') latin", c["latin_alpha_chars"], 6)
    check("count_scripts('Mumbai') other", c["other_chars"], 0)

    c = count_scripts("मुंबई")
    check("count_scripts('मुंबई') devanagari", c["devanagari_chars"], 5)
    check("count_scripts('मुंबई') latin", c["latin_alpha_chars"], 0)

    c = count_scripts("Mumbai 100 °C!")
    check("count_scripts mixed other>0", c["other_chars"] > 0, True)

    c = count_scripts("")
    check("count_scripts('') all zero", (c["devanagari_chars"], c["latin_alpha_chars"], c["other_chars"]), (0, 0, 0))

    # --- classify: empty ---
    check("classify('') -> empty", classify(""), "empty")
    check("classify('   ') -> empty", classify("   "), "empty")
    check("classify(None) -> empty", classify(None), "empty")

    # --- classify: devanagari ---
    check("classify(pure Hindi) -> devanagari",
          classify("महाराष्ट्र की राजधानी मुंबई है।"), "devanagari")
    check("classify(Hindi with some Latin) -> devanagari",
          classify("महाराष्ट्र की राजधानी Mumbai है और भारत का प्रमुख शहर है।"), "devanagari")

    # --- classify: roman ---
    check("classify(pure English) -> roman",
          classify("Mumbai is the capital of Maharashtra."), "roman")
    check("classify(pure Hinglish, no Devanagari) -> roman",
          classify("Maharashtra ki rajdhani Mumbai hai."), "roman")
    check("classify(Roman with a couple Devanagari chars) -> roman",
          classify("Maharashtra ki rajdhani Mumbai hai, matlab राजधानी."), "roman")

    # --- classify: mixed ---
    # devanagari_chars <= latin_alpha_chars <= devanagari_chars*2
    check("classify(balanced code-mix) -> mixed",
          classify("Mumbai matlab राजधानी शहर है city"), "mixed")

    # exact boundary: latin == devanagari*2 -> NOT roman (needs strictly >), falls to mixed
    # 3 latin chars, "राजधानी" area with e.g. 2 devanagari would be boundary;
    # build a precise boundary case programmatically instead of guessing text:
    dev = "राजधानी"  # 7 devanagari chars (approx, doesn't matter for the boundary logic below)
    dcount = count_scripts(dev)["devanagari_chars"]
    boundary_latin = "a" * (dcount * 2)  # latin_alpha_chars == devanagari_chars * 2 exactly
    boundary_text = dev + boundary_latin
    bc = count_scripts(boundary_text)
    check("boundary case has latin == devanagari*2", bc["latin_alpha_chars"], dcount * 2)
    check("classify(latin == devanagari*2) -> mixed (not roman, needs strict >)",
          classify(boundary_text), "mixed")

    over_boundary_text = dev + "a" * (dcount * 2 + 1)  # one more latin char -> roman
    check("classify(latin == devanagari*2 + 1) -> roman",
          classify(over_boundary_text), "roman")

    # --- expected_script ---
    check("expected_script('en')", expected_script("en"), {"roman"})
    check("expected_script('hi')", expected_script("hi"), {"devanagari"})
    check("expected_script('hinglish')", expected_script("hinglish"), {"roman", "mixed"})
    try:
        expected_script("klingon")
        failures.append("FAIL: expected_script('klingon') should have raised ValueError")
    except ValueError:
        print("ok:   expected_script('klingon') raises ValueError")

    # --- is_script_adherent ---
    check("is_script_adherent(English text, 'en')",
          is_script_adherent("Mumbai is the capital.", "en"), True)
    check("is_script_adherent(Hindi text, 'en') -> False",
          is_script_adherent("मुंबई राजधानी है।", "en"), False)
    check("is_script_adherent(Hindi text, 'hi')",
          is_script_adherent("मुंबई राजधानी है।", "hi"), True)
    check("is_script_adherent(Roman Hinglish, 'hinglish')",
          is_script_adherent("Maharashtra ki rajdhani Mumbai hai.", "hinglish"), True)
    check("is_script_adherent(Devanagari answer to hinglish prompt, 'hinglish') -> False",
          is_script_adherent("महाराष्ट्र की राजधानी मुंबई है।", "hinglish"), False)
    check("is_script_adherent(empty, any variant) -> False",
          is_script_adherent("", "hinglish"), False)
    check("is_script_adherent(empty, 'hi') -> False",
          is_script_adherent("   ", "hi"), False)

    # --- real contamination example from results.json (the actual bug) ---
    contaminated = "हमारे सौर मंडल में 8 ग्रह हैं: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, यूरेनस और नेपच्यून।"
    check("real contaminated hinglish answer classifies as devanagari",
          classify(contaminated), "devanagari")
    check("real contaminated hinglish answer is NOT script_adherent for 'hinglish'",
          is_script_adherent(contaminated, "hinglish"), False)

    good_hinglish = "Hamare saur mandal mein aath grah hain."
    check("a real Roman hinglish answer IS script_adherent",
          is_script_adherent(good_hinglish, "hinglish"), True)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(" ", f)
        raise SystemExit(1)
    else:
        print("All script_check.py unit tests passed.")


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    _run_tests()
