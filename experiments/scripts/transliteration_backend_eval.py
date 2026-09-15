"""Milestone 2.1: evaluate transliteration backends on 30 hand-picked cases.

Two candidates named in BUILD_PLAN.md: indic_transliteration_py and IndicXlit
(ai4bharat-transliteration). IndicXlit could not be evaluated: its dependency
fairseq fails to build on Python 3.14 (FileNotFoundError on fairseq/version.txt
during metadata generation, unrelated to network or pip config -- fairseq is
unmaintained and this is a known break on newer Python). This is itself a
valid disqualifying finding for a library pick, not a setup problem to solve
around, so IndicXlit is scored as "could not install" rather than skipped
silently.

Cases are romanized spellings as people actually type them (casual Hinglish),
not strict ITRANS with diacritics -- that's the realistic input shape for a
"did the model answer in the right words" check. A case "passes" if forward
transliteration (roman -> native script) round-trips to something a human
would call equivalent to the target. Judged by eye, this file records the
target and the actual output side by side rather than asserting equality,
because casual-spelling failures are exactly the finding worth keeping.
"""

from __future__ import annotations

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# (casual romanized input, target native-script spelling, native script)
CASES: list[tuple[str, str, str]] = [
    ("namaskara", "नमस्कार", sanscript.DEVANAGARI),
    ("namaskar", "नमस्कार", sanscript.DEVANAGARI),
    ("dhanyavad", "धन्यवाद", sanscript.DEVANAGARI),
    ("shukriya", "शुक्रिया", sanscript.DEVANAGARI),
    ("paani", "पानी", sanscript.DEVANAGARI),
    ("pani", "पानी", sanscript.DEVANAGARI),
    ("kaise", "कैसे", sanscript.DEVANAGARI),
    ("kaisa", "कैसा", sanscript.DEVANAGARI),
    ("accha", "अच्छा", sanscript.DEVANAGARI),
    ("theek hai", "ठीक है", sanscript.DEVANAGARI),
    ("mumbai", "मुंबई", sanscript.DEVANAGARI),
    ("dilli", "दिल्ली", sanscript.DEVANAGARI),
    ("bharat", "भारत", sanscript.DEVANAGARI),
    ("pyaar", "प्यार", sanscript.DEVANAGARI),
    ("zindagi", "ज़िंदगी", sanscript.DEVANAGARI),
    ("kya haal hai", "क्या हाल है", sanscript.DEVANAGARI),
    ("mera naam", "मेरा नाम", sanscript.DEVANAGARI),
    ("aap kaise hain", "आप कैसे हैं", sanscript.DEVANAGARI),
    ("nahi pata", "नहीं पता", sanscript.DEVANAGARI),
    ("bahut accha", "बहुत अच्छा", sanscript.DEVANAGARI),
    ("namaskara", "ನಮಸ್ಕಾರ", sanscript.KANNADA),
    ("dhanyavaada", "ಧನ್ಯವಾದ", sanscript.KANNADA),
    ("hege iddira", "ಹೇಗೆ ಇದ್ದೀರಾ", sanscript.KANNADA),
    ("bengaluru", "ಬೆಂಗಳೂರು", sanscript.KANNADA),
    ("chennai", "சென்னை", sanscript.TAMIL),
    ("vanakkam", "வணக்கம்", sanscript.TAMIL),
    ("nandri", "நன்றி", sanscript.TAMIL),
    ("eppadi irukkinga", "எப்படி இருக்கீங்க", sanscript.TAMIL),
    ("hyderabad", "హైదరాబాద్", sanscript.TELUGU),
    ("dhanyavaadamulu", "ధన్యవాదములు", sanscript.TELUGU),
]

INDICXLIT_INSTALL_ERROR = (
    "ai4bharat-transliteration requires fairseq, which fails at "
    "metadata-generation with FileNotFoundError: fairseq/version.txt on "
    "Python 3.14, with or without --no-build-isolation. Not installable "
    "in this environment; excluded from scoring on that basis alone."
)


def _strip_trailing_halant(text: str) -> str:
    """Drop a trailing virama (schwa-deletion marker) from each word.

    ITRANS forward transliteration renders a final consonant literally,
    so "namaskar" ends in a bare consonant + halant (्) where a native
    speaker's spelling drops it via implicit schwa deletion. This is
    noise on the comparison, not a real content difference -- strip it
    before scoring so genuine vowel-length errors aren't hidden behind
    trailing-halant mismatches, and aren't masked by it either.
    """
    return " ".join(w.rstrip("्") for w in text.split(" "))


def run_indic_transliteration() -> list[tuple[str, str, str, bool, bool]]:
    rows = []
    for roman, target, script in CASES:
        got = transliterate(roman, sanscript.ITRANS, script)
        exact = got == target
        normalized = _strip_trailing_halant(got) == _strip_trailing_halant(target)
        rows.append((roman, target, got, exact, normalized))
    return rows


def main() -> None:
    print(f"indic_transliteration: {len(CASES)} cases\n")
    rows = run_indic_transliteration()
    exact_passed = sum(1 for *_x, exact, _n in rows if exact)
    norm_passed = sum(1 for *_x, _e, norm in rows if norm)
    for roman, target, got, exact, norm in rows:
        mark = "OK  " if exact else ("HALANT" if norm else "MISS")
        print(f"{mark:6} {roman!r:20} target={target!r:15} got={got!r}")
    print(f"\nindic_transliteration: {exact_passed}/{len(CASES)} exact match, "
          f"{norm_passed}/{len(CASES)} after stripping trailing halant")
    print(f"\nIndicXlit: not scored. {INDICXLIT_INSTALL_ERROR}")


if __name__ == "__main__":
    main()
