"""
Deterministic script (writing-system) classifier. No LLM involved.

Used to verify that a model's response is actually written in the script
the experiment asked for -- e.g. that a "Romanized Hinglish" answer is
actually in Roman letters, not Devanagari.

Ported from the repo root's script_check.py (Milestone 1.1); the
Devanagari-vs-Latin character-counting logic there is unchanged here --
it's tested and validated on 90 real responses. Milestone 1.2 generalizes
that same ratio logic ("dominant script wins if it outnumbers Latin;
Latin wins if it outnumbers the dominant script more than 2:1; otherwise
mixed") from a Devanagari-only special case to any of the 8 additional
Indic scripts below, keeping the original thresholds and tie-breaking.

Unicode block ranges (verified against unicode.org/Wikipedia's Unicode
block pages):
  Devanagari : U+0900-U+097F
  Gurmukhi   : U+0A00-U+0A7F
  Gujarati   : U+0A80-U+0AFF
  Odia       : U+0B00-U+0B7F  (Unicode block name: "Oriya")
  Tamil      : U+0B80-U+0BFF
  Telugu     : U+0C00-U+0C7F
  Kannada    : U+0C80-U+0CFF
  Malayalam  : U+0D00-U+0D7F
  Bengali    : U+0980-U+09FF

KNOWN LIMITATION: the danda and double danda (।, ॥ -- U+0964, U+0965) are
placed in the Unicode Devanagari block, but are used as sentence-ending
punctuation across most of the scripts above (Bengali, Odia, Gurmukhi,
Gujarati, etc. generally don't have their own script-specific full stop
and reuse these). A pure-Bengali/Odia/Gurmukhi/etc. sentence can
therefore show 1-2 stray devanagari_chars from its own punctuation. This
does not change classify()'s output in practice -- the dominant script's
character count is far larger than a couple of punctuation marks in any
real sentence -- but it means devanagari_chars is not a perfectly pure
signal of "this text contains Devanagari letters." Documented here
rather than special-cased, to keep the ported character-counting logic
exactly as validated in Milestone 1.1.
"""

from __future__ import annotations

import re

SCRIPT_RANGES: dict[str, str] = {
    "devanagari": r"ऀ-ॿ",
    "gurmukhi": r"਀-੿",
    "gujarati": r"઀-૿",
    "odia": r"଀-୿",
    "tamil": r"஀-௿",
    "telugu": r"ఀ-౿",
    "kannada": r"ಀ-೿",
    "malayalam": r"ഀ-ൿ",
    "bengali": r"ঀ-৿",
}

SCRIPT_RES: dict[str, re.Pattern[str]] = {
    name: re.compile(f"[{block}]") for name, block in SCRIPT_RANGES.items()
}

DEVANAGARI_RE = SCRIPT_RES["devanagari"]  # kept for backward compatibility
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")
_DANDA_RE = re.compile(r"[।॥]")


def is_only_danda_punctuation(text: str) -> bool:
    """True if text's only Devanagari-block character(s) are the danda
    or double danda (।/॥ -- sentence-ending punctuation, not a letter),
    with no other script/Latin content either.

    A response that is purely "।" has devanagari_chars == 1 and nothing
    else, so classify() confidently returns "devanagari" (dominant_n=1 >
    latin_alpha_chars=0) even though there is no actual Devanagari (or
    any) letter in it -- see script.py's module docstring on the danda's
    shared cross-script punctuation role. This helper lets a caller
    (metric.py's script_adherence) distinguish that degenerate case from
    a real Devanagari response, without changing classify()'s or
    count_scripts()'s pinned behavior for real text.
    """
    text = text or ""
    if text.strip() == "":
        return False
    counts = count_scripts(text)
    if counts["latin_alpha_chars"] > 0:
        return False
    if any(counts[f"{name}_chars"] > 0 for name in SCRIPT_RANGES if name != "devanagari"):
        return False
    devanagari_only = SCRIPT_RES["devanagari"].findall(text)
    non_danda_devanagari = [c for c in devanagari_only if not _DANDA_RE.match(c)]
    return counts["devanagari_chars"] > 0 and len(non_danda_devanagari) == 0


def count_scripts(text: str | None) -> dict[str, int]:
    """Count characters by script bucket.

    <script>_chars    : one key per script in SCRIPT_RANGES (devanagari,
                         gurmukhi, gujarati, odia, tamil, telugu, kannada,
                         malayalam, bengali), each counting characters in
                         that script's Unicode block.
    latin_alpha_chars : ASCII alphabetic (A-Z, a-z)
    other_chars       : everything else (digits, punctuation, whitespace,
                         symbols, any script not listed above)
    """
    text = text or ""
    counts = {f"{name}_chars": len(pattern.findall(text)) for name, pattern in SCRIPT_RES.items()}
    latin_alpha_chars = len(LATIN_ALPHA_RE.findall(text))
    other_chars = len(text) - sum(counts.values()) - latin_alpha_chars
    counts["latin_alpha_chars"] = latin_alpha_chars
    counts["other_chars"] = other_chars
    return counts


def classify(text: str | None) -> str:
    """Classify a string's dominant script.

    empty      : blank or whitespace only
    <script>   : one non-Latin script (e.g. "devanagari", "tamil") has more
                 characters than Latin, AND is the largest non-Latin
                 script present (ties broken by SCRIPT_RANGES iteration
                 order, devanagari first, matching the original
                 Devanagari-only behavior for pure-Devanagari text)
    roman      : latin_alpha_chars > dominant_script_chars * 2
    mixed      : otherwise (includes ties, and cases where
                 dominant_script_chars <= latin_alpha_chars <=
                 dominant_script_chars * 2)

    For Devanagari-vs-Latin-only text, this reproduces the original
    script_check.py behavior exactly: dominant_script_chars is
    devanagari_chars, so "d > l" / "l > d * 2" are unchanged.
    """
    if text is None or text.strip() == "":
        return "empty"
    counts = count_scripts(text)
    dominant_script, dominant_n = max(
        ((name, counts[f"{name}_chars"]) for name in SCRIPT_RANGES), key=lambda kv: kv[1]
    )
    l = counts["latin_alpha_chars"]  # noqa: E741 -- ported unchanged from script_check.py
    if dominant_n > l:
        return dominant_script
    if l > dominant_n * 2:
        return "roman"
    return "mixed"


def expected_script(variant: str) -> set[str]:
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


def is_script_adherent(text: str | None, variant: str) -> bool:
    """True if classify(text) is in expected_script(variant), and text is
    non-empty."""
    label = classify(text)
    if label == "empty":
        return False
    return label in expected_script(variant)
