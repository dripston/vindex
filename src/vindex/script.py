"""
Deterministic script (writing-system) classifier. No LLM involved.

Used to verify that a model's response is actually written in the script
the experiment asked for -- e.g. that a "Romanized Hinglish" answer is
actually in Roman letters, not Devanagari.

Ported from the repo root's script_check.py (Milestone 1.1). The
character-counting logic below is unchanged from that file -- it's
tested and validated on 90 real responses. Only the API around it
(type hints, module location) was cleaned up.
"""

from __future__ import annotations

import re

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")


def count_scripts(text: str | None) -> dict[str, int]:
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


def classify(text: str | None) -> str:
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
    l = counts["latin_alpha_chars"]  # noqa: E741 -- ported unchanged from script_check.py
    if d > l:
        return "devanagari"
    if l > d * 2:
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
