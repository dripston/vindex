"""Deterministic writing-system (script) classifier.

Counts characters per Unicode block to decide which script a string is
written in, for example to verify that a "Romanized Hinglish" answer is
in Latin letters rather than Devanagari. No model or LLM is involved.

Recognizes 9 Indic scripts plus Latin; text in other scripts (e.g.
Cyrillic, CJK) counts as "other" and typically classifies as
``"mixed"``. Unicode blocks:

- Devanagari: U+0900-U+097F
- Bengali: U+0980-U+09FF
- Gurmukhi: U+0A00-U+0A7F
- Gujarati: U+0A80-U+0AFF
- Odia: U+0B00-U+0B7F (Unicode block name "Oriya")
- Tamil: U+0B80-U+0BFF
- Telugu: U+0C00-U+0C7F
- Kannada: U+0C80-U+0CFF
- Malayalam: U+0D00-U+0D7F

Limitation: the danda and double danda (।॥, U+0964-U+0965) live in the
Devanagari block but are used as sentence punctuation by several other
scripts, so non-Devanagari text can contain a few ``devanagari_chars``.
This does not affect :func:`classify` on real sentences.
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
    """Return True if ``text`` consists only of danda punctuation.

    A string such as ``"।"`` classifies as ``"devanagari"`` even though it
    contains no letters. This lets callers (e.g.
    :func:`vindex.script_adherence`) treat it as having no script signal.

    Args:
        text: Text to check. Empty or whitespace-only text returns False.
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
    """Count characters by script.

    Args:
        text: Text to count. ``None`` is treated as empty.

    Returns:
        A dict with one ``<script>_chars`` key per script in
        :data:`SCRIPT_RANGES`, plus ``latin_alpha_chars`` (ASCII A-Z, a-z)
        and ``other_chars`` (digits, punctuation, whitespace, symbols, and
        unrecognized scripts).
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

    Let ``d`` be the character count of the most frequent Indic script
    (ties go to the earlier script in :data:`SCRIPT_RANGES`, Devanagari
    first) and ``l`` the Latin letter count.

    Args:
        text: Text to classify.

    Returns:
        ``"empty"`` for ``None`` or whitespace-only text; the script name
        (e.g. ``"devanagari"``, ``"tamil"``) if ``d > l``; ``"roman"`` if
        ``l > 2 * d``; otherwise ``"mixed"``.
    """
    if text is None or text.strip() == "":
        return "empty"
    counts = count_scripts(text)
    dominant_script, dominant_n = max(
        ((name, counts[f"{name}_chars"]) for name in SCRIPT_RANGES), key=lambda kv: kv[1]
    )
    l = counts["latin_alpha_chars"]  # noqa: E741
    if dominant_n > l:
        return dominant_script
    if l > dominant_n * 2:
        return "roman"
    return "mixed"


def expected_script(variant: str) -> set[str]:
    """Return the acceptable classifications for a question variant.

    - ``"en"``: ``{"roman"}``
    - ``"hi"``: ``{"devanagari"}``
    - ``"hinglish"``: ``{"roman", "mixed"}`` (code-mixing is acceptable;
      Devanagari is not)

    Raises:
        ValueError: If ``variant`` is not one of the above.
    """
    if variant == "en":
        return {"roman"}
    if variant == "hi":
        return {"devanagari"}
    if variant == "hinglish":
        return {"roman", "mixed"}
    raise ValueError(f"unknown variant: {variant!r}")


def is_script_adherent(text: str | None, variant: str) -> bool:
    """Return True if non-empty ``text`` classifies as expected for ``variant``.

    Raises:
        ValueError: If ``variant`` is unknown (see :func:`expected_script`).
    """
    label = classify(text)
    if label == "empty":
        return False
    return label in expected_script(variant)
