"""script_adherence: did the response come back in the prompt's script?

Combines script classification (:mod:`vindex.script`) and the Hinglish
heuristic (:mod:`vindex.language`) into a reference-free metric. It
checks script and language modality only, never correctness: a single
Devanagari letter answering a Devanagari question passes. Pair it with
:func:`vindex.indic_judge` or :func:`vindex.calibrated_similarity` to
check content.

The prompt is assigned a bucket:

- ``native-script``: the prompt's dominant script is Indic.
- ``romanized``: Latin script with no Hindi function words.
- ``code-mixed``: mixed scripts, or Latin script containing Hindi
  function words (Hinglish).

Pass rules by bucket:

- native-script: the response is in the same script, or mixed.
- romanized: the response is Roman.
- code-mixed: the response is Roman or mixed.

A Hinglish prompt answered in pure Devanagari fails with
``script_mismatch``. This is deliberate: it is the failure the metric is
designed to catch.

Labels:

- ``empty``: prompt or response is empty or whitespace-only.
- ``matched``: the response passes and, for native-script prompts, uses
  the same script.
- ``mixed``: the response is code-mixed and the bucket accepts it
  (counted as a pass).
- ``script_mismatch``: the response's script is wrong for the bucket.
- ``language_mismatch``: only with ``strict_language_check=True``. A
  Hinglish prompt is answered in Roman script with no Hindi function
  words (right script, wrong language).
- ``no_script_signal``: the prompt or response has no letters in any
  recognized script (emoji, digits, punctuation only, or an unsupported
  script such as Cyrillic or CJK), so no verdict can be made. Always
  fails.
"""

from __future__ import annotations

import unicodedata

from vindex.language import looks_like_hinglish
from vindex.result import MetricResult, coerce_text
from vindex.script import SCRIPT_RANGES, classify, count_scripts, is_only_danda_punctuation

_INDIC_SCRIPTS = frozenset(SCRIPT_RANGES)


def _non_letter_script_count(text: str, script_chars: int, script_pattern: str) -> int:
    """Return ``script_chars`` minus the script's digits and danda marks.

    Indic Unicode blocks include decimal digits (category ``Nd``) and the
    shared danda/double danda (।॥); neither is a letter.
    """
    import re

    from vindex.script import _DANDA_RE

    non_letters_in_script = sum(
        1
        for c in re.findall(f"[{script_pattern}]", text)
        if unicodedata.category(c) == "Nd" or _DANDA_RE.match(c)
    )
    return script_chars - non_letters_in_script


def _has_no_script_signal(text: str) -> bool:
    """Return True if ``text`` has no letters in any recognized script.

    Indic digits (e.g. ``"१४०"``) and danda punctuation are not counted
    as letters, so responses such as ``"१४०००"`` or ``"१।"`` have no
    script signal even though :func:`vindex.script.classify` assigns them
    a script. Text that mixes real letters with Indic digits is
    unaffected.
    """
    if is_only_danda_punctuation(text):
        return True
    counts = count_scripts(text)
    dominant_n = max(
        (
            _non_letter_script_count(text, counts[f"{name}_chars"], SCRIPT_RANGES[name])
            for name in SCRIPT_RANGES
        ),
        default=0,
    )
    return dominant_n == 0 and counts["latin_alpha_chars"] == 0


def _prompt_bucket(prompt: str) -> str:
    label = classify(prompt)
    if label in _INDIC_SCRIPTS:
        return "native-script"
    if label == "mixed":
        return "code-mixed"
    if label == "roman":
        return "code-mixed" if looks_like_hinglish(prompt) else "romanized"
    return "empty"


def script_adherence(
    prompt: str | None, response: str | None, strict_language_check: bool = False
) -> MetricResult:
    """Check whether the response uses the script/language of the prompt.

    No reference answer is needed. See the module docstring for prompt
    buckets, pass rules, and labels.

    Args:
        prompt: The prompt. ``None`` is treated as empty.
        response: The model's response. ``None`` is treated as empty.
        strict_language_check: If True, a Hinglish prompt answered in
            Roman-script English fails with ``language_mismatch``. Off by
            default because the underlying heuristic has known false
            positives: one incidental function-word match in the prompt
            ("Who directed Se7en?" -> "se") is enough to treat an English
            prompt as Hinglish. When False, such responses are labelled
            ``matched``. Enable it only after verifying it on your data.

    Returns:
        A :class:`vindex.result.MetricResult`. ``detail`` contains
        ``prompt_label``, ``response_label``, and (once the prompt has
        script signal) ``prompt_bucket``.
    """
    prompt = coerce_text(prompt)
    response = coerce_text(response)

    prompt_label = classify(prompt)
    response_label = classify(response)

    if prompt_label == "empty" or response_label == "empty":
        return MetricResult(
            score=0.0,
            passed=False,
            label="empty",
            reason="prompt or response is empty.",
            detail={"prompt_label": prompt_label, "response_label": response_label},
        )

    if _has_no_script_signal(prompt):
        # A prompt with no script signal would otherwise classify as
        # "mixed" and be bucketed as code-mixed (Hinglish).
        return MetricResult(
            score=0.0,
            passed=False,
            label="no_script_signal",
            reason=(
                "prompt has no alphabetic characters in any recognized script "
                "(e.g. emoji, digits, punctuation-only, or an unrecognized script "
                "like Cyrillic/CJK) -- not a genuine code-mixed/Hinglish prompt, so "
                "no script-adherence verdict can be made against it."
            ),
            detail={"prompt_label": prompt_label, "response_label": response_label},
        )

    bucket = _prompt_bucket(prompt)
    detail = {
        "prompt_bucket": bucket,
        "prompt_label": prompt_label,
        "response_label": response_label,
    }

    if _has_no_script_signal(response):
        return MetricResult(
            score=0.0,
            passed=False,
            label="no_script_signal",
            reason=(
                "response has no alphabetic characters in any recognized script "
                "(e.g. emoji, digits, or punctuation only) -- not a genuine "
                "code-mixed response, so it does not count as a script-adherence pass."
            ),
            detail=detail,
        )

    if bucket == "native-script":
        if response_label == prompt_label:
            return MetricResult(
                score=1.0,
                passed=True,
                label="matched",
                reason=f"prompt and response both in {prompt_label}.",
                detail=detail,
            )
        if response_label == "mixed":
            return MetricResult(
                score=1.0,
                passed=True,
                label="mixed",
                reason=f"prompt in {prompt_label}; response is code-mixed.",
                detail=detail,
            )
        return MetricResult(
            score=0.0,
            passed=False,
            label="script_mismatch",
            reason=f"prompt in {prompt_label}; response in {response_label}.",
            detail=detail,
        )

    # romanized or code-mixed: both accept "roman" or "mixed" responses.
    if response_label in ("roman", "mixed"):
        if (
            strict_language_check
            and bucket == "code-mixed"
            and response_label == "roman"
            and looks_like_hinglish(prompt)
        ):
            if not looks_like_hinglish(response):
                # Prompt match count cannot separate a genuine one-word
                # Hinglish signal ("Mumbai kahan hai?") from an incidental
                # match ("Se7en"), which is why this check is opt-in.
                return MetricResult(
                    score=0.0,
                    passed=False,
                    label="language_mismatch",
                    reason="prompt is Romanized Hindi; response is Roman-script English.",
                    detail=detail,
                )
        label = "matched" if response_label == "roman" else "mixed"
        reason = (
            f"prompt is {bucket}; response is {response_label}."
            if label == "mixed"
            else f"prompt is {bucket}; response is Roman script, as expected."
        )
        return MetricResult(score=1.0, passed=True, label=label, reason=reason, detail=detail)

    return MetricResult(
        score=0.0,
        passed=False,
        label="script_mismatch",
        reason=f"prompt is {bucket}; response in {response_label}, not Roman script.",
        detail=detail,
    )
