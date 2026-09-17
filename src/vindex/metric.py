"""
script_adherence: did response come back in script/language prompt used?

Combines script.py (script classification) and language.py (v0 Hindi
function-word heuristic) into one public MetricResult-returning metric.

Prompt bucket (from classify(prompt)):
  native-script : prompt's dominant label is an Indic script (devanagari,
                  tamil, kannada, ...) -- not roman/mixed/empty.
  romanized     : classify(prompt) == "roman" AND prompt has no Hindi
                  function words (language.looks_like_hinglish is False).
  code-mixed    : classify(prompt) == "mixed", OR classify(prompt) ==
                  "roman" AND prompt has Hindi function words (Romanized
                  Hindi/Hinglish prompt).

Pass rule (from BUILD_PLAN 1.4), by prompt bucket:
  native-script prompt -> response native (same script) or mixed passes
  romanized prompt     -> response roman passes
  code-mixed prompt    -> response roman or mixed passes

Labels:
  empty             : prompt or response is empty/whitespace-only
  matched           : response passes its prompt bucket's rule, and (for
                       native-script prompts) is the *same* Indic script
                       as the prompt
  mixed             : response classify() == "mixed" and prompt bucket
                       accepts mixed -- counted as a pass, labeled
                       distinctly from a same-script "matched" since it's
                       code-mixed, not a clean script match
  script_mismatch   : response modality is wrong for the prompt bucket
                       (e.g. Devanagari prompt answered in Roman, or Roman
                       prompt answered in Devanagari/a different Indic
                       script)
  language_mismatch : romanized or code-mixed prompt has Hindi function
                       words (i.e. is Romanized Hindi/Hinglish, not plain
                       English), response is "roman" (passes the script
                       check) but contains zero Hindi function words --
                       right script, wrong language. v0 heuristic, see
                       language.py's limitations; only fires when the
                       language.py signal is available (Latin-script
                       responses to romanized/code-mixed prompts).
  no_script_signal  : response classify() == "mixed" only because it has
                       NO alphabetic characters in any recognized script
                       (e.g. emoji-only, CJK/Cyrillic-only, digits-only,
                       punctuation-only) -- classify() has nothing to rank
                       (dominant_n == 0 and latin_alpha_chars == 0), so it
                       falls through to "mixed" by construction, not
                       because the response is genuinely code-mixed
                       content. Without this distinction, a broken/
                       off-topic/wrong-language response with no real
                       script content would be silently scored passed=True
                       under the same "mixed counts as a pass" rule that
                       exists for real Hinglish code-mixing -- this label
                       exists specifically so that rule does not also
                       cover responses with no script content to judge at
                       all. Always passed=False.
"""

from __future__ import annotations

from vindex.language import looks_like_hinglish
from vindex.result import MetricResult
from vindex.script import SCRIPT_RANGES, classify, count_scripts, is_only_danda_punctuation

_INDIC_SCRIPTS = frozenset(SCRIPT_RANGES)


def _has_no_script_signal(text: str) -> bool:
    """True if text has zero alphabetic characters in any recognized
    script -- classify() then labels it "mixed" only because it has
    nothing to rank (see this module's "no_script_signal" label docs),
    not because it is genuinely code-mixed content. Also true for a
    response that is purely danda punctuation (see
    script.is_only_danda_punctuation) -- classify() confidently returns
    "devanagari" for "।" alone, but a lone sentence-ending mark is not a
    real Devanagari response either."""
    if is_only_danda_punctuation(text):
        return True
    counts = count_scripts(text)
    dominant_n = max((counts[f"{name}_chars"] for name in SCRIPT_RANGES), default=0)
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


def script_adherence(prompt: str | None, response: str | None) -> MetricResult:
    """Did response come back in the script/language the prompt used?

    No reference answer needed -- this checks a property of the response
    itself (and its relation to the prompt), not response correctness.
    """
    prompt = prompt or ""
    response = response or ""

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
        if bucket == "code-mixed" and response_label == "roman" and looks_like_hinglish(prompt):
            if not looks_like_hinglish(response):
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
