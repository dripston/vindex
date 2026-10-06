"""indic_judge: an LLM judge built for Indic content.

The other vindex metrics check surface properties (script, normalized
string match, embedding similarity). :func:`indic_judge` answers whether
an answer is actually correct. Semantic correctness without a reference
cannot be checked deterministically, so this is the one metric in the
package that calls an LLM.

How it differs from a generic LLM judge:

- **Hindi rubric.** The rubric (:mod:`vindex.judge_rubric`) is written in
  Hindi rather than an English rubric applied to Hindi content. An
  English-reasoning judge was observed mistranslating समुद्र तल ("sea
  level") as "sea floor" mid-reasoning and scoring a correct answer 0.
- **Script-aware.** The rubric states that Romanized Hindi is not an
  error; script checking is left to :func:`vindex.script_adherence`.
- **Reference-free by default.** A gold reference can mask comprehension
  drift: the judge may reach the right verdict by pattern-matching the
  gold string while its reasoning is already wrong. Reference-based
  mode is available by passing ``gold``.
- **Align-then-judge** (reference-based mode). The answer is first
  diffed against gold (:mod:`vindex.judge_align`); an exact word-level
  match skips the LLM call, and a partial mismatch sends only the
  disagreeing spans.
- **Conservative gate.** Passes only at high judge-reported confidence
  and a top score; anything ambiguous is labelled ``"flagged"``.
- **Determinism.** The default judge runs at temperature 0 and the
  judge model id is recorded in every result's ``detail``.

Self-enhancement bias: never judge a model with itself or a model from
the same family. A best-effort, name-based check adds a
``self_enhancement_bias_warning`` to ``detail`` when you pass
``answering_model_id``; its absence does not mean the pairing is safe.
See :data:`JUDGE_SELECTION_GUIDANCE` for choosing a judge model.

Limitation: on the 62-trace human-agreement benchmark
(:func:`vindex.datasets.load_judge_benchmark`), an English-rubric
baseline agreed with human graders slightly more often than the Hindi
rubric (93.5% vs 90.3%). Most disagreements came from the Hindi rubric
being stricter about answer completeness. Do not assume the Hindi rubric
agrees with humans better than an English one.
"""

from __future__ import annotations

import json
import re

from vindex.judge_align import align
from vindex.judge_model import GroqJudge, JudgeModel
from vindex.judge_rubric import build_reference_based_prompt, build_reference_free_prompt
from vindex.result import MetricResult, coerce_text

JUDGE_SELECTION_GUIDANCE = """
Judge model selection is not a solved default -- pick deliberately:

- openai/gpt-oss-120b (via Groq): the model vindex's rubric was
  developed and validated against. Recommended default for
  Hindi/Hinglish content specifically because it is the combination
  with direct evidence behind it, not because it is assumed to
  generalize to every Indic language.
- Smaller models degrade on Indic content specifically, not just on
  language tasks in general -- vindex's calibration data
  (vindex.calibration) and the HindiWiC finding (Dairkee & Dubossarsky,
  2024, cited in vindex.calibration.MURIL_WARNING) both show
  Indic-language competence is not a simple function of overall model
  capability. Do not assume a smaller/cheaper model "should be fine"
  for Indic judging just because it performs adequately on English
  tasks -- verify against your own labelled data (see
  vindex.datasets.score_judge_benchmark). On the shipped 62-trace
  benchmark, an English rubric agreed with human graders 93.5% of the
  time vs 90.3% for the Hindi rubric, so rubric language alone is not a
  substitute for checking your own data either.
- NEVER use the same model (or a model from the same family/provider
  fine-tune lineage) as both the model being judged and the judge
  itself -- self-enhancement bias is a documented effect, not a
  theoretical risk. indic_judge cannot fully detect this for you; it
  performs only a partial, name-based check.
"""

_MAX_SCORE = 5


def _iter_balanced_objects(text: str) -> list[str]:
    """Return every balanced ``{...}`` span in ``text``, in order.

    Braces inside JSON string literals are ignored. Nested objects are
    returned as part of their enclosing span, not separately.
    """
    candidates = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape = False
        j = start
        while j < len(text):
            c = text[j]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                j += 1
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : j + 1])
                    break
            j += 1
        i = start + 1
    return candidates


def _extract_json(text: str) -> str:
    """Extract the JSON object from a judge's raw text response.

    Strips Markdown code fences, then returns the first balanced
    ``{...}`` span that parses as valid JSON, so braces mentioned in
    surrounding prose are skipped. If no candidate parses, the raw text
    is returned so that ``json.loads`` raises on it.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    for candidate in _iter_balanced_objects(text):
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    return text


def _parse_judge_response(raw: str) -> tuple[float, str, str, str]:
    """Parse a judge's JSON response.

    Args:
        raw: Raw text returned by the judge model.

    Returns:
        ``(score, reasoning, confidence, raw_confidence)``. ``score`` is the
        1-5 rubric score normalized to [0, 1]. ``confidence`` is ``"high"``
        or ``"low"`` and is used for gating; any other value (e.g.
        ``"medium"``) is normalized to ``"low"``. ``raw_confidence`` is the
        value the judge actually sent, for audit trails.

    Raises:
        ValueError: If the response is malformed, including a score that
            is a boolean, not a JSON number, not finite, or outside 1-5.
            Out-of-range scores are rejected rather than clamped. Callers
            treat this as a failed judge call, never a default score.
    """
    data = json.loads(_extract_json(raw))
    raw_score_field = data["score"]
    if isinstance(raw_score_field, bool):
        # bool is a subclass of int; reject it explicitly.
        raise ValueError(f"score {raw_score_field!r} is a boolean, not a number")
    if not isinstance(raw_score_field, (int, float)):
        # Reject numeric strings such as "5".
        raise ValueError(f"score {raw_score_field!r} is not a JSON number")
    score_value = float(raw_score_field)
    try:
        raw_score = int(round(score_value))
    except (OverflowError, ValueError) as exc:
        # json.loads accepts Infinity/-Infinity; int(round(inf)) raises
        # OverflowError, which is re-raised as ValueError here.
        raise ValueError(f"score {score_value!r} is not a finite number") from exc
    if not 1 <= raw_score <= _MAX_SCORE:
        raise ValueError(f"score {raw_score!r} is outside the documented 1-{_MAX_SCORE} range")
    normalized_score = (raw_score - 1) / (_MAX_SCORE - 1)
    reasoning = str(data.get("reasoning", ""))
    raw_confidence = str(data.get("confidence", "low"))
    confidence = raw_confidence.lower()
    if confidence not in ("high", "low"):
        confidence = "low"
    return normalized_score, reasoning, confidence, raw_confidence


def _family(model_id: str) -> str:
    """Return a coarse model-family key for the self-enhancement check.

    Strips only a trailing ``-<digits>[b|m]`` size token, so
    ``"openai/gpt-oss-120b"`` and ``"openai/gpt-oss-20b"`` share the family
    ``"openai/gpt-oss"``. This is deliberately narrow: provider prefixes
    and any suffix after the size token (``-instruct``, ``-turbo``,
    ``-mini``) defeat it, so pairs such as ``gpt-4o`` / ``gpt-4o-mini`` or
    ``llama-3.1-70b-instruct`` / ``llama-3.1-8b-instruct`` are not
    detected. The absence of a warning never means a pairing is safe.
    """
    parts = model_id.split("-")
    if len(parts) > 1 and parts[-1].rstrip("bm").isdigit():
        return "-".join(parts[:-1])
    return model_id


def _warn_if_same_family(judge_model_id: str, answering_model_id: str | None) -> str | None:
    if not answering_model_id:
        return None
    if _family(judge_model_id) == _family(answering_model_id):
        return (
            f"judge model {judge_model_id!r} appears to be from the same "
            f"family as the model being judged {answering_model_id!r} -- "
            "self-enhancement bias risk. Treat this "
            "result as suspect; use a judge from a different model "
            "family/provider."
        )
    return None


def indic_judge(
    question: str | None,
    answer: str | None,
    gold: str | None = None,
    judge: JudgeModel | None = None,
    answering_model_id: str | None = None,
) -> MetricResult:
    """Judge whether an answer correctly and accurately answers a question.

    Reference-free by default. Pass ``gold`` to use reference-based mode,
    which aligns the answer against gold first: an exact word-level match
    returns ``"matched"`` without calling the LLM (and without requiring
    an API key).

    The gate is conservative: ``passed`` is True only when the judge
    reports ``"high"`` confidence and a normalized score >= 0.8. Raw 1-5
    scores normalize as ``(raw - 1) / 4``, so only a raw 5 passes; a raw 4
    (0.75) is ``"flagged"`` even at high confidence.

    Args:
        question: The question asked. ``None`` is treated as empty.
        answer: The answer to judge. ``None`` is treated as empty.
        gold: Optional reference answer. Non-string values (e.g. a pandas
            NaN) are treated as ``None``.
        judge: Judge model implementing
            :class:`vindex.judge_model.JudgeModel`. Defaults to
            :class:`vindex.judge_model.GroqJudge`, which requires
            ``GROQ_API_KEY``.
        answering_model_id: Identifier of the model that produced
            ``answer``. Enables a same-family self-enhancement-bias
            warning in ``detail``.

    Returns:
        A :class:`vindex.result.MetricResult` with label ``"matched"``,
        ``"flagged"``, ``"empty"``, or ``"judge_error"`` (unparseable judge
        response). ``detail`` includes ``mode``, ``judge_model_id``, and,
        when the judge was called, ``confidence``, ``judge_reasoning``, and
        ``raw_confidence`` if it differed from the normalized value.

    Raises:
        ValueError: If the default judge is needed and ``GROQ_API_KEY``
            is not set.
    """
    question = coerce_text(question)
    answer = coerce_text(answer)

    if question.strip() == "" or answer.strip() == "":
        return MetricResult(
            score=0.0,
            passed=False,
            label="empty",
            reason="question or answer is empty.",
            detail={"question": question, "answer": answer},
        )

    if gold is not None and not isinstance(gold, str):
        # e.g. a pandas NaN from a dataframe cell: treat as reference-free.
        gold = None

    # Construct the default judge (which validates GROQ_API_KEY) only
    # once an LLM call is actually needed, so an exact gold match works
    # without a key. A JudgeModel's model_id is a plain attribute.
    caller_supplied_judge = judge
    default_model_id = (
        caller_supplied_judge.model_id
        if caller_supplied_judge is not None
        else GroqJudge.DEFAULT_MODEL_ID
    )
    family_warning = _warn_if_same_family(default_model_id, answering_model_id)

    if gold is not None and gold.strip() != "":
        alignment = align(answer, gold)
        if alignment.aligned:
            detail = {
                "mode": "reference_based",
                "judge_model_id": default_model_id,
                "aligned": True,
                "reasoning": "answer aligns exactly with gold at the word level; no LLM call made.",
            }
            if family_warning:
                detail["self_enhancement_bias_warning"] = family_warning
            return MetricResult(
                score=1.0,
                passed=True,
                label="matched",
                reason="answer aligns exactly with gold; no judge call needed.",
                detail=detail,
            )
        prompt = build_reference_based_prompt(
            question, alignment.mismatched_response, alignment.mismatched_gold
        )
        mode = "reference_based"
    else:
        prompt = build_reference_free_prompt(question, answer)
        mode = "reference_free"

    judge = caller_supplied_judge or GroqJudge()
    raw_response = judge.call(prompt)
    try:
        score, reasoning, confidence, raw_confidence = _parse_judge_response(raw_response)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return MetricResult(
            score=0.0,
            passed=False,
            label="judge_error",
            reason=f"judge response could not be parsed: {exc}",
            detail={"mode": mode, "judge_model_id": judge.model_id, "raw_response": raw_response},
        )

    passed = confidence == "high" and score >= 0.8
    label = "matched" if passed else "flagged"
    reason = (
        f"judge score {score:.2f} at {confidence} confidence "
        f"({'passes' if passed else 'flagged, not confirmed as passing'} "
        "the conservative threshold: high confidence and score >= 0.8)."
    )

    detail = {
        "mode": mode,
        "judge_model_id": judge.model_id,
        "confidence": confidence,
        "judge_reasoning": reasoning,
    }
    if raw_confidence.lower() != confidence:
        # Report what the judge actually sent (e.g. "medium") alongside
        # the normalized value used for gating.
        detail["raw_confidence"] = raw_confidence
    if family_warning:
        detail["self_enhancement_bias_warning"] = family_warning

    return MetricResult(score=score, passed=passed, label=label, reason=reason, detail=detail)

