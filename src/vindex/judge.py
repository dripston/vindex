"""
indic_judge: an LLM judge built for Indic, not an English judge pointed
at Hindi (Milestone 5).

WHY THIS METRIC EXISTS

script_adherence (Milestone 1), script_normalized_match's pieces
(Milestone 2), and calibrated_similarity (Milestone 3) all check
surface properties: right script, strings match after normalization,
embeddings close enough. None of them answers "is the answer
correct" -- that is the question human linguists get paid to answer,
and it is the gap this whole project is aimed at closing. A library
without an answer to that question does not close that gap.

THIS METRIC IS AN LLM JUDGE. That is correct and unavoidable: semantic
correctness with no reference answer cannot be done deterministically.
Every other metric in this package is deterministic on purpose; this
one is not, and says so plainly rather than pretending otherwise.

WHAT'S DIFFERENT FROM A GENERIC LLM JUDGE

- The rubric (vindex.judge_rubric) is written IN Hindi, not an English
  rubric pointed at Hindi content -- this project's own Phase 0 data
  found an English-reasoning judge silently mistranslating समुद्र तल
  ("sea level") as "sea floor" mid-thought and scoring a correct
  answer 0.0. See judge_rubric.py's module docstring for the full
  mechanism, not just the symptom.
- Script-aware: the rubric states explicitly that Romanized Hindi is
  not an error, so the judge doesn't re-introduce the same
  script-vs-language confusion script_adherence already handles
  separately (Milestone 5.2).
- Reference-free by DEFAULT (Milestone 5.3): this project's own data
  showed a gold reference masks comprehension drift -- the judge
  reached the right verdict by pattern-matching the gold string while
  its actual reasoning was already wrong (see judge_rubric.py). Both
  modes are supported; reference-free is the default specifically
  because trusting a reference-based judge's apparent correctness is
  the mistake this project's own evidence warns against.
- Align-then-judge (Milestone 5.4, reference-based mode only): diffs
  response against gold first (vindex.judge_align); an exact word-level
  match skips the LLM call entirely, and a partial mismatch sends only
  the disagreeing spans, not the full text, bounding cost/latency/
  variance to what actually disagrees.
- Conservative default (Milestone 5.5): passes only at high judge-
  reported confidence AND a high score. Ambiguous cases (low
  confidence, or a middling score) are flagged, not silently passed --
  a false alarm costs a human reviewer a few seconds; a silent pass on
  a wrong answer in a banking flow costs a lot more.
- Deterministic discipline (Milestone 5.6): temperature 0
  (vindex.judge_model's GroqJudge hard-codes this), and the judge
  model's exact identifier is recorded in every result's detail
  payload -- this project's own data showed gpt-oss-20b reworded
  output roughly a third of the time even at temperature 0, so a
  judge whose backing model silently updates makes every historical
  score incomparable.

SELF-ENHANCEMENT BIAS (Milestone 5.7): never judge a model with
itself, or a model from the same family, without treating the result
as suspect -- a model is measurably biased toward rating its own
output favorably. indic_judge does not (cannot, in general) detect
this automatically; judge_model_id and the model-under-test's own
identifier, if you have it, are both in every result's detail payload
specifically so you can check this yourself, and raise a warning if
you pass judge and answering models that look like the same family
(see _warn_if_same_family below).

JUDGE SELECTION GUIDANCE (Milestone 5.7): documented in
JUDGE_SELECTION_GUIDANCE below rather than silently defaulting to one
model nobody questions -- this project's own data plus the HindiWiC
finding (cited in vindex.calibration's MuRIL warning) both say judge
competence on Indic content is not uniform across models and degrades
with model size, so a specific recommendation is more honest than an
unexamined default.

MILESTONE 5.8 (validate against human labels): NOT covered by this
module. It needs a real annotation study (human graders scoring the
same cases this judge scores, per language, compared against an
English-rubric baseline) that has not been run. Do not present this
module's scores as validated against human judgment until that study
exists -- see the module-level TODO at the bottom of this file.
"""

from __future__ import annotations

import json
import re

from vindex.judge_align import align
from vindex.judge_model import GroqJudge, JudgeModel
from vindex.judge_rubric import build_reference_based_prompt, build_reference_free_prompt
from vindex.result import MetricResult

JUDGE_SELECTION_GUIDANCE = """
Judge model selection is not a solved default -- pick deliberately:

- openai/gpt-oss-120b (via Groq): the model this project's own Phase 0
  rubric-quality experiments were run against (experiments/FINDINGS.md).
  Recommended default for Hindi/Hinglish content specifically because
  it is the one combination this project has direct evidence for, not
  because it is assumed to generalize to every Indic language.
- Smaller models degrade on Indic content specifically, not just on
  language tasks in general -- this project's calibration data
  (vindex.calibration) and the HindiWiC finding (Dairkee & Dubossarsky,
  2024, cited in vindex.calibration.MURIL_WARNING) both show
  Indic-language competence is not a simple function of overall model
  capability. Do not assume a smaller/cheaper model "should be fine"
  for Indic judging just because it performs adequately on English
  tasks -- verify against your own labelled data (see judge_align's
  cost-bounding, and Milestone 5.8's human-agreement validation, not
  yet run).
- NEVER use the same model (or a model from the same family/provider
  fine-tune lineage) as both the model being judged and the judge
  itself -- self-enhancement bias is a documented effect, not a
  theoretical risk. indic_judge cannot fully detect this for you; see
  _warn_if_same_family below for the partial, name-based check this
  module does perform.
"""

_MAX_SCORE = 5


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _parse_judge_response(raw: str) -> tuple[float, str, str]:
    """Parse the judge's JSON response into (score in [0,1], reasoning,
    confidence). Raises ValueError on a malformed response -- callers
    should treat that as a failed judge call, not silently default to
    a score."""
    data = json.loads(_extract_json(raw))
    raw_score = int(round(float(data["score"])))
    raw_score = max(1, min(_MAX_SCORE, raw_score))
    normalized_score = (raw_score - 1) / (_MAX_SCORE - 1)
    reasoning = str(data.get("reasoning", ""))
    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in ("high", "low"):
        confidence = "low"
    return normalized_score, reasoning, confidence


def _family(model_id: str) -> str:
    """Coarse model-family key for the same-family self-enhancement
    check -- e.g. "openai/gpt-oss-120b" and "openai/gpt-oss-20b" share
    the "openai/gpt-oss" family despite being different sizes, which
    is exactly the case _warn_if_same_family exists to catch (a judge
    from the same lineage as the model it is judging, even at a
    different size, is still a self-enhancement bias risk)."""
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
            "self-enhancement bias risk (Milestone 5.7). Treat this "
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
    """Is `answer` a correct, factually accurate response to `question`?

    Reference-free by default (gold=None) -- see this module's
    docstring for why that is the recommended mode, not just the
    convenient one. Pass `gold` to use reference-based mode instead,
    which runs align-then-judge (Milestone 5.4): an exact match with
    `gold` skips the LLM call entirely.

    judge defaults to vindex.judge_model.GroqJudge() (needs
    GROQ_API_KEY in the environment). Pass answering_model_id (the
    identifier of the model that produced `answer`, if you have it) to
    get a same-family self-enhancement-bias warning in the result's
    detail payload when it matches the judge model's family.

    CONSERVATIVE DEFAULT (Milestone 5.5): passed=True only when the
    judge reports "high" confidence AND a normalized score >= 0.8 (a
    raw 4 or 5 out of 5). Anything else -- including a high score at
    low confidence -- is passed=False with label "flagged", not
    silently treated as a pass. Better a false alarm than a silent
    pass on a wrong answer in a banking flow.
    """
    question = question or ""
    answer = answer or ""

    if question.strip() == "" or answer.strip() == "":
        return MetricResult(
            score=0.0,
            passed=False,
            label="empty",
            reason="question or answer is empty.",
            detail={"question": question, "answer": answer},
        )

    judge = judge or GroqJudge()
    family_warning = _warn_if_same_family(judge.model_id, answering_model_id)

    if gold is not None and gold.strip() != "":
        alignment = align(answer, gold)
        if alignment.aligned:
            detail = {
                "mode": "reference_based",
                "judge_model_id": judge.model_id,
                "aligned": True,
                "reasoning": "answer aligns exactly with gold at the word level; no LLM call made.",
            }
            if family_warning:
                detail["self_enhancement_bias_warning"] = family_warning
            return MetricResult(
                score=1.0,
                passed=True,
                label="matched",
                reason="answer aligns exactly with gold; no judge call needed (Milestone 5.4).",
                detail=detail,
            )
        prompt = build_reference_based_prompt(
            question, alignment.mismatched_response, alignment.mismatched_gold
        )
        mode = "reference_based"
    else:
        prompt = build_reference_free_prompt(question, answer)
        mode = "reference_free"

    raw_response = judge.call(prompt)
    try:
        score, reasoning, confidence = _parse_judge_response(raw_response)
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
    if family_warning:
        detail["self_enhancement_bias_warning"] = family_warning

    return MetricResult(score=score, passed=passed, label=label, reason=reason, detail=detail)


# MILESTONE 5.8 (not implemented): validate against human labels --
# needs a real annotation study (human graders on the same cases,
# per-language agreement, compared against an English-rubric
# baseline). See this module's docstring.
