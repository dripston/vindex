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

MILESTONE 5.8 (validate against human labels, vs an English-rubric
baseline): run, using the Milestone 6.3 annotation data. Result: an
English-rubric baseline agreed with the same human graders slightly
MORE than this module's Hindi rubric on that 62-trace set (93.5% vs
90.3%) -- the opposite of this module's motivating hypothesis.
Inspection of the disagreements shows this module being more
conservative about answer completeness, not less accurate about
comprehension (see experiments/README.md's Milestone 5.8 section for
the full breakdown). Do not present the Hindi rubric as validated to
agree with humans better than an English one on this evidence; that
specific claim did not hold on this sample.
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
  cost-bounding, and Milestone 5.8's human-agreement validation:
  93.5% for an English rubric vs 90.3% for this module's Hindi
  rubric on the same 62 traces, so rubric language alone is not a
  substitute for checking your own data either).
- NEVER use the same model (or a model from the same family/provider
  fine-tune lineage) as both the model being judged and the judge
  itself -- self-enhancement bias is a documented effect, not a
  theoretical risk. indic_judge cannot fully detect this for you; see
  _warn_if_same_family below for the partial, name-based check this
  module does perform.
"""

_MAX_SCORE = 5


def _iter_balanced_objects(text: str) -> list[str]:
    """Find every balanced {...} span in `text`, in order, by brace
    counting -- correctly skipping over braces inside string literals
    so a `{` or `}` in a quoted value doesn't miscount. One candidate
    per top-level `{` found (nested objects are captured as part of
    their enclosing span, not separately)."""
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

    FIXED (a real bug, found by an independent outside review): this
    used to be `re.search(r"\\{.*\\}", text, re.DOTALL)` -- greedy, so
    it grabbed from the FIRST `{` to the LAST `}` in the whole text,
    not a real JSON object. A judge response that mentions any brace in
    its prose before its actual JSON answer (e.g. "Reasoning: use
    {a:1}. Final: {"score":5,...}") got the entire span from the first
    brace to the last one, including the prose text glued in the
    middle -- not valid JSON, so the whole response was thrown away as
    judge_error instead of parsing the real object that was there.

    Now: find every balanced {...} span (brace-counted, string-literal-
    aware) and return the first one that is ACTUALLY valid JSON, not
    just the first one that is balanced -- "{a:1}" in the example above
    is balanced but not valid JSON (unquoted key), so it is skipped in
    favor of the real object that follows. Falls back to the raw text
    if no candidate parses, so a genuinely malformed response still
    reaches json.loads() and raises there, same as before."""
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
    """Parse the judge's JSON response into (score in [0,1], reasoning,
    confidence, raw_confidence). Raises ValueError on a malformed
    response -- callers should treat that as a failed judge call, not
    silently default to a score.

    confidence is the normalized value used for the pass/fail gate --
    "high" or "low" only, with anything else (e.g. "medium", a typo, a
    language other than English) folded into "low" as the safe
    default. raw_confidence is exactly what the judge sent, unmodified
    (added after an independent outside review found that when a judge
    sent "medium", the normalized "low" silently overwrote it
    everywhere, including in `detail` -- so the audit trail claimed the
    judge said "low" when it actually said "medium", misrepresenting
    what happened even though the conservative low-confidence
    *behavior* was correct). Report both: use `confidence` for gating
    logic, `raw_confidence` for anything a human or a log will read.

    A score outside the rubric's documented 1-5 range is ALSO malformed
    and raises ValueError, rather than being clamped into range. The
    rubric prompt asks for "1 to 5" explicitly (judge_rubric.py); a
    score of e.g. 100 (a judge misreading the scale, or an adversarial/
    corrupted response) is not a valid 1-5 score that happens to be out
    of bounds -- it is evidence the response did not follow the
    contract at all. Silently clamping it to 5 would turn a malformed
    response into indic_judge's most confident possible pass
    (score=1.0, and passed=True if confidence is also "high"), which is
    worse than raising and surfacing it as judge_error the way every
    other malformed shape already is."""
    data = json.loads(_extract_json(raw))
    raw_score_field = data["score"]
    if isinstance(raw_score_field, bool):
        # FIXED (a real bug, found by an independent outside review):
        # bool is a subclass of int in Python, so float(True) == 1.0
        # and float(False) == 0.0 both succeed silently. {"score":
        # true} used to be accepted as score=1 (passing the 1-5 range
        # check as the lowest valid score) and {"score": false} was
        # accepted as score=0 (correctly rejected by the range check,
        # but only by accident -- 0 happens to be out of [1,5], not
        # because a bool was recognized as an invalid type). A judge
        # emitting a boolean for "score" has not followed the
        # documented contract any more than a string or a list would
        # have -- reject explicitly, consistent with every other
        # malformed shape degrading to judge_error.
        raise ValueError(f"score {raw_score_field!r} is a boolean, not a number")
    if not isinstance(raw_score_field, (int, float)):
        # FIXED (a real bug, found by an independent outside review):
        # float("5") == 5.0 succeeds silently, so {"score": "5"} (a
        # JSON string, not a number) passed through as a clean score=1.0
        # pass -- exactly the "didn't follow the numeric contract"
        # failure the bool check above exists to catch, just via a
        # different JSON type it missed. Reject any score that is not
        # already a JSON number (int/float), consistent with the bool
        # rejection immediately above and the list/dict rejection that
        # already happens implicitly (float(["a"]) raises TypeError,
        # caught by indic_judge's except tuple).
        raise ValueError(f"score {raw_score_field!r} is not a JSON number")
    score_value = float(raw_score_field)
    try:
        raw_score = int(round(score_value))
    except (OverflowError, ValueError) as exc:
        # float("inf")/float("-inf") pass json.loads (Python's decoder
        # accepts the non-standard "Infinity"/"-Infinity" tokens by
        # default) and pass float(), but round()/int() on an infinite
        # float raises OverflowError, not ValueError -- so it wasn't
        # caught by the score-range check below, or by the caller's
        # except tuple, and crashed indic_judge() instead of degrading
        # to judge_error like every other malformed shape. NaN was
        # already caught (float("nan") != anything, so the range check
        # below raised ValueError) -- this closes the same gap for +-inf.
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
    """Coarse model-family key for the same-family self-enhancement
    check -- e.g. "openai/gpt-oss-120b" and "openai/gpt-oss-20b" share
    the "openai/gpt-oss" family despite being different sizes, which
    is exactly the case _warn_if_same_family exists to catch (a judge
    from the same lineage as the model it is judging, even at a
    different size, is still a self-enhancement bias risk).

    HOW NARROW THIS ACTUALLY IS: only a trailing "-<digits>[bm]" size
    token is stripped. It does NOT strip a provider/host prefix, so
    "openai/gpt-oss-120b" vs "groq/gpt-oss-120b-turbo" (same base
    model, different host, extra suffix) are treated as different
    families -- no warning. Any fine-tune or variant suffix after the
    size token ("-instruct", "-turbo", "-ft", a version tag) breaks
    detection the same way. This is not just "a fine-tune with a
    deliberately unrelated name evades it" (this module's own docstring
    and JUDGE_SELECTION_GUIDANCE's phrasing) -- in practice, almost any
    real-world model-naming convention already evades it, since none of
    them are "identical name plus a bare size suffix, nothing else."
    Only a same-repo, same-naming-convention, different-SIZE variant is
    reliably caught. Treat this warning as a narrow, best-effort catch
    for one specific naming pattern, not a general same-family
    detector.

    MISSES THE MOST COMMON REAL PAIRING (found by an independent
    outside review): "gpt-4o" vs "gpt-4o-mini" -- almost certainly the
    single most common self-judging pair in production today -- is NOT
    caught. Splitting "gpt-4o-mini" on "-" gives ["gpt", "4o", "mini"];
    the trailing token "mini" is not a bare size digit, so this
    function returns the model_id unchanged instead of stripping
    anything, and "gpt-4o" (unchanged) != "gpt-4o-mini" (unchanged).
    Same for "llama-3.1-70b-instruct" vs "llama-3.1-8b-instruct" (a
    trailing "-instruct" suffix after the size token blocks the strip
    the same way the docstring above already describes for other
    suffixes). A reliable general fix needs an actual model-family
    lookup table or a smarter naming-convention parser, not a
    marginally-smarter regex -- a fragile pattern expansion that still
    misses the next naming convention would give a false sense that
    this check is more complete than it is, which is worse than the
    current, honestly-narrow state. Not attempted here for that
    reason; `detail`'s absence of a self-enhancement warning must never
    be read as "safe from self-enhancement bias" -- verify your own
    judge/answering-model pairing directly."""
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
    judge reports "high" confidence AND a normalized score >= 0.8.
    CORRECTED (this docstring previously said "a raw 4 or 5 out of
    5" -- wrong, found by an independent outside review): normalization
    is `(raw_score - 1) / (_MAX_SCORE - 1)`, so a raw 4 normalizes to
    (4-1)/4 = 0.75, which does NOT clear the 0.8 gate. Only a raw 5
    passes; a raw 4 at high confidence is passed=False, label=
    "flagged" -- exactly like a raw 4 at low confidence. Anything below
    a raw 5 -- including a high score at low confidence -- is
    passed=False with label "flagged", not silently treated as a pass.
    Better a false alarm than a silent pass on a wrong answer in a
    banking flow.
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

    judge = judge or GroqJudge()
    family_warning = _warn_if_same_family(judge.model_id, answering_model_id)

    if gold is not None and not isinstance(gold, str):
        # A non-string, non-None gold (e.g. a pandas nan in a dataframe
        # cell) would crash on gold.strip() below -- treat it the same
        # as gold=None (reference-free mode) rather than crash, same
        # fix as coerce_text applies to question/answer above.
        gold = None

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
        # The judge sent something other than exactly "high"/"low"
        # (e.g. "medium") -- report what it actually said alongside the
        # normalized value used for gating, instead of silently
        # overwriting it (see _parse_judge_response's docstring).
        detail["raw_confidence"] = raw_confidence
    if family_warning:
        detail["self_enhancement_bias_warning"] = family_warning

    return MetricResult(score=score, passed=passed, label=label, reason=reason, detail=detail)


# MILESTONE 5.8 (done): validate against human labels, vs an
# English-rubric baseline. See this module's docstring and
# experiments/README.md for the result -- the English-rubric baseline
# agreed with humans slightly more than this module's Hindi rubric on
# the 6.3 study's 62 traces (93.5% vs 90.3%).
