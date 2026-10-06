"""Benchmark datasets for verifying vindex on your own models.

The calibration table behind :func:`vindex.calibrated_similarity` and the
human-agreement figures for :func:`vindex.indic_judge` are both backed by
labelled data, shipped here so you can measure your own encoder or judge
model on the same cases instead of relying on the published numbers.

- :func:`load_similarity_benchmark` returns 89 encoder-independent cases
  (10 questions x 3 languages x {correct, wrong_subtle, wrong_hard}, minus
  a few missing variants), each scored against a single English gold
  answer. Score them with your own encoder and pass the results to
  :func:`vindex.calibrate` via :func:`similarity_benchmark_for_calibration`.
- :func:`load_judge_benchmark` returns 62 Hindi trap-word traces with
  verdicts from two human graders, 19 of which form an untouched 30%
  holdout. Use :func:`score_judge_benchmark` to measure your judge's
  agreement with them.

Grader disclosure: grader 1 is the author of ``indic_judge`` (a conflict
of interest, mitigated by a frozen rubric, blind grading sheet, and
untouched holdout); grader 2 is an independent fluent Hindi speaker with
no stake in the result. Do not report agreement against grader 1 alone
without this disclosure.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from typing import Literal

_PACKAGE_DATA = "vindex.data.benchmarks"
_SIMILARITY_FILENAME = "similarity_benchmark.csv"
_JUDGE_FILENAME = "judge_benchmark.json"


@dataclass(frozen=True, slots=True)
class SimilarityBenchmarkCase:
    """One similarity benchmark case.

    Attributes:
        case_id: Unique id, e.g. ``"capital_maharashtra__hi__wrong_hard"``.
        task_id: Question id shared across language/label variants, e.g.
            ``"capital_maharashtra"``.
        language: ``"en"``, ``"hi"``, or ``"hinglish"``.
        label: ``"correct"``, ``"wrong_subtle"``, or ``"wrong_hard"``. Ground
            truth for scoring, not a vindex verdict.
        gold_text: Short English reference answer.
        answer: Candidate answer to score against ``gold_text``.
    """

    case_id: str
    task_id: str
    language: str
    label: Literal["correct", "wrong_subtle", "wrong_hard"]
    gold_text: str
    answer: str


@dataclass(frozen=True, slots=True)
class JudgeBenchmarkTrace:
    """One judge benchmark trace with human verdicts.

    Attributes:
        trace_id: Trace id, e.g. ``"T001"``.
        question: Question the judge grades an answer to.
        answer: Candidate answer.
        trap_word: Ambiguous term the trace is built around (may not appear
            verbatim in ``answer``).
        correct_meaning: The intended reading of ``trap_word``.
        wrong_meaning: The incorrect reading of ``trap_word``.
        ground_truth_answer_type: ``"correct"`` or ``"wrong"``, by
            construction, independent of any grader.
        human_verdict_grader1: Verdict from grader 1 (see the module
            docstring disclosure).
        human_verdict_grader2_independent: Verdict from the independent
            grader.
        is_holdout: True for the 19 traces (30.6%) neither grader saw until
            both full gradings were submitted.
    """

    trace_id: str
    question: str
    answer: str
    trap_word: str
    correct_meaning: str
    wrong_meaning: str
    ground_truth_answer_type: Literal["correct", "wrong"]
    human_verdict_grader1: Literal["correct", "wrong"]
    human_verdict_grader2_independent: Literal["correct", "wrong"]
    is_holdout: bool


def similarity_benchmark_for_calibration(
    cases: list[SimilarityBenchmarkCase],
    score_fn: Callable[[str, str], float],
    *,
    language: str | None = None,
) -> tuple[list[float], list[float]]:
    """Score benchmark cases and split them into calibrate() inputs.

    ``"wrong_subtle"`` and ``"wrong_hard"`` cases are both bucketed as
    wrong. Example::

        from vindex.calibration import calibrate
        from vindex.datasets import (
            load_similarity_benchmark,
            similarity_benchmark_for_calibration,
        )

        cases = load_similarity_benchmark()
        correct, wrong = similarity_benchmark_for_calibration(
            cases, my_cosine_similarity_fn, language="hi"
        )
        result = calibrate(correct, wrong)

    Args:
        cases: Cases from :func:`load_similarity_benchmark`.
        score_fn: Your scoring function, called as
            ``score_fn(gold_text, answer)``; typically your encoder's cosine
            similarity.
        language: Restrict to ``"en"``, ``"hi"``, or ``"hinglish"``. If
            omitted, all three languages are pooled.

    Returns:
        ``(similarities_correct, similarities_wrong)``, the two lists
        :func:`vindex.calibrate` takes.
    """
    correct = []
    wrong = []
    for case in cases:
        if language is not None and case.language != language:
            continue
        score = score_fn(case.gold_text, case.answer)
        if case.label == "correct":
            correct.append(score)
        else:
            wrong.append(score)
    return correct, wrong


def load_similarity_benchmark() -> list[SimilarityBenchmarkCase]:
    """Load the 89-case similarity benchmark.

    Returns:
        The cases behind the shipped calibration table. See the module
        docstring for usage.
    """
    text = resources.files(_PACKAGE_DATA).joinpath(_SIMILARITY_FILENAME).read_text(
        encoding="utf-8"
    )
    reader = csv.DictReader(text.splitlines())
    return [
        SimilarityBenchmarkCase(
            case_id=row["case_id"],
            task_id=row["task_id"],
            language=row["language"],
            label=row["label"],  # type: ignore[arg-type]
            gold_text=row["gold_text"],
            answer=row["answer"],
        )
        for row in reader
    ]


def load_judge_benchmark() -> list[JudgeBenchmarkTrace]:
    """Load the 62-trace judge benchmark.

    Returns:
        Traces with both human graders' verdicts. See the module docstring
        for usage and the grader disclosure.
    """
    text = resources.files(_PACKAGE_DATA).joinpath(_JUDGE_FILENAME).read_text(
        encoding="utf-8"
    )
    rows = json.loads(text)
    return [
        JudgeBenchmarkTrace(
            trace_id=row["trace_id"],
            question=row["question"],
            answer=row["answer"],
            trap_word=row["trap_word"],
            correct_meaning=row["correct_meaning"],
            wrong_meaning=row["wrong_meaning"],
            ground_truth_answer_type=row["ground_truth_answer_type"],
            human_verdict_grader1=row["human_verdict_grader1"],
            human_verdict_grader2_independent=row["human_verdict_grader2_independent"],
            is_holdout=row["is_holdout"],
        )
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class JudgeAgreementReport:
    """Agreement of a judge callback with the human graders.

    Attributes:
        n: Number of traces scored.
        agreement_grader1: Fraction of traces where the judge matched
            grader 1.
        agreement_grader2: Fraction of traces where the judge matched
            grader 2.
        disagreements: Trace ids where the judge disagreed with either
            grader. Inspect these first: they show whether the judge is
            wrong or being appropriately strict about something the
            rubric did not anticipate.
    """

    n: int
    agreement_grader1: float
    agreement_grader2: float
    disagreements: tuple[str, ...]


def score_judge_benchmark(
    judge_verdict: Callable[[JudgeBenchmarkTrace], Literal["correct", "wrong"]],
    *,
    only_holdout: bool = False,
) -> JudgeAgreementReport:
    """Measure a judge's agreement with the human graders.

    For reference, ``indic_judge`` with ``openai/gpt-oss-120b`` agreed on
    90.3% of all 62 traces and 94.7% of the holdout. Agreement may differ
    for other judge models, so measure your own. This function makes no
    LLM calls itself; the callback can wrap ``indic_judge``, another judge
    entirely, or a heuristic::

        from vindex import indic_judge
        from vindex.datasets import score_judge_benchmark

        def verdict(trace):
            r = indic_judge(trace.question, trace.answer)
            return "correct" if r.passed else "wrong"

        report = score_judge_benchmark(verdict)

    Args:
        judge_verdict: Callback that receives a :class:`JudgeBenchmarkTrace`
            and returns ``"correct"`` or ``"wrong"``.
        only_holdout: Score only the 19 holdout traces.

    Returns:
        A :class:`JudgeAgreementReport`.
    """
    traces = load_judge_benchmark()
    if only_holdout:
        traces = [t for t in traces if t.is_holdout]

    disagreements = []
    matches_g1 = 0
    matches_g2 = 0
    for trace in traces:
        verdict = judge_verdict(trace)
        if verdict == trace.human_verdict_grader1:
            matches_g1 += 1
        if verdict == trace.human_verdict_grader2_independent:
            matches_g2 += 1
        disagrees_g1 = verdict != trace.human_verdict_grader1
        disagrees_g2 = verdict != trace.human_verdict_grader2_independent
        if disagrees_g1 or disagrees_g2:
            disagreements.append(trace.trace_id)

    n = len(traces)
    return JudgeAgreementReport(
        n=n,
        agreement_grader1=matches_g1 / n if n else 0.0,
        agreement_grader2=matches_g2 / n if n else 0.0,
        disagreements=tuple(disagreements),
    )
