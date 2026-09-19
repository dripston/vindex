"""
Tests for vindex.datasets (v0.5.0): benchmark data + scoring helpers so
a caller can verify calibrated_similarity/indic_judge on their own
encoder/judge instead of trusting this project's shipped numbers alone.
Real data, real files -- no mocks, matching this project's preference
for real data over mocks.
"""

from __future__ import annotations

from vindex.datasets import (
    JudgeBenchmarkTrace,
    SimilarityBenchmarkCase,
    load_judge_benchmark,
    load_similarity_benchmark,
    score_judge_benchmark,
    similarity_benchmark_for_calibration,
)

# --- load_similarity_benchmark ---


def test_load_similarity_benchmark_returns_list_of_cases() -> None:
    cases = load_similarity_benchmark()
    assert isinstance(cases, list)
    assert all(isinstance(c, SimilarityBenchmarkCase) for c in cases)


def test_load_similarity_benchmark_has_89_cases() -> None:
    # 10 questions x 3 languages x {correct, wrong_subtle, wrong_hard} =
    # 90, minus 1 combination missing from the source data -- matches
    # experiments/results_clean/discrimination_per_case.csv's real
    # unique case_id count, not a round number chosen for its own sake.
    cases = load_similarity_benchmark()
    assert len(cases) == 89


def test_load_similarity_benchmark_covers_all_three_languages() -> None:
    cases = load_similarity_benchmark()
    assert {c.language for c in cases} == {"en", "hi", "hinglish"}


def test_load_similarity_benchmark_covers_all_three_labels() -> None:
    cases = load_similarity_benchmark()
    assert {c.label for c in cases} == {"correct", "wrong_subtle", "wrong_hard"}


def test_load_similarity_benchmark_case_ids_are_unique() -> None:
    cases = load_similarity_benchmark()
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_load_similarity_benchmark_includes_documented_example() -> None:
    # README.md's calibrated_similarity example uses this exact
    # question -- confirms the shipped benchmark is the same data the
    # docs are built from, not a divergent copy.
    cases = load_similarity_benchmark()
    task_ids = {c.task_id for c in cases}
    assert "capital_maharashtra" in task_ids


# --- similarity_benchmark_for_calibration ---


def test_similarity_benchmark_for_calibration_buckets_by_label() -> None:
    cases = load_similarity_benchmark()

    def fake_score(gold: str, answer: str) -> float:
        return 1.0 if answer == gold else 0.0

    correct, wrong = similarity_benchmark_for_calibration(cases, fake_score, language="en")
    n_en = sum(1 for c in cases if c.language == "en")
    assert len(correct) + len(wrong) == n_en
    n_en_correct = sum(1 for c in cases if c.language == "en" and c.label == "correct")
    assert len(correct) == n_en_correct


def test_similarity_benchmark_for_calibration_treats_both_wrong_labels_as_wrong() -> None:
    cases = [
        SimilarityBenchmarkCase("c1", "t1", "en", "correct", "gold", "ans1"),
        SimilarityBenchmarkCase("c2", "t1", "en", "wrong_subtle", "gold", "ans2"),
        SimilarityBenchmarkCase("c3", "t1", "en", "wrong_hard", "gold", "ans3"),
    ]
    correct, wrong = similarity_benchmark_for_calibration(cases, lambda g, a: 0.5)
    assert len(correct) == 1
    assert len(wrong) == 2


def test_similarity_benchmark_for_calibration_language_filter() -> None:
    cases = [
        SimilarityBenchmarkCase("c1", "t1", "en", "correct", "gold", "ans1"),
        SimilarityBenchmarkCase("c2", "t1", "hi", "correct", "gold", "ans2"),
    ]
    correct, wrong = similarity_benchmark_for_calibration(cases, lambda g, a: 0.5, language="en")
    assert len(correct) == 1
    assert len(wrong) == 0


def test_similarity_benchmark_for_calibration_passes_gold_and_answer_through() -> None:
    cases = [
        SimilarityBenchmarkCase("c1", "t1", "en", "correct", "the gold text", "the answer text"),
    ]
    seen = []

    def recording_score(gold: str, answer: str) -> float:
        seen.append((gold, answer))
        return 1.0

    similarity_benchmark_for_calibration(cases, recording_score)
    assert seen == [("the gold text", "the answer text")]


# --- load_judge_benchmark ---


def test_load_judge_benchmark_returns_list_of_traces() -> None:
    traces = load_judge_benchmark()
    assert isinstance(traces, list)
    assert all(isinstance(t, JudgeBenchmarkTrace) for t in traces)


def test_load_judge_benchmark_has_62_traces() -> None:
    # Matches Milestone 6.3's human-agreement study exactly -- see
    # README.md's Limitations section and docs/annotation/BIAS_PROTOCOL.md.
    traces = load_judge_benchmark()
    assert len(traces) == 62


def test_load_judge_benchmark_has_19_holdout_traces() -> None:
    # 30.6% holdout per BIAS_PROTOCOL.md, selected via random.seed(1234)
    # before grading started, not hand-picked.
    traces = load_judge_benchmark()
    assert sum(1 for t in traces if t.is_holdout) == 19


def test_load_judge_benchmark_trace_ids_are_unique() -> None:
    traces = load_judge_benchmark()
    ids = [t.trace_id for t in traces]
    assert len(ids) == len(set(ids))


def test_load_judge_benchmark_includes_documented_samudra_tal_case() -> None:
    # The exact समुद्र तल (sea level / sea floor) case this project's
    # whole indic_judge rubric-in-Hindi decision is built on.
    traces = load_judge_benchmark()
    assert any(t.trap_word == "समुद्र तल" for t in traces)


def test_load_judge_benchmark_verdicts_are_correct_or_wrong() -> None:
    traces = load_judge_benchmark()
    for t in traces:
        assert t.ground_truth_answer_type in ("correct", "wrong")
        assert t.human_verdict_grader1 in ("correct", "wrong")
        assert t.human_verdict_grader2_independent in ("correct", "wrong")


# --- score_judge_benchmark ---


def test_score_judge_benchmark_perfect_verdict_scores_using_ground_truth() -> None:
    # A judge_verdict that always agrees with BOTH human graders should
    # score 1.0 agreement with both -- exercised against the real
    # dataset, not a fixture, since the point is confirming this
    # actually reads and scores the real 62 traces correctly.
    traces = load_judge_benchmark()
    by_id = {t.trace_id: t for t in traces}

    def oracle_verdict(trace: JudgeBenchmarkTrace) -> str:
        # Not every trace has both graders agreeing with ground truth,
        # so agree with grader1 specifically to get a verifiable number.
        return by_id[trace.trace_id].human_verdict_grader1

    report = score_judge_benchmark(oracle_verdict)
    assert report.n == 62
    assert report.agreement_grader1 == 1.0


def test_score_judge_benchmark_reports_disagreements() -> None:
    traces = load_judge_benchmark()

    def always_correct(trace: JudgeBenchmarkTrace) -> str:
        return "correct"

    report = score_judge_benchmark(always_correct)
    n_grader1_wrong = sum(1 for t in traces if t.human_verdict_grader1 == "wrong")
    assert report.n == 62
    # every trace where grader1 said "wrong" is a disagreement against
    # a judge_verdict that always says "correct"
    assert len(report.disagreements) >= n_grader1_wrong


def test_score_judge_benchmark_only_holdout_scores_19() -> None:
    report = score_judge_benchmark(lambda t: "correct", only_holdout=True)
    assert report.n == 19


def test_score_judge_benchmark_agreement_is_fraction_between_0_and_1() -> None:
    report = score_judge_benchmark(lambda t: "wrong")
    assert 0.0 <= report.agreement_grader1 <= 1.0
    assert 0.0 <= report.agreement_grader2 <= 1.0
