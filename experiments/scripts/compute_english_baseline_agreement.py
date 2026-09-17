"""
Milestone 5.8: compare indic_judge's human agreement against the
English-rubric baseline's, on the exact same 62 traces and the exact
same human grades already collected for Milestone 6.3.

Reads data/trace_study_traces_en_baseline.json (the pinned 6.3 traces
plus en_judge_passed, written by run_english_rubric_baseline.py) and
the same two grading sheets used in compute_trace_study_agreement.py,
then reports, side by side, per grader and split tuning/holdout:

  - grader vs indic_judge (judge_passed)      -- already known: 90.3% overall
  - grader vs english-rubric baseline (en_judge_passed)

This is the one comparison Milestone 5.8 asks for and 6.3 alone does
not answer: does the Hindi rubric actually agree with humans more than
an English rubric on the same content, or not.

Run:
  python experiments/scripts/compute_english_baseline_agreement.py \
      --grader1 data/grading_submissions/grader1_sheet.csv \
      --grader2 data/grading_submissions/grader2_sheet.csv \
      --holdout T001,T002,T003,T006,T007,T008,T016,T023,T029,T038,T043,T045,T046,T047,T049,T050,T052,T057,T062
"""

from __future__ import annotations

import argparse
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TRACES_PATH = os.path.join(DATA_DIR, "trace_study_traces_en_baseline.json")


def _load_grades(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grades = {}
    for row in rows:
        verdict = row["grader_verdict_correct_or_wrong"].strip().lower()
        if verdict not in ("correct", "wrong", "unsure"):
            raise ValueError(
                f"trace {row['trace_id']}: verdict must be correct/wrong/unsure, got {verdict!r}"
            )
        grades[row["trace_id"]] = verdict
    return grades


def _agreement(a: dict[str, str], b: dict[str, str], trace_ids: set[str]) -> tuple[int, int]:
    matched = 0
    total = 0
    for tid in trace_ids:
        if tid not in a or tid not in b:
            continue
        if a[tid] == "unsure" or b[tid] == "unsure":
            continue
        total += 1
        if a[tid] == b[tid]:
            matched += 1
    return matched, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grader1", required=True)
    parser.add_argument("--grader2", required=True)
    parser.add_argument(
        "--holdout", required=True, help="comma-separated trace_ids, e.g. T001,T017"
    )
    args = parser.parse_args()

    with open(TRACES_PATH, encoding="utf-8") as f:
        traces = {t["trace_id"]: t for t in json.load(f)}

    grader1 = _load_grades(args.grader1)
    grader2 = _load_grades(args.grader2)
    holdout_ids = set(args.holdout.split(","))
    all_ids = set(traces.keys())
    tuning_ids = all_ids - holdout_ids

    hindi_verdict = {tid: ("correct" if t["judge_passed"] else "wrong") for tid, t in traces.items()}
    en_verdict = {tid: ("correct" if t["en_judge_passed"] else "wrong") for tid, t in traces.items()}

    print(f"total traces: {len(all_ids)}  tuning: {len(tuning_ids)}  holdout: {len(holdout_ids)}\n")

    for label, ids in [("TUNING SET", tuning_ids), ("HOLDOUT SET", holdout_ids), ("ALL", all_ids)]:
        print(f"--- {label} ({len(ids)} traces) ---")
        for gname, g in [("grader1", grader1), ("grader2", grader2)]:
            m, t = _agreement(g, hindi_verdict, ids)
            hindi_pct = f"{m}/{t} = {m / t:.1%}" if t else "no data"
            m, t = _agreement(g, en_verdict, ids)
            en_pct = f"{m}/{t} = {m / t:.1%}" if t else "no data"
            print(f"  {gname} vs indic_judge (Hindi rubric):   {hindi_pct}")
            print(f"  {gname} vs english_baseline (En rubric): {en_pct}")
        print()

    # Where do the two judges actually disagree with each other?
    disagreements = [
        tid
        for tid in sorted(all_ids)
        if hindi_verdict[tid] != en_verdict[tid]
    ]
    print(f"indic_judge vs english_baseline disagree on {len(disagreements)}/{len(all_ids)} traces:")
    for tid in disagreements:
        t = traces[tid]
        print(
            f"  {tid} {t['trap_word']:12} {t['answer_type']:8} "
            f"hindi={'pass' if t['judge_passed'] else 'flag'} "
            f"en={'pass' if t['en_judge_passed'] else 'flag'}"
        )


if __name__ == "__main__":
    main()
