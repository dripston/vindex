"""
Milestone 6.3 / step 4: compute human-vs-human and human-vs-judge
agreement, on the tuning set and the holdout separately.

Reads two completed grading sheets (yours and the second grader's --
same columns as data/trace_study_grading_sheet.csv, with
grader_verdict_correct_or_wrong filled in) plus the pinned ground-truth
file data/trace_study_traces.json, and reports:

  - inter-annotator agreement (grader 1 vs grader 2)
  - each grader vs indic_judge's own judge_passed verdict
  - all of the above, split by tuning-set vs holdout-set trace_ids

Does NOT grade anything itself -- both grading sheets must already be
filled in by two independent humans, per docs/annotation/BIAS_PROTOCOL.md.
Run this only after both graders have submitted, and only after
recording which trace_ids are the holdout BEFORE grading started (see
BIAS_PROTOCOL.md Step 2).

Run:
  python experiments/scripts/compute_trace_study_agreement.py \
      --grader1 path/to/grader1_sheet.csv \
      --grader2 path/to/grader2_sheet.csv \
      --holdout T001,T017,T033,...
"""

from __future__ import annotations

import argparse
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TRACES_PATH = os.path.join(DATA_DIR, "trace_study_traces.json")


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

    judge_verdict = {
        tid: ("correct" if t["judge_passed"] else "wrong") for tid, t in traces.items()
    }

    print(f"total traces: {len(all_ids)}  tuning: {len(tuning_ids)}  holdout: {len(holdout_ids)}\n")

    for label, ids in [("TUNING SET", tuning_ids), ("HOLDOUT SET", holdout_ids), ("ALL", all_ids)]:
        print(f"--- {label} ({len(ids)} traces) ---")
        m, t = _agreement(grader1, grader2, ids)
        pct = f"{m}/{t} = {m / t:.1%}" if t else "no data"
        print(f"  grader1 vs grader2 (inter-annotator): {pct}")
        m, t = _agreement(grader1, judge_verdict, ids)
        pct = f"{m}/{t} = {m / t:.1%}" if t else "no data"
        print(f"  grader1 vs indic_judge:                {pct}")
        m, t = _agreement(grader2, judge_verdict, ids)
        pct = f"{m}/{t} = {m / t:.1%}" if t else "no data"
        print(f"  grader2 vs indic_judge:                {pct}")
        print()

    print(
        "Report these numbers in README.md / experiments/README.md, "
        "alongside the disclosure required by BIAS_PROTOCOL.md Step 5, "
        "whatever the numbers are."
    )


if __name__ == "__main__":
    main()
