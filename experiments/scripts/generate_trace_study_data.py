"""
Milestone 6.3: generate the trace-study dataset.

Parses the 31 trap-seeded question/correct-answer/wrong-answer blocks
(drafted with help from Sarvam's chatbot, reviewed before use -- see
experiments/README.md) into 62 (question, answer) pairs, runs each
through vindex.indic_judge in reference-free mode, and captures the
FULL reasoning trace for each.

This is the "generate ~30 trap-seeded questions... judge verdicts
reference-free with full reasoning traces captured" step of
BUILD_PLAN.md 6.3. It does NOT grade anything -- grading is human work
(the project owner plus one independent fluent-Hindi-speaker grader),
done on the blinded sheet this script also produces. See
experiments/README.md's Milestone 6.3 section and
docs/annotation/BIAS_PROTOCOL.md for the process this data feeds into.

PINNING: once run, data/trace_study_traces.json is a pinned artifact,
same discipline as data/results_clean.json -- the judge model is not
perfectly deterministic even at temperature 0 (this project's own
finding), so re-running this script after grading has started would
silently invalidate every label already assigned. Do not re-run this
after data/trace_study_traces.json exists and grading has begun.

Run:  python experiments/scripts/generate_trace_study_data.py
Output:
  data/trace_study_traces.json   (pinned: question, answer_type,
                                   trap_word, answer text, judge score/
                                   passed/label, full reasoning trace)
  data/trace_study_grading_sheet.csv  (blinded: trace_id + trace text
                                   only, shuffled order, no labels)
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vindex.judge import indic_judge  # noqa: E402

SEED = 42
TRACES_PATH = os.path.join(DATA_DIR, "trace_study_traces.json")
GRADING_SHEET_PATH = os.path.join(DATA_DIR, "trace_study_grading_sheet.csv")

RAW_TESTCASES_PATH = os.path.join(DATA_DIR, "trap_words", "trace_study_source_testcases.txt")


def _parse_blocks(text: str) -> list[dict[str, str]]:
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    cases = []
    for block in blocks:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            m = re.match(r"^([A-Z_]+):\s*(.*)$", line.strip())
            if m:
                fields[m.group(1)] = m.group(2)
        required = {
            "WORD",
            "CORRECT_MEANING",
            "WRONG_MEANING",
            "QUESTION",
            "CORRECT_ANSWER",
            "WRONG_ANSWER",
        }
        missing = required - fields.keys()
        if missing:
            raise ValueError(f"block missing fields {missing}: {block[:80]}")
        cases.append(fields)
    return cases


def main() -> None:
    if os.path.exists(TRACES_PATH):
        raise SystemExit(
            f"{TRACES_PATH} already exists -- this dataset is pinned once "
            "generated (see module docstring). Refusing to overwrite. "
            "Delete it manually only if grading has NOT started yet."
        )

    with open(RAW_TESTCASES_PATH, encoding="utf-8") as f:
        raw = f.read()
    cases = _parse_blocks(raw)
    print(f"parsed {len(cases)} word blocks")

    traces = []
    trace_id = 0
    for case in cases:
        for answer_type in ("correct", "wrong"):
            answer = case["CORRECT_ANSWER"] if answer_type == "correct" else case["WRONG_ANSWER"]
            result = indic_judge(question=case["QUESTION"], answer=answer)
            trace_id += 1
            traces.append(
                {
                    "trace_id": f"T{trace_id:03d}",
                    "trap_word": case["WORD"],
                    "correct_meaning": case["CORRECT_MEANING"],
                    "wrong_meaning": case["WRONG_MEANING"],
                    "question": case["QUESTION"],
                    "answer_type": answer_type,  # "correct" or "wrong" -- ground truth
                    "answer": answer,
                    "judge_score": result.score,
                    "judge_passed": result.passed,
                    "judge_label": result.label,
                    "judge_reasoning": result.detail.get("judge_reasoning", ""),
                    "judge_confidence": result.detail.get("confidence", ""),
                    "judge_model_id": result.detail.get("judge_model_id", ""),
                }
            )
            print(
                f"  [{trace_id:2d}/{len(cases) * 2}] {case['WORD']:12s} "
                f"{answer_type:8s} -> judge={result.label} passed={result.passed}"
            )

    with open(TRACES_PATH, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(traces)} traces to {TRACES_PATH}")

    # Blinded grading sheet: shuffled order, trace_id + question + answer
    # + judge_reasoning only -- NO answer_type, NO judge_score/passed/
    # label, NO trap_word. A grader must not be able to infer ground
    # truth or the judge's own verdict from the sheet.
    rng = random.Random(SEED)
    shuffled = traces.copy()
    rng.shuffle(shuffled)

    import csv

    with open(GRADING_SHEET_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trace_id",
                "question",
                "answer",
                "judge_reasoning",
                "grader_verdict_correct_or_wrong",
                "grader_notes",
            ]
        )
        for t in shuffled:
            writer.writerow(
                [t["trace_id"], t["question"], t["answer"], t["judge_reasoning"], "", ""]
            )

    print(f"wrote blinded grading sheet ({len(shuffled)} rows, shuffled) to {GRADING_SHEET_PATH}")
    print(
        "\nNEXT STEP: do not open trace_study_traces.json while grading. "
        "Grade only from trace_study_grading_sheet.csv. See "
        "docs/annotation/BIAS_PROTOCOL.md."
    )


if __name__ == "__main__":
    main()
