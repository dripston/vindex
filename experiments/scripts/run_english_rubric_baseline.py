"""
Milestone 5.8: run the English-rubric baseline judge against the same
62 pinned traces from the Milestone 6.3 study, so its agreement with
the human grades already collected can be compared directly against
indic_judge's own 90.3%.

Reads data/trace_study_traces.json (pinned, NOT regenerated -- same
questions/answers indic_judge was already scored against). Calls the
English-rubric baseline (vindex.judge_rubric_en_baseline) via the same
GroqJudge backend and the same score/confidence parsing indic_judge
uses, so the only variable that differs from the existing judge_passed
column is rubric language.

Writes data/trace_study_traces_en_baseline.json: the same 62 records,
with three new fields (en_judge_score, en_judge_passed,
en_judge_confidence) appended -- does not modify the original pinned
file.

Usage:
    python experiments/scripts/run_english_rubric_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vindex.judge_model import GroqJudge
from vindex.judge_rubric_en_baseline import build_reference_free_prompt_en_baseline
from vindex.judge import _parse_judge_response  # reuse indic_judge's own parser

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = REPO_ROOT / "data" / "trace_study_traces.json"
OUT_PATH = REPO_ROOT / "data" / "trace_study_traces_en_baseline.json"


def main() -> None:
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    judge = GroqJudge()

    for i, t in enumerate(traces, 1):
        prompt = build_reference_free_prompt_en_baseline(t["question"], t["answer"])
        raw = judge.call(prompt)
        try:
            score, reasoning, confidence, _raw_confidence = _parse_judge_response(raw)
            passed = confidence == "high" and score >= 0.8
        except Exception as exc:  # noqa: BLE001 -- record the failure, don't crash the run
            score, reasoning, confidence, passed = 0.0, f"PARSE_ERROR: {exc}", "low", False

        t["en_judge_score"] = score
        t["en_judge_passed"] = passed
        t["en_judge_confidence"] = confidence
        t["en_judge_reasoning"] = reasoning

        print(f"[{i:2}/{len(traces)}] {t['trace_id']} {t['trap_word']:12} "
              f"{t['answer_type']:8} -> en_judge passed={passed}", flush=True)

    OUT_PATH.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(traces)} traces (with en_judge_* fields) to {OUT_PATH}")


if __name__ == "__main__":
    main()
