"""Milestone 2.4: beat the exact-match baseline, report the number.

The baseline: run_experiment.py's naive exact_match (lowercase, strip
punctuation, collapse whitespace, nothing script-aware) scores 0/30 on
data/results_clean.json when compared against each task's bare gold
("Mumbai") -- because the model answers are full sentences ("Mumbai is
the capital of Maharashtra..."), so a bare-entity gold can never
string-equal a full-sentence answer. That register mismatch, not
script, is what drove that particular 0/30 -- see
experiments/testcases.py's own gold_en_full/gold_hi_full/
gold_hinglish_full fields, added specifically so a register-matched
gold exists to compare against.

This script re-runs the same 30 (answer, gold) pairs from
data/results_clean.json through vindex.match's three match modes,
using the register-matched full-sentence gold for each row's variant
(gold_en_full / gold_hi_full / gold_hinglish_full from testcases.py,
joined by task_id). This is the fair comparison: same dataset, same
answers, a gold that matches the answer's register, scored by
vindex instead of by the naive baseline scorer.
"""

from __future__ import annotations

import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, EXPERIMENTS_DIR)

from testcases import TASKS  # noqa: E402

from vindex.match import char_similarity_score, exact_match_score, token_f1_score  # noqa: E402

RESULTS_CLEAN_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")

_FULL_GOLD_FIELD = {
    "en": "gold_en_full",
    "hi": "gold_hi_full",
    "hinglish": "gold_hinglish_full",
}


def _baseline_normalize(s: str | None) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s


def _baseline_exact_match(answer: str, gold: str) -> bool:
    a = _baseline_normalize(answer)
    g = _baseline_normalize(gold)
    return a == g or a == g + "." or g == a


def main() -> None:
    full_gold_by_task = {task["id"]: task for task in TASKS}

    with open(RESULTS_CLEAN_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    baseline_hits = 0
    vindex_exact_hits = 0
    f1_scores = []
    char_scores = []

    print(f"{len(rows)} rows from results_clean.json\n")
    for row in rows:
        answer = row["answer"]
        variant = row["variant"]
        full_gold = full_gold_by_task[row["task_id"]][_FULL_GOLD_FIELD[variant]]

        baseline_ok = _baseline_exact_match(answer, row["gold"])
        vindex_exact = exact_match_score(answer, full_gold)
        f1 = token_f1_score(answer, full_gold)
        char_sim = char_similarity_score(answer, full_gold)

        baseline_hits += int(baseline_ok)
        vindex_exact_hits += int(vindex_exact == 1.0)
        f1_scores.append(f1)
        char_scores.append(char_sim)

        print(
            f"{row['case_id']:35s} baseline={'hit ' if baseline_ok else 'miss'} "
            f"vindex_exact={vindex_exact:.1f} f1={f1:.2f} char={char_sim:.2f}"
        )

    n = len(rows)
    print(f"\nbaseline exact match (bare gold, naive normalize): {baseline_hits}/{n}")
    print(f"vindex exact_match_score (full-sentence gold):      {vindex_exact_hits}/{n}")
    print(f"vindex token_f1_score, mean:                        {sum(f1_scores) / n:.3f}")
    print(f"vindex char_similarity_score, mean:                 {sum(char_scores) / n:.3f}")


if __name__ == "__main__":
    main()
