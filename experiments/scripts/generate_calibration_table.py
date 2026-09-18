"""
Regenerate vindex.calibration.CALIBRATION_TABLE's warnings directly from
experiments/results_clean/discrimination_per_case.csv, using the real
vindex.calibrate() function on the real per-case similarity scores.

Written after an independent outside review pointed out that
calibrate()'s .warnings field (too-few-cases, out-of-range scores,
at-or-below-chance fit) only ever fires for a CALLER's own data -- the
shipped CALIBRATION_TABLE cells were hand-authored as raw
CalibratedThreshold(...) values before .warnings existed, and never
carried any, even though several of them (e.g. LaBSE/en, real AUC
0.070) would trip the exact same guard if fed through calibrate()
themselves. This script closes that gap by actually running
calibrate() on each shipped cell's real correct/wrong-hard similarity
scores, so the warnings are computed, not hand-typed.

Usage: python experiments/scripts/generate_calibration_table.py
Prints CALIBRATION_TABLE entries with real .threshold/.accuracy/
.warnings, ready to paste into calibration.py.
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from vindex.calibration import calibrate  # noqa: E402

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results_clean", "discrimination_per_case.csv"
)

ENCODERS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-base",
    "google/muril-base-cased",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]
LANGUAGES = ["en", "hi", "hinglish"]


def load_scores() -> dict[tuple[str, str], tuple[list[float], list[float]]]:
    """(encoder, language) -> (correct_scores, wrong_hard_scores), from
    the english_gold + full_sentence slice -- the same slice
    CALIBRATION_TABLE's shipped cells were fit from (see
    calibration.py's module docstring)."""
    by_cell: dict[tuple[str, str], tuple[list[float], list[float]]] = {
        (e, lang): ([], []) for e in ENCODERS for lang in LANGUAGES
    }
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["gold_mode"] != "english_gold" or row["gold_length"] != "full_sentence":
                continue
            key = (row["encoder"], row["variant"])
            if key not in by_cell:
                continue
            sim = float(row["cosine_similarity"])
            if row["label"] == "correct":
                by_cell[key][0].append(sim)
            elif row["label"] == "wrong_hard":
                by_cell[key][1].append(sim)
    return by_cell


def main() -> None:
    by_cell = load_scores()
    for encoder in ENCODERS:
        print(f'    "{encoder}": {{')
        for lang in LANGUAGES:
            correct, wrong = by_cell[(encoder, lang)]
            result = calibrate(correct, wrong)
            warnings_repr = repr(result.warnings) if result.warnings else "()"
            print(
                f'        "{lang}": CalibratedThreshold('
                f"{result.threshold}, {result.accuracy_at_threshold}, "
                f"{result.accuracy_at_default}, n_cases={result.n_cases}, "
                f"warnings={warnings_repr}),"
            )
        print("    },")


if __name__ == "__main__":
    main()
