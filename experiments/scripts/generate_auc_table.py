"""
Regenerate the calibrated_similarity AUC table in README.md directly from
experiments/results_clean/discrimination_summary.csv.

Written after an independent outside review caught that the README's
"AUC (threshold-free)" column was, in every one of 15 rows, actually a
copy of the calibrated-threshold column -- a hand-transcription error,
not a code bug (similarity.py/calibration.py were never wrong). The
prose built on that wrong column was wrong too: it named a chance-level
cell (e5-base/hi, real AUC 0.500) as the strongest signal, and called a
perfectly-inverted encoder (LaBSE/en, real AUC 0.070) "barely above
chance". Never hand-transcribe this table again -- run this script and
paste its output.

Usage: python experiments/scripts/generate_auc_table.py
"""

from __future__ import annotations

import csv
import os

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results_clean", "discrimination_summary.csv"
)

ENCODER_DISPLAY = {
    "intfloat/multilingual-e5-base": "multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "paraphrase-multilingual-mpnet-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "sentence-transformers/LaBSE": "LaBSE",
    "google/muril-base-cased": "muril-base-cased",
}


def load_rows() -> list[dict[str, str]]:
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # The 15 shipped cells: english_gold + full_sentence, one row per
    # (encoder, variant/language). english_gold is discrimination.py's
    # own "primary config for the headline chart" (see its comment at
    # the gold_mode, gold_length = "english_gold", "full_sentence"
    # assignment) -- the gold answer is in English regardless of
    # question language, so it's comparable across variants. This is
    # also the slice experiments/README.md's Milestone 3 section reports.
    return [
        r for r in rows if r["gold_mode"] == "english_gold" and r["gold_length"] == "full_sentence"
    ]


def main() -> None:
    rows = load_rows()
    if len(rows) != 15:
        raise SystemExit(f"expected 15 shipped cells, got {len(rows)} -- check the filter")

    table = []
    for r in rows:
        auc = float(r["roc_auc_hard"])
        table.append(
            {
                "encoder": ENCODER_DISPLAY[r["encoder"]],
                "language": r["variant"],
                "auc": auc,
                "acc_at_half": float(r["accuracy_at_0.5_hard"]),
                "threshold": float(r["best_threshold_hard"]),
                "acc_calibrated": float(r["best_accuracy_hard"]),
            }
        )

    table.sort(key=lambda x: -x["auc"])

    print(
        "| encoder                          | language | AUC (threshold-free) | "
        "accuracy @ 0.5 | calibrated threshold | accuracy @ calibrated |"
    )
    print(
        "|-----------------------------------|----------|----------------------:|"
        "----------------:|-----------:|------------------------:|"
    )
    for row in table:
        strong = row["auc"] >= 0.75
        weak = row["auc"] <= 0.55 or row["auc"] >= 0.95  # near-chance or MuRIL-style saturation
        auc_str = f"{row['auc']:.3f}"
        if strong:
            auc_str = f"**{auc_str}**"
        if weak:
            auc_str = f"{auc_str} ⚠️"
        print(
            f"| {row['encoder']:<35}| {row['language']:<8} | {auc_str:<22}| "
            f"{row['acc_at_half']:.3f}           | {row['threshold']:.3f}      | "
            f"{row['acc_calibrated']:.3f}                   |"
        )

    n_above_chance = sum(1 for r in table if r["auc"] > 0.5)
    n_strong = sum(1 for r in table if r["auc"] >= 0.75)
    mean_auc = sum(r["auc"] for r in table) / len(table)
    n_below_half = sum(1 for r in table if r["auc"] < 0.5)
    print()
    print(f"mean AUC across {len(table)} cells: {mean_auc:.3f}")
    print(f"cells with AUC > 0.5: {n_above_chance}")
    print(f"cells with AUC >= 0.75 (bold-worthy): {n_strong}")
    print(f"cells with AUC < 0.5 (inverted): {n_below_half}")


if __name__ == "__main__":
    main()
