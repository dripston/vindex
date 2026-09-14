"""
Part 6: before/after comparison of the contamination fix.

This script's whole job is the before/after diff, so it legitimately reads
data derived from the contaminated results.json on purpose -- see
Correction 1 in ARCHITECTURE.md. It doesn't open results.json itself, but
OLD_SUMMARY_PATH below is a CSV computed FROM it (by discrimination.py, in
an earlier run), which is the same thing one level removed.

Reads the OLD (contaminated results.json -> results/discrimination_summary.csv)
and NEW (clean data/results_clean.json -> results_clean/discrimination_summary.csv)
summary files already on disk -- NO encoder re-run, pure CSV arithmetic, since
every metric needed is already a column in discrimination_summary.csv.

Produces one row per (encoder, gold_mode, gold_length, variant) with:
  mean_similarity, roc_auc_hard, roc_auc_subtle, separation,
  accuracy_at_0.5_hard, best_accuracy_hard
each as _old, _new, _delta (new - old).

Also renders results_clean/before_after_auc.png: grouped bars, old vs new
ROC AUC (hard negatives), one panel per variant, encoders on the x-axis
(fixed at the english_gold x full_sentence config for the chart, since a
2D grid of gold_mode x gold_length x variant panels would be unreadable).

Runnable standalone (assuming both discrimination_summary.csv files exist,
which they do from prior runs):
  python experiments/scripts/contamination_impact.py

This script only computes numbers. It does not interpret them.
"""
import os
import sys
import csv
import logging

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD_SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "discrimination_summary.csv")
NEW_SUMMARY_PATH = os.path.join(REPO_ROOT, "results_clean", "discrimination_summary.csv")
OUTPUT_CSV_PATH = os.path.join(REPO_ROOT, "results_clean", "contamination_impact.csv")
OUTPUT_CHART_PATH = os.path.join(REPO_ROOT, "results_clean", "before_after_auc.png")

VARIANTS = ["en", "hi", "hinglish"]
GOLD_MODES = ["english_gold", "same_language_gold"]
GOLD_LENGTHS = ["short", "full_sentence"]

# Metrics compared: (output_name, source csv column name)
METRICS = [
    ("mean_similarity", "mean_correct"),
    ("roc_auc_hard", "roc_auc_hard"),
    ("roc_auc_subtle", "roc_auc_subtle"),
    ("separation", "separation"),
    ("accuracy_at_0.5_hard", "accuracy_at_0.5_hard"),
    ("best_accuracy_hard", "best_accuracy_hard"),
]

# Chart uses this single config (register-matched full-sentence gold,
# English reference) since a chart can't show the full 4-way grid legibly.
CHART_GOLD_MODE = "english_gold"
CHART_GOLD_LENGTH = "full_sentence"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("contamination_impact")


def _read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def main():
    if not os.path.exists(OLD_SUMMARY_PATH):
        raise FileNotFoundError(
            f"{OLD_SUMMARY_PATH} not found -- discrimination.py must have already "
            f"been run on the contaminated dataset."
        )
    if not os.path.exists(NEW_SUMMARY_PATH):
        raise FileNotFoundError(
            f"{NEW_SUMMARY_PATH} not found -- discrimination.py must have already "
            f"been run on results_clean.json."
        )

    old_rows = _read_csv_rows(OLD_SUMMARY_PATH)
    new_rows = _read_csv_rows(NEW_SUMMARY_PATH)

    def find_row(rows, encoder, gold_mode, gold_length, variant):
        for r in rows:
            if (r["encoder"] == encoder and r["gold_mode"] == gold_mode
                    and r["gold_length"] == gold_length and r["variant"] == variant):
                return r
        return None

    encoders = []
    seen = set()
    for r in old_rows:
        if r["encoder"] not in seen:
            encoders.append(r["encoder"])
            seen.add(r["encoder"])

    comparison_rows = []
    for encoder in encoders:
        for gold_mode in GOLD_MODES:
            for gold_length in GOLD_LENGTHS:
                for variant in VARIANTS:
                    old_r = find_row(old_rows, encoder, gold_mode, gold_length, variant)
                    new_r = find_row(new_rows, encoder, gold_mode, gold_length, variant)
                    if old_r is None or new_r is None:
                        log.warning(
                            "Missing row for encoder=%s gold_mode=%s gold_length=%s variant=%s "
                            "(old=%s, new=%s) -- skipping.",
                            encoder, gold_mode, gold_length, variant,
                            old_r is not None, new_r is not None,
                        )
                        continue

                    row = {
                        "encoder": encoder,
                        "gold_mode": gold_mode,
                        "gold_length": gold_length,
                        "variant": variant,
                    }
                    max_abs_delta = 0.0
                    for out_name, csv_field in METRICS:
                        old_v = _to_float(old_r.get(csv_field))
                        new_v = _to_float(new_r.get(csv_field))
                        delta = (new_v - old_v) if (old_v is not None and new_v is not None) else None
                        row[f"{out_name}_old"] = old_v
                        row[f"{out_name}_new"] = new_v
                        row[f"{out_name}_delta"] = delta
                        if delta is not None:
                            max_abs_delta = max(max_abs_delta, abs(delta))
                    row["_max_abs_delta"] = max_abs_delta  # for sorting only, not written below

                    comparison_rows.append(row)

    # Write CSV (drop the internal sort-helper field).
    csv_rows = [{k: v for k, v in r.items() if k != "_max_abs_delta"} for r in comparison_rows]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    log.info("Wrote %s (%d rows)", OUTPUT_CSV_PATH, len(csv_rows))

    print_top_changed(comparison_rows, n=15)
    print_hinglish_rows(comparison_rows)
    make_chart(old_rows, new_rows, encoders)


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "  -  "


def _print_row_table(rows, title):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)
    hdr = (f"{'encoder':46s} {'gold_mode':19s} {'gold_len':13s} {'var':9s} "
           f"{'sim_old':>8s} {'sim_new':>8s} {'d_sim':>8s} "
           f"{'aucH_old':>9s} {'aucH_new':>9s} {'d_aucH':>8s} "
           f"{'aucS_old':>9s} {'aucS_new':>9s} {'d_aucS':>8s} "
           f"{'sep_old':>8s} {'sep_new':>8s} {'d_sep':>8s} "
           f"{'acc.5_old':>9s} {'acc.5_new':>9s} {'d_acc.5':>8s} "
           f"{'bestAcc_old':>11s} {'bestAcc_new':>11s} {'d_bestAcc':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['encoder']:46s} {r['gold_mode']:19s} {r['gold_length']:13s} {r['variant']:9s} "
              f"{_fmt(r['mean_similarity_old']):>8s} {_fmt(r['mean_similarity_new']):>8s} {_fmt(r['mean_similarity_delta']):>8s} "
              f"{_fmt(r['roc_auc_hard_old']):>9s} {_fmt(r['roc_auc_hard_new']):>9s} {_fmt(r['roc_auc_hard_delta']):>8s} "
              f"{_fmt(r['roc_auc_subtle_old']):>9s} {_fmt(r['roc_auc_subtle_new']):>9s} {_fmt(r['roc_auc_subtle_delta']):>8s} "
              f"{_fmt(r['separation_old']):>8s} {_fmt(r['separation_new']):>8s} {_fmt(r['separation_delta']):>8s} "
              f"{_fmt(r['accuracy_at_0.5_hard_old']):>9s} {_fmt(r['accuracy_at_0.5_hard_new']):>9s} {_fmt(r['accuracy_at_0.5_hard_delta']):>8s} "
              f"{_fmt(r['best_accuracy_hard_old']):>11s} {_fmt(r['best_accuracy_hard_new']):>11s} {_fmt(r['best_accuracy_hard_delta']):>10s}")


def print_top_changed(rows, n=15):
    ranked = sorted(rows, key=lambda r: r["_max_abs_delta"], reverse=True)
    _print_row_table(
        ranked[:n],
        f"TOP {n} ROWS BY LARGEST ABSOLUTE DELTA (any metric), DESCENDING",
    )


def print_hinglish_rows(rows):
    hinglish_rows = [r for r in rows if r["variant"] == "hinglish"]
    hinglish_rows.sort(key=lambda r: (r["encoder"], r["gold_mode"], r["gold_length"]))
    _print_row_table(hinglish_rows, "ALL HINGLISH-VARIANT ROWS (the contaminated column)")


def make_chart(old_rows, new_rows, encoders):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def find_row(rows, encoder, variant):
        for r in rows:
            if (r["encoder"] == encoder and r["variant"] == variant
                    and r["gold_mode"] == CHART_GOLD_MODE and r["gold_length"] == CHART_GOLD_LENGTH):
                return r
        return None

    short_names = [e.split("/")[-1] for e in encoders]
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(6 * len(VARIANTS), 6), sharey=True)
    if len(VARIANTS) == 1:
        axes = [axes]

    x = np.arange(len(encoders))
    width = 0.35

    for ax, v in zip(axes, VARIANTS):
        old_vals, new_vals = [], []
        for enc in encoders:
            old_r = find_row(old_rows, enc, v)
            new_r = find_row(new_rows, enc, v)
            old_vals.append(_to_float(old_r["roc_auc_hard"]) or 0 if old_r else 0)
            new_vals.append(_to_float(new_r["roc_auc_hard"]) or 0 if new_r else 0)
        ax.bar(x - width / 2, old_vals, width, label="old (contaminated)", color="#C44E52")
        ax.bar(x + width / 2, new_vals, width, label="new (clean)", color="#55A868")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"variant = {v}")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Encoder")

    axes[0].set_ylabel("ROC AUC (correct vs wrong_hard)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Before vs after contamination fix: ROC AUC (hard negatives)\n"
                 f"(gold_mode={CHART_GOLD_MODE}, gold_length={CHART_GOLD_LENGTH})")
    fig.tight_layout()

    fig.savefig(OUTPUT_CHART_PATH, dpi=150)
    log.info("Wrote %s", OUTPUT_CHART_PATH)


if __name__ == "__main__":
    main()
