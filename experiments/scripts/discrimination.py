"""
Discrimination test: can each encoder tell a CORRECT answer apart from a
WRONG one, rather than just scoring correct answers "high in isolation"
(which is all encoder_comparison.py could show)?

For each of the 10 tasks x 3 language variants, three answers are scored:
  - correct       : the real model answer, loaded from DATASET_PATH
  - wrong_subtle  : hand-authored, right topic/entity, one detail wrong
  - wrong_hard    : hand-authored, plainly wrong (different entity/fact)

Each is scored against 4 gold configurations:
  gold_mode   in {english_gold, same_language_gold}
  gold_length in {short, full_sentence}

giving a 5 (encoder) x 2 (gold_mode) x 2 (gold_length) x 3 (variant) x 3
(label) grid. For each (encoder, gold_mode, gold_length, variant) cell we
report mean/std per label, ROC AUC (correct vs wrong_hard, correct vs
wrong_subtle), best-accuracy threshold, and accuracy at threshold 0.5.

Reuses the same EncoderWrapper, prefixing, pooling, seeding, and caching
logic as encoder_comparison.py. Does NOT call an LLM, does NOT regenerate
the 30 real model answers, and does NOT modify the dataset file it reads.

Run:  python experiments/scripts/discrimination.py [--data PATH]
Output:
  results_clean/discrimination_per_case.csv
  results_clean/discrimination_summary.csv
  results_clean/auc_by_encoder.png
  results_clean/score_distributions.png
  summary table printed to stdout

This script only computes numbers. It does not interpret them.
"""
import os
import sys
import csv
import random
import logging
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Determinism (identical setup to encoder_comparison.py)
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    torch = None

# ---------------------------------------------------------------------------
# Paths / cache
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results_clean")
DATASET_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("discrimination")

from testcases import build_discrimination_cases  # noqa: E402

# Reuse the encoder wrapper, prefixing, and pooling logic verbatim.
from encoder_comparison import (  # noqa: E402
    EncoderWrapper, cosine, ENCODERS, _hf_cache_has,
)

GOLD_MODES = ["english_gold", "same_language_gold"]
GOLD_LENGTHS = ["short", "full_sentence"]
VARIANTS = ["en", "hi", "hinglish"]
LABELS = ["correct", "wrong_subtle", "wrong_hard"]


def _set_offline_mode_if_all_cached():
    all_cached = all(_hf_cache_has(m) for m in ENCODERS)
    if all_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log.info("All %d encoders already cached -- running fully offline.", len(ENCODERS))
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        log.info("Not all encoders cached yet -- allowing network for first-time downloads.")


def roc_auc(pos_scores, neg_scores):
    """ROC AUC for separating pos_scores (label=1, 'correct') from
    neg_scores (label=0, e.g. 'wrong_hard'), via sklearn. 0.5 = guessing,
    1.0 = perfect separation, by similarity score as the decision variable
    (higher score => more likely 'correct')."""
    from sklearn.metrics import roc_auc_score
    if not pos_scores or not neg_scores:
        return None
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_score = list(pos_scores) + list(neg_scores)
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        # e.g. all scores identical -> undefined AUC
        return None


def best_threshold_accuracy(pos_scores, neg_scores):
    """Sweep every score as a candidate threshold (predict 'correct' iff
    score >= threshold) and return the threshold that maximizes accuracy,
    plus that accuracy. Ties broken by the lowest such threshold."""
    if not pos_scores or not neg_scores:
        return None, None
    all_scores = sorted(set(pos_scores) | set(neg_scores))
    # Candidate thresholds: midpoints between consecutive sorted scores,
    # plus below-min and above-max, so every possible split is considered.
    candidates = [all_scores[0] - 1e-6]
    for i in range(len(all_scores) - 1):
        candidates.append((all_scores[i] + all_scores[i + 1]) / 2.0)
    candidates.append(all_scores[-1] + 1e-6)

    n = len(pos_scores) + len(neg_scores)
    best_acc = -1.0
    best_thr = None
    for thr in candidates:
        correct = sum(1 for s in pos_scores if s >= thr) + sum(1 for s in neg_scores if s < thr)
        acc = correct / n
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def accuracy_at_threshold(pos_scores, neg_scores, thr=0.5):
    if not pos_scores or not neg_scores:
        return None
    n = len(pos_scores) + len(neg_scores)
    correct = sum(1 for s in pos_scores if s >= thr) + sum(1 for s in neg_scores if s < thr)
    return correct / n


def main(dataset_path=None):
    _set_offline_mode_if_all_cached()
    cases = build_discrimination_cases(dataset_path or DATASET_PATH)

    skipped = [c["case_id"] for c in cases if not c["answer"]]
    if skipped:
        log.warning("Empty answer for %d case(s) -- skipping: %s", len(skipped), skipped)
    cases = [c for c in cases if c["answer"]]

    per_case_rows = []   # -> discrimination_per_case.csv
    # score_store[(encoder, gold_mode, gold_length, variant, label)] = [similarities]
    score_store = {}

    for model_name in ENCODERS:
        log.info("Loading encoder: %s", model_name)
        try:
            enc = EncoderWrapper(model_name).load()
        except Exception as e:
            log.error("Failed to load %s: %s -- skipping this encoder.", model_name, e)
            continue

        for gold_mode, gold_length in product(GOLD_MODES, GOLD_LENGTHS):
            for c in cases:
                gold_text = c["golds"][(gold_mode, gold_length)]
                answer = c["answer"]
                try:
                    ans_emb = enc.encode([answer], is_query=True)[0]
                    gold_emb = enc.encode([gold_text], is_query=False)[0]
                    sim = cosine(ans_emb, gold_emb)
                except Exception as e:
                    log.error("Scoring failed for %s / %s / %s / %s: %s",
                              model_name, gold_mode, gold_length, c["case_id"], e)
                    continue

                key = (model_name, gold_mode, gold_length, c["variant"], c["label"])
                score_store.setdefault(key, []).append(sim)

                per_case_rows.append({
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "gold_length": gold_length,
                    "variant": c["variant"],
                    "label": c["label"],
                    "case_id": c["case_id"],
                    "task_id": c["task_id"],
                    "gold_text": gold_text,
                    "answer": answer,
                    "cosine_similarity": sim,
                })

        del enc
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Aggregate: one row per (encoder, gold_mode, gold_length, variant)
    # -----------------------------------------------------------------
    summary_rows = []
    for model_name in ENCODERS:
        for gold_mode, gold_length in product(GOLD_MODES, GOLD_LENGTHS):
            for v in VARIANTS:
                by_label = {}
                for label in LABELS:
                    key = (model_name, gold_mode, gold_length, v, label)
                    by_label[label] = score_store.get(key, [])

                if not any(by_label.values()):
                    continue

                row = {
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "gold_length": gold_length,
                    "variant": v,
                }
                for label in LABELS:
                    scores = by_label[label]
                    row[f"n_{label}"] = len(scores)
                    row[f"mean_{label}"] = float(np.mean(scores)) if scores else None
                    row[f"std_{label}"] = float(np.std(scores, ddof=0)) if scores else None

                correct_scores = by_label["correct"]
                hard_scores = by_label["wrong_hard"]
                subtle_scores = by_label["wrong_subtle"]

                row["separation"] = (
                    row["mean_correct"] - row["mean_wrong_hard"]
                    if correct_scores and hard_scores else None
                )
                row["separation_subtle"] = (
                    row["mean_correct"] - row["mean_wrong_subtle"]
                    if correct_scores and subtle_scores else None
                )

                row["roc_auc_hard"] = roc_auc(correct_scores, hard_scores)
                row["roc_auc_subtle"] = roc_auc(correct_scores, subtle_scores)

                best_thr, best_acc = best_threshold_accuracy(correct_scores, hard_scores)
                row["best_threshold_hard"] = best_thr
                row["best_accuracy_hard"] = best_acc
                row["accuracy_at_0.5_hard"] = accuracy_at_threshold(correct_scores, hard_scores, 0.5)

                summary_rows.append(row)

    # -----------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------
    per_case_path = os.path.join(RESULTS_DIR, "discrimination_per_case.csv")
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        if per_case_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_case_rows)
    log.info("Wrote %s (%d rows)", per_case_path, len(per_case_rows))

    summary_path = os.path.join(RESULTS_DIR, "discrimination_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            for r in summary_rows:
                for k in fieldnames:
                    r.setdefault(k, None)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    log.info("Wrote %s (%d rows)", summary_path, len(summary_rows))

    print_summary(summary_rows)
    make_auc_chart(summary_rows)
    make_distribution_chart(per_case_rows)


def print_summary(summary_rows):
    print("\n" + "=" * 150)
    print("DISCRIMINATION SUMMARY (numbers only, no interpretation)")
    print("=" * 150)
    hdr = (f"{'encoder':46s} {'gold_mode':19s} {'gold_len':13s} {'var':9s} "
           f"{'mean_correct':>12s} {'mean_subtle':>12s} {'mean_hard':>10s} "
           f"{'sep':>7s} {'sep_sub':>8s} {'auc_hard':>9s} {'auc_subtle':>10s} "
           f"{'best_thr':>9s} {'best_acc':>9s} {'acc@0.5':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        def f3(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{r['encoder']:46s} {r['gold_mode']:19s} {r['gold_length']:13s} {r['variant']:9s} "
              f"{f3(r['mean_correct']):>12s} {f3(r['mean_wrong_subtle']):>12s} {f3(r['mean_wrong_hard']):>10s} "
              f"{f3(r['separation']):>7s} {f3(r['separation_subtle']):>8s} "
              f"{f3(r['roc_auc_hard']):>9s} {f3(r['roc_auc_subtle']):>10s} "
              f"{f3(r['best_threshold_hard']):>9s} {f3(r['best_accuracy_hard']):>9s} "
              f"{f3(r['accuracy_at_0.5_hard']):>8s}")


def make_auc_chart(summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Primary config for the headline chart: english_gold x full_sentence
    # (full-sentence gold matches model-answer register; english_gold is
    # the cross-script comparison the earlier experiment used).
    gold_mode, gold_length = "english_gold", "full_sentence"
    rows = [r for r in summary_rows if r["gold_mode"] == gold_mode and r["gold_length"] == gold_length]

    encoders = []
    seen = set()
    for r in rows:
        if r["encoder"] not in seen:
            encoders.append(r["encoder"])
            seen.add(r["encoder"])
    short_names = [e.split("/")[-1] for e in encoders]
    variant_colors = {"en": "#4C72B0", "hi": "#DD8452", "hinglish": "#55A868"}

    x = np.arange(len(encoders))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, auc_key, title in [
        (axes[0], "roc_auc_hard", "ROC AUC: correct vs wrong_hard"),
        (axes[1], "roc_auc_subtle", "ROC AUC: correct vs wrong_subtle"),
    ]:
        for i, v in enumerate(VARIANTS):
            vals = []
            for enc_name in encoders:
                match = [r for r in rows if r["encoder"] == enc_name and r["variant"] == v]
                vals.append(match[0][auc_key] if match and match[0][auc_key] is not None else 0)
            offset = (i - 1) * width
            ax.bar(x + offset, vals, width, label=v, color=variant_colors.get(v))
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Encoder")
    axes[0].set_ylabel("ROC AUC")
    axes[0].legend(title="variant")

    fig.suptitle(f"ROC AUC by encoder, grouped by variant\n(gold_mode={gold_mode}, gold_length={gold_length})")
    fig.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "auc_by_encoder.png")
    fig.savefig(chart_path, dpi=150)
    log.info("Wrote %s", chart_path)


def make_distribution_chart(per_case_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gold_mode, gold_length = "english_gold", "full_sentence"
    rows = [r for r in per_case_rows if r["gold_mode"] == gold_mode and r["gold_length"] == gold_length]

    encoders = []
    seen = set()
    for r in rows:
        if r["encoder"] not in seen:
            encoders.append(r["encoder"])
            seen.add(r["encoder"])

    label_colors = {"correct": "#4C72B0", "wrong_subtle": "#DD8452", "wrong_hard": "#C44E52"}
    label_order = ["wrong_hard", "wrong_subtle", "correct"]

    fig, axes = plt.subplots(1, len(encoders), figsize=(5 * len(encoders), 5), sharey=True)
    if len(encoders) == 1:
        axes = [axes]

    for ax, enc_name in zip(axes, encoders):
        for j, label in enumerate(label_order):
            vals = [r["cosine_similarity"] for r in rows if r["encoder"] == enc_name and r["label"] == label]
            if not vals:
                continue
            y = np.random.RandomState(SEED).normal(loc=j, scale=0.06, size=len(vals))
            ax.scatter(vals, y, alpha=0.6, s=22, color=label_colors[label], label=label)
        ax.set_yticks(range(len(label_order)))
        ax.set_yticklabels(label_order)
        ax.set_xlabel("Cosine similarity")
        ax.set_title(enc_name.split("/")[-1])
        ax.set_xlim(-0.05, 1.05)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Score distributions: correct vs wrong_subtle vs wrong_hard\n"
                 f"(gold_mode={gold_mode}, gold_length={gold_length}, all variants pooled)")
    fig.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "score_distributions.png")
    fig.savefig(chart_path, dpi=150)
    log.info("Wrote %s", chart_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="dataset_path", default=None,
                         help=f"Path to the answers JSON (default: {DATASET_PATH})")
    args = parser.parse_args()
    main(args.dataset_path)
