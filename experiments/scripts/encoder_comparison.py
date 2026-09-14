"""
Rerun the EN/HI/Hinglish similarity comparison across multiple multilingual
sentence encoders, to check whether the score collapse seen with
all-MiniLM-L6-v2 (an English-only encoder) is an encoder-choice artifact or
a problem that survives with multilingual encoders too.

Reuses the exact 30 cases and model answers from DATASET_PATH (defaults to
the pinned data/results_clean.json). Does NOT call any LLM and does NOT
regenerate answers.

Run:  python experiments/scripts/encoder_comparison.py [--data PATH]
Output:
  results_clean/encoder_comparison.csv   (per encoder x variant x case)
  results_clean/encoder_summary.csv      (aggregate table)
  results_clean/encoder_comparison.png   (bar chart)
  summary table printed to stdout

This script only computes numbers. It does not interpret them.
"""
import os
import sys
import json
import random
import logging

import numpy as np

# ---------------------------------------------------------------------------
# Determinism
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

# Use the default HF cache (~/.cache/huggingface) so models persist across runs.
# Do not refetch if already cached: if every encoder below is already present
# in the local cache, force fully offline mode so no network calls are made
# at all. If any encoder is missing, stay online just long enough to fetch it
# (subsequent runs will then be fully offline).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _hf_cache_has(model_name):
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
    except Exception:
        return False
    repo_id = model_name
    return any(r.repo_id == repo_id for r in cache.repos)

sys.path.insert(0, EXPERIMENTS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("encoder_comparison")

from testcases import build_cases  # noqa: E402

# ---------------------------------------------------------------------------
# Encoders under test
# ---------------------------------------------------------------------------
ENCODERS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-base",
    "google/muril-base-cased",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
VARIANTS = ["en", "hi", "hinglish"]

# multilingual-e5 models require "query: " / "passage: " prefixes on inputs.
E5_MODELS = {"intfloat/multilingual-e5-base"}

# MuRIL is a plain BERT encoder (transformers), not a sentence-transformers
# model -- needs manual mean pooling with attention masking, not CLS.
RAW_TRANSFORMER_MODELS = {"google/muril-base-cased"}


# ---------------------------------------------------------------------------
# Encoder wrapper: uniform .encode(list[str]) -> np.ndarray [n, dim]
# ---------------------------------------------------------------------------
class EncoderWrapper:
    def __init__(self, model_name):
        self.model_name = model_name
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None

    def load(self):
        if self.model_name in RAW_TRANSFORMER_MODELS:
            from transformers import AutoTokenizer, AutoModel
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.eval()
        else:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
        return self

    def _prep(self, texts, is_query):
        if self.model_name in E5_MODELS:
            prefix = "query: " if is_query else "passage: "
            return [f"{prefix}{t}" for t in texts]
        return list(texts)

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode(self, texts, is_query=True):
        texts = self._prep(texts, is_query)
        if self.model_name in RAW_TRANSFORMER_MODELS:
            with torch.no_grad():
                enc = self._hf_tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                )
                out = self._hf_model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                return pooled.cpu().numpy()
        else:
            emb = self._st_model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Load answers from an existing dataset file (do not regenerate)
# ---------------------------------------------------------------------------
def load_answers(dataset_path=None):
    path = dataset_path or DATASET_PATH
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    # Rows that failed script adherence after every retry attempt are kept
    # in results_clean.json (for the record) but excluded from scoring --
    # see scripts/verify_clean_dataset.py, which asserts this invariant.
    return {
        r["case_id"]: r["answer"]
        for r in rows
        if not r.get("script_adherence_failure", False)
    }


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


def main(dataset_path=None):
    _set_offline_mode_if_all_cached()
    cases = build_cases()
    answers_by_case = load_answers(dataset_path)

    missing = [c["case_id"] for c in cases if c["case_id"] not in answers_by_case]
    if missing:
        log.warning("No stored answer for %d cases (skipping): %s", len(missing), missing)

    per_case_rows = []   # -> encoder_comparison.csv
    summary_rows = []    # -> encoder_summary.csv

    for model_name in ENCODERS:
        log.info("Loading encoder: %s", model_name)
        try:
            enc = EncoderWrapper(model_name).load()
        except Exception as e:
            log.error("Failed to load %s: %s -- skipping this encoder.", model_name, e)
            continue

        for gold_mode in ["english_gold", "same_language_gold"]:
            # Collect per-variant lists for aggregate stats.
            by_variant_scores = {v: [] for v in VARIANTS}

            for c in cases:
                if c["case_id"] not in answers_by_case:
                    continue
                answer = answers_by_case[c["case_id"]]
                if not answer:
                    log.warning("Empty answer for %s -- skipping case.", c["case_id"])
                    continue

                gold_text = c["gold"] if gold_mode == "english_gold" else c["gold_same_lang"]

                try:
                    # is_query=True for the model answer (the thing being scored),
                    # is_query=False for the gold reference (the "passage"/reference)
                    # -- only affects e5 prefixing; harmless no-op for other models.
                    ans_emb = enc.encode([answer], is_query=True)[0]
                    gold_emb = enc.encode([gold_text], is_query=False)[0]
                    sim = cosine(ans_emb, gold_emb)
                except Exception as e:
                    log.error("Scoring failed for %s / %s / %s: %s",
                              model_name, gold_mode, c["case_id"], e)
                    continue

                by_variant_scores[c["variant"]].append(sim)

                per_case_rows.append({
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "variant": c["variant"],
                    "case_id": c["case_id"],
                    "task_id": c["task_id"],
                    "gold_text": gold_text,
                    "answer": answer,
                    "cosine_similarity": sim,
                })

            # Aggregate per variant for this encoder x gold_mode.
            means = {}
            for v in VARIANTS:
                scores = by_variant_scores[v]
                if not scores:
                    means[v] = None
                    continue
                arr = np.array(scores, dtype=float)
                row = {
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "variant": v,
                    "n": len(arr),
                    "mean_cosine_similarity": float(arr.mean()),
                    "std_cosine_similarity": float(arr.std(ddof=0)),
                }
                for thr in THRESHOLDS:
                    row[f"pass_at_{thr}"] = int((arr >= thr).sum())
                summary_rows.append(row)
                means[v] = float(arr.mean())

            # Gaps: EN mean - HI mean, EN mean - Hinglish mean.
            if means.get("en") is not None:
                for v in ["hi", "hinglish"]:
                    if means.get(v) is not None:
                        gap_row = {
                            "encoder": model_name,
                            "gold_mode": gold_mode,
                            "variant": f"gap_en_minus_{v}",
                            "n": None,
                            "mean_cosine_similarity": means["en"] - means[v],
                            "std_cosine_similarity": None,
                        }
                        for thr in THRESHOLDS:
                            gap_row[f"pass_at_{thr}"] = None
                        summary_rows.append(gap_row)

        # Free memory before loading the next (possibly large) encoder.
        del enc
        if torch is not None:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -----------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------
    import csv

    per_case_path = os.path.join(RESULTS_DIR, "encoder_comparison.csv")
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        if per_case_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_case_rows)
    log.info("Wrote %s (%d rows)", per_case_path, len(per_case_rows))

    summary_path = os.path.join(RESULTS_DIR, "encoder_summary.csv")
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

    # -----------------------------------------------------------------
    # Print summary table to stdout
    # -----------------------------------------------------------------
    print_summary(summary_rows)

    # -----------------------------------------------------------------
    # Bar chart: mean similarity by encoder, grouped by variant
    # -----------------------------------------------------------------
    make_chart(summary_rows)


def print_summary(summary_rows):
    print("\n" + "=" * 100)
    print("ENCODER COMPARISON SUMMARY (numbers only, no interpretation)")
    print("=" * 100)
    for gold_mode in ["english_gold", "same_language_gold"]:
        print(f"\n--- gold_mode = {gold_mode} ---")
        hdr = f"{'encoder':52s} {'variant':10s} {'n':>4s} {'mean':>7s} {'std':>7s}"
        for thr in THRESHOLDS:
            hdr += f" {'p>=' + str(thr):>8s}"
        print(hdr)
        print("-" * len(hdr))
        for r in summary_rows:
            if r["gold_mode"] != gold_mode:
                continue
            if r["variant"].startswith("gap_"):
                continue
            line = (f"{r['encoder']:52s} {r['variant']:10s} {r['n']:>4d} "
                    f"{r['mean_cosine_similarity']:>7.3f} {r['std_cosine_similarity']:>7.3f}")
            for thr in THRESHOLDS:
                v = r.get(f"pass_at_{thr}")
                line += f" {v if v is not None else '-':>8}"
            print(line)
        print()
        print(f"  Gaps (EN mean - other mean), gold_mode={gold_mode}:")
        for r in summary_rows:
            if r["gold_mode"] != gold_mode:
                continue
            if r["variant"].startswith("gap_"):
                print(f"    {r['encoder']:52s} {r['variant']:20s} {r['mean_cosine_similarity']:.3f}")


def make_chart(summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One chart, gold_mode=english_gold (primary comparison from the original
    # experiment); encoders on x-axis, grouped bars per variant.
    gold_mode = "english_gold"
    rows = [r for r in summary_rows
            if r["gold_mode"] == gold_mode and not r["variant"].startswith("gap_")]

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

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, v in enumerate(VARIANTS):
        means = []
        stds = []
        for enc_name in encoders:
            match = [r for r in rows if r["encoder"] == enc_name and r["variant"] == v]
            if match:
                means.append(match[0]["mean_cosine_similarity"])
                stds.append(match[0]["std_cosine_similarity"])
            else:
                means.append(0)
                stds.append(0)
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=v, color=variant_colors.get(v))

    ax.set_xlabel("Encoder")
    ax.set_ylabel("Mean cosine similarity vs English gold")
    ax.set_title("Mean similarity by encoder, grouped by language variant\n"
                  "(gold_mode = english_gold)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=20, ha="right")
    ax.legend(title="variant")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "encoder_comparison.png")
    fig.savefig(chart_path, dpi=150)
    log.info("Wrote %s", chart_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="dataset_path", default=None,
                         help=f"Path to the answers JSON (default: {DATASET_PATH})")
    args = parser.parse_args()
    main(args.dataset_path)
