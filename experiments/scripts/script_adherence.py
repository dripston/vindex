"""
Scaled script-adherence experiment: does an explicit script instruction
fix Hinglish/Romanized-Hindi script adherence, across multiple models and
multiple instruction strengths -- not just the 10-question, 1-model
signal from the regeneration run.

150 hand-authored prompts (script_adherence_cases.py) x 4 system-prompt
conditions x 3 Groq models = 1800 calls.

Reuses script_check.py's classify()/count_scripts() UNCHANGED (imported,
not reimplemented) to label every response and to decide adherence per
the rules in the module docstring below.

Checkpointed to disk after every call (results_adherence/checkpoint.jsonl,
append-only) so a crash loses nothing; --resume skips calls already in
the checkpoint.

This is a new, independent experiment. Does not read, write, or import
results_clean.json, testcases.py, or any file from the prior contamination
experiment.

Run:            python experiments/scripts/script_adherence.py
Resume/rerun:   python experiments/scripts/script_adherence.py --resume
                (also the default -- see main(): checkpointed calls are
                 always skipped whether or not --resume is passed, since
                 skipping is strictly safe; --resume exists per the task
                 spec as the documented resume invocation.)

Output:
  results_adherence/per_case.csv
  results_adherence/summary.csv
  results_adherence/adherence_by_condition.png
  results_adherence/script_confusion.png
  pivot table + headline contrast printed to stdout

This script only computes numbers. It does not interpret them.
"""
import io
import os
import sys
import re
import csv
import json
import time
import logging
import argparse
from collections import defaultdict

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, REPO_ROOT)          # script_check.py lives at repo root
sys.path.insert(0, EXPERIMENTS_DIR)    # script_adherence_cases.py lives in experiments/

from script_check import count_scripts, classify  # noqa: E402  -- reused verbatim
from script_adherence_cases import build_prompts  # noqa: E402

# --- load .env (same pattern as run_experiment.py) ---
ENV = {}
with open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, val = line.split("=", 1)
            ENV[k.strip()] = val.strip()
os.environ.setdefault("GROQ_API_KEY", ENV.get("GROQ_API_KEY", ""))

from groq import Groq  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("script_adherence")

RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results_adherence")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint.jsonl")
PER_CASE_CSV = os.path.join(RESULTS_DIR, "per_case.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
CHART_ADHERENCE = os.path.join(RESULTS_DIR, "adherence_by_condition.png")
CHART_CONFUSION = os.path.join(RESULTS_DIR, "script_confusion.png")

# ---------------------------------------------------------------------------
# Part 3: models
# ---------------------------------------------------------------------------
# Chosen after checking Groq's available chat models on this account:
#   - openai/gpt-oss-20b  (small)
#   - openai/gpt-oss-120b (large)
#   - qwen/qwen3.8-27b    (mid, different lineage)
# qwen/qwen3.6-27b was tried first and rejected: it emits its reasoning as
# visible <think>...</think> text that regularly exceeds 1000+ tokens, and
# this account's OTPM (output tokens/min) limit for that model is only
# 1000 -- incompatible with 1800 calls. qwen3.8-27b answers directly with
# finish_reason=stop at max_tokens=300 and shares the same 8000 tok/min
# limit as the other two models.
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

MAX_TOKENS = 300
TEMPERATURE = 0.0
SLEEP_BETWEEN_CALLS = 1.3  # seconds; paces well under the 8000 tok/min cap
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Part 2: instruction conditions
# ---------------------------------------------------------------------------
FEWSHOT_EXAMPLES = (
    "\n\nExample:\n"
    "User: Bharat ki rajdhani kya hai?\n"
    "Assistant: Bharat ki rajdhani New Delhi hai."
)

CONDITIONS = {
    "none": None,  # no system message at all
    "implicit": "Reply in the SAME language and script the user used.",
    "explicit": (
        "Reply in the SAME language as the user. If the user's message is written "
        "in Latin/Roman letters (Romanized Hindi or Hinglish), your reply MUST use "
        "ONLY Latin/Roman letters as well. Using Devanagari script (देवनागरी) in "
        "your reply to a Romanized message is FORBIDDEN."
    ),
    "fewshot": (
        "Reply in the SAME language as the user. If the user's message is written "
        "in Latin/Roman letters (Romanized Hindi or Hinglish), your reply MUST use "
        "ONLY Latin/Roman letters as well. Using Devanagari script (देवनागरी) in "
        "your reply to a Romanized message is FORBIDDEN." + FEWSHOT_EXAMPLES
    ),
}
CONDITION_NAMES = ["none", "implicit", "explicit", "fewshot"]

client = Groq(api_key=os.environ["GROQ_API_KEY"])

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text):
    """Some models (qwen) emit visible <think>...</think> reasoning in the
    content field. Classification must run on the final answer only, or
    English commentary inside the reasoning trace would contaminate the
    script label. No-op for models that don't emit these tags."""
    if not text:
        return text
    return THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Part 4: expected script per input form (rules given in the task)
# ---------------------------------------------------------------------------
def expected_scripts_for_form(input_form):
    if input_form == "devanagari":
        return {"devanagari", "mixed"}
    if input_form == "roman":
        return {"roman"}
    if input_form == "codemix":
        return {"roman", "mixed"}
    raise ValueError(f"unknown input_form: {input_form!r}")


def is_adherent(output_script, input_form):
    if output_script == "empty":
        return False
    return output_script in expected_scripts_for_form(input_form)


# ---------------------------------------------------------------------------
# API call with retry + explicit 429 backoff
# ---------------------------------------------------------------------------
def call_model(model, system_content, user_text):
    """Returns (raw_content_or_None, error_str_or_None, attempts_made)."""
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_text})

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            return (content or "").strip(), None, attempt + 1
        except Exception as e:
            last_err = str(e)
            is_rate_limit = "429" in last_err or "rate_limit" in last_err.lower()
            if is_rate_limit:
                # honor Retry-After if present in the error text, else back off harder
                m = re.search(r"try again in ([\d.]+)s", last_err, re.IGNORECASE)
                wait = float(m.group(1)) + 1.0 if m else 15 * (attempt + 1)
                log.warning("  [429] %s -- backing off %.1fs (attempt %d/%d)",
                            last_err[:150], wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
            else:
                log.warning("  [retry %d/%d] %s", attempt + 1, MAX_RETRIES, last_err[:200])
                time.sleep(5 * (attempt + 1))
    return None, last_err, MAX_RETRIES


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def make_call_id(model, condition, case_id, input_form):
    return f"{model}||{condition}||{case_id}||{input_form}"


def load_checkpoint():
    """Returns dict call_id -> row (parsed from the JSONL checkpoint)."""
    done = {}
    if not os.path.exists(CHECKPOINT_PATH):
        return done
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["call_id"]] = row
    return done


def append_checkpoint(row):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main data collection
# ---------------------------------------------------------------------------
def run_all_calls():
    prompts = build_prompts()  # 150
    done = load_checkpoint()
    log.info("Loaded %d already-completed calls from checkpoint.", len(done))

    total_planned = len(MODELS) * len(CONDITION_NAMES) * len(prompts)
    log.info("Total planned calls: %d (%d models x %d conditions x %d prompts)",
              total_planned, len(MODELS), len(CONDITION_NAMES), len(prompts))

    n_done_this_run = 0
    n_skipped = 0
    n_empty = 0
    n_error = 0
    empty_log = []

    i = 0
    for model in MODELS:
        for condition in CONDITION_NAMES:
            system_content = CONDITIONS[condition]
            for p in prompts:
                i += 1
                call_id = make_call_id(model, condition, p["case_id"], p["input_form"])

                if call_id in done:
                    n_skipped += 1
                    continue

                raw, err, attempts = call_model(model, system_content, p["text"])
                cleaned = strip_think_blocks(raw) if raw is not None else None

                if raw is None:
                    n_error += 1
                    output_script = "empty"
                    counts = {"devanagari_chars": 0, "latin_alpha_chars": 0, "other_chars": 0}
                    adherent = False
                    empty_log.append({
                        "call_id": call_id, "model": model, "condition": condition,
                        "case_id": p["case_id"], "input_form": p["input_form"],
                        "request_text": p["text"], "system_prompt": system_content,
                        "reason": f"API error after {attempts} attempts: {err}",
                    })
                else:
                    counts = count_scripts(cleaned)
                    output_script = classify(cleaned)
                    adherent = is_adherent(output_script, p["input_form"])
                    if output_script == "empty":
                        n_empty += 1
                        empty_log.append({
                            "call_id": call_id, "model": model, "condition": condition,
                            "case_id": p["case_id"], "input_form": p["input_form"],
                            "request_text": p["text"], "system_prompt": system_content,
                            "reason": "empty/whitespace-only response",
                        })

                row = {
                    "call_id": call_id,
                    "model": model,
                    "condition": condition,
                    "case_id": p["case_id"],
                    "category": p["category"],
                    "input_form": p["input_form"],
                    "request_text": p["text"],
                    "raw_response": raw,
                    "cleaned_response": cleaned,
                    "devanagari_chars": counts["devanagari_chars"],
                    "latin_alpha_chars": counts["latin_alpha_chars"],
                    "other_chars": counts["other_chars"],
                    "output_script": output_script,
                    "adherent": adherent,
                    "api_error": err,
                    "attempts": attempts,
                }
                append_checkpoint(row)
                n_done_this_run += 1

                if i % 25 == 0 or i == total_planned:
                    log.info("[%d/%d] model=%s condition=%s done_this_run=%d skipped=%d "
                              "empty=%d error=%d",
                              i, total_planned, model, condition,
                              n_done_this_run, n_skipped, n_empty, n_error)

                time.sleep(SLEEP_BETWEEN_CALLS)

    log.info("Run complete. New calls made: %d, resumed/skipped: %d, empty responses: %d, "
              "hard errors: %d", n_done_this_run, n_skipped, n_empty, n_error)

    return empty_log, total_planned


def write_empty_log(empty_log):
    path = os.path.join(RESULTS_DIR, "empty_responses.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["call_id", "model", "condition", "case_id", "input_form",
                      "request_text", "system_prompt", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(empty_log)
    log.info("Wrote %s (%d rows)", path, len(empty_log))


# ---------------------------------------------------------------------------
# Part 5: metrics
# ---------------------------------------------------------------------------
def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_ztest(x1, n1, x2, n2):
    """Two-proportion z-test, returns (z_stat, p_value). Uses statsmodels."""
    from statsmodels.stats.proportion import proportions_ztest
    if n1 == 0 or n2 == 0:
        return None, None
    try:
        z_stat, p_val = proportions_ztest([x1, x2], [n1, n2])
        return float(z_stat), float(p_val)
    except Exception:
        return None, None


def build_summary(all_rows):
    """Per (model, condition, input_form): n, adherence rate + CI, script
    breakdown %, mean char counts."""
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["model"], r["condition"], r["input_form"])].append(r)

    summary_rows = []
    for (model, condition, input_form), rows in groups.items():
        n = len(rows)
        n_adherent = sum(1 for r in rows if r["adherent"])
        rate = n_adherent / n if n else 0.0
        lo, hi = wilson_ci(n_adherent, n)

        script_counts = defaultdict(int)
        for r in rows:
            script_counts[r["output_script"]] += 1
        pct = {s: script_counts.get(s, 0) / n * 100 if n else 0.0
               for s in ["devanagari", "roman", "mixed", "empty"]}

        mean_dev = sum(r["devanagari_chars"] for r in rows) / n if n else 0.0
        mean_lat = sum(r["latin_alpha_chars"] for r in rows) / n if n else 0.0

        summary_rows.append({
            "model": model,
            "condition": condition,
            "input_form": input_form,
            "n": n,
            "n_adherent": n_adherent,
            "adherence_rate": rate,
            "adherence_ci_low": lo,
            "adherence_ci_high": hi,
            "pct_devanagari": pct["devanagari"],
            "pct_roman": pct["roman"],
            "pct_mixed": pct["mixed"],
            "pct_empty": pct["empty"],
            "mean_devanagari_chars": mean_dev,
            "mean_latin_alpha_chars": mean_lat,
        })
    return summary_rows


def build_headline_contrast(all_rows):
    """Per model, for roman and codemix input: adherence(implicit) vs
    adherence(explicit), delta, two-proportion z-test p-value."""
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["model"], r["condition"], r["input_form"])].append(r)

    rows = []
    for model in MODELS:
        for input_form in ["roman", "codemix"]:
            imp = groups.get((model, "implicit", input_form), [])
            exp = groups.get((model, "explicit", input_form), [])
            n_imp, n_exp = len(imp), len(exp)
            x_imp = sum(1 for r in imp if r["adherent"])
            x_exp = sum(1 for r in exp if r["adherent"])
            rate_imp = x_imp / n_imp if n_imp else None
            rate_exp = x_exp / n_exp if n_exp else None
            delta = (rate_exp - rate_imp) if (rate_imp is not None and rate_exp is not None) else None
            z_stat, p_val = two_proportion_ztest(x_imp, n_imp, x_exp, n_exp) if n_imp and n_exp else (None, None)

            rows.append({
                "model": model,
                "input_form": input_form,
                "n_implicit": n_imp,
                "adherence_implicit": rate_imp,
                "n_explicit": n_exp,
                "adherence_explicit": rate_exp,
                "delta": delta,
                "z_stat": z_stat,
                "p_value": p_val,
            })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_per_case_csv(all_rows):
    fieldnames = ["call_id", "model", "condition", "case_id", "category", "input_form",
                  "request_text", "raw_response", "cleaned_response",
                  "devanagari_chars", "latin_alpha_chars", "other_chars",
                  "output_script", "adherent", "api_error", "attempts"]
    with open(PER_CASE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    log.info("Wrote %s (%d rows)", PER_CASE_CSV, len(all_rows))


def write_summary_csv(summary_rows):
    fieldnames = ["model", "condition", "input_form", "n", "n_adherent",
                  "adherence_rate", "adherence_ci_low", "adherence_ci_high",
                  "pct_devanagari", "pct_roman", "pct_mixed", "pct_empty",
                  "mean_devanagari_chars", "mean_latin_alpha_chars"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Wrote %s (%d rows)", SUMMARY_CSV, len(summary_rows))


def print_pivot_table(summary_rows):
    print("\n" + "=" * 110)
    print("PIVOT: rows = model x condition, columns = input_form, cells = adherence rate [95% CI]")
    print("=" * 110)
    by_key = {(r["model"], r["condition"], r["input_form"]): r for r in summary_rows}
    input_forms = ["devanagari", "roman", "codemix"]

    hdr = f"{'model':24s} {'condition':10s}" + "".join(f" {f:>28s}" for f in input_forms)
    print(hdr)
    print("-" * len(hdr))
    for model in MODELS:
        for condition in CONDITION_NAMES:
            cells = []
            for form in input_forms:
                r = by_key.get((model, condition, form))
                if r is None:
                    cells.append("  n/a")
                else:
                    lo, hi = r["adherence_ci_low"], r["adherence_ci_high"]
                    cells.append(f"{r['adherence_rate']:.3f} [{lo:.3f},{hi:.3f}]")
            short_model = model.split("/")[-1]
            print(f"{short_model:24s} {condition:10s}" + "".join(f" {c:>28s}" for c in cells))


def print_headline_contrast(headline_rows):
    print("\n" + "=" * 110)
    print("HEADLINE CONTRAST: adherence(implicit) vs adherence(explicit), roman & codemix input")
    print("=" * 110)
    hdr = (f"{'model':24s} {'input_form':10s} {'n_impl':>7s} {'adh_impl':>9s} "
           f"{'n_expl':>7s} {'adh_expl':>9s} {'delta':>8s} {'z':>7s} {'p_value':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in headline_rows:
        def f3(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        short_model = r["model"].split("/")[-1]
        print(f"{short_model:24s} {r['input_form']:10s} {r['n_implicit']:>7d} "
              f"{f3(r['adherence_implicit']):>9s} {r['n_explicit']:>7d} "
              f"{f3(r['adherence_explicit']):>9s} {f3(r['delta']):>8s} "
              f"{f3(r['z_stat']):>7s} {f3(r['p_value']):>10s}")


def make_adherence_chart(summary_rows):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_forms = ["devanagari", "roman", "codemix"]
    condition_colors = {
        "none": "#8172B2", "implicit": "#4C72B0",
        "explicit": "#DD8452", "fewshot": "#55A868",
    }
    by_key = {(r["model"], r["condition"], r["input_form"]): r for r in summary_rows}

    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    x = np.arange(len(input_forms))
    width = 0.2

    for ax, model in zip(axes, MODELS):
        for i, cond in enumerate(CONDITION_NAMES):
            rates, err_low, err_high = [], [], []
            for form in input_forms:
                r = by_key.get((model, cond, form))
                if r is None:
                    rates.append(0)
                    err_low.append(0)
                    err_high.append(0)
                else:
                    rate = r["adherence_rate"]
                    lo = r["adherence_ci_low"] if r["adherence_ci_low"] is not None else rate
                    hi = r["adherence_ci_high"] if r["adherence_ci_high"] is not None else rate
                    rates.append(rate)
                    err_low.append(max(0, rate - lo))
                    err_high.append(max(0, hi - rate))
            offset = (i - 1.5) * width
            ax.bar(x + offset, rates, width, yerr=[err_low, err_high], capsize=2,
                   label=cond, color=condition_colors[cond])
        ax.set_title(model.split("/")[-1])
        ax.set_xticks(x)
        ax.set_xticklabels(input_forms)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Input form")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6)

    axes[0].set_ylabel("Adherence rate (Wilson 95% CI)")
    axes[0].legend(title="condition", fontsize=8)
    fig.suptitle("Script adherence rate by condition and input form, per model")
    fig.tight_layout()
    fig.savefig(CHART_ADHERENCE, dpi=150)
    log.info("Wrote %s", CHART_ADHERENCE)


def make_confusion_chart(all_rows):
    """Roman input only: stacked bars of output script, per model per condition."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roman_rows = [r for r in all_rows if r["input_form"] == "roman"]
    script_labels = ["roman", "mixed", "devanagari", "empty"]
    script_colors = {"roman": "#55A868", "mixed": "#DD8452",
                      "devanagari": "#C44E52", "empty": "#888888"}

    fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    x = np.arange(len(CONDITION_NAMES))

    for ax, model in zip(axes, MODELS):
        model_rows = [r for r in roman_rows if r["model"] == model]
        bottoms = np.zeros(len(CONDITION_NAMES))
        for label in script_labels:
            pcts = []
            for cond in CONDITION_NAMES:
                cond_rows = [r for r in model_rows if r["condition"] == cond]
                n = len(cond_rows)
                cnt = sum(1 for r in cond_rows if r["output_script"] == label)
                pcts.append(cnt / n * 100 if n else 0.0)
            ax.bar(x, pcts, bottom=bottoms, label=label, color=script_colors[label])
            bottoms += np.array(pcts)
        ax.set_title(model.split("/")[-1])
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_NAMES, rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Condition")

    axes[0].set_ylabel("% of responses (roman input only)")
    axes[0].legend(title="output script", fontsize=8)
    fig.suptitle("What script came back for ROMAN input, by model and condition")
    fig.tight_layout()
    fig.savefig(CHART_CONFUSION, dpi=150)
    log.info("Wrote %s", CHART_CONFUSION)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help="Resume from checkpoint (default behavior always "
                              "skips already-completed calls; this flag is "
                              "accepted for the documented resume invocation).")
    parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    empty_log, total_planned = run_all_calls()

    done = load_checkpoint()
    all_rows = list(done.values())

    n_empty_or_error = len(empty_log)
    empty_rate = n_empty_or_error / total_planned if total_planned else 0.0
    write_empty_log(empty_log)

    print(f"\nEMPTY/FAILED RESPONSES: {n_empty_or_error} / {total_planned} "
          f"({empty_rate:.2%})")

    write_per_case_csv(all_rows)
    summary_rows = build_summary(all_rows)
    write_summary_csv(summary_rows)

    print_pivot_table(summary_rows)
    headline_rows = build_headline_contrast(all_rows)
    print_headline_contrast(headline_rows)

    make_adherence_chart(summary_rows)
    make_confusion_chart(all_rows)

    if empty_rate > 0.02:
        print(f"\nFAILING LOUDLY: empty/failed rate {empty_rate:.2%} exceeds 2% threshold.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
