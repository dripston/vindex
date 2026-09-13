"""
Part 3 (this turn): record the script-adherence finding as data.

Compares two system prompts on the same 30 questions:
  ORIGINAL : "Reply in the SAME language and script the user used."
             (run_experiment.py's prompt -- results.json)
  STRICT   : "...use ONLY Roman/Latin letters... Devanagari is FORBIDDEN..."
             (regenerate_dataset.py's escalating prompts -- results_clean.json)

Does NOT call an LLM, does NOT re-run any generation. Reads the two
existing JSON files and classifies results.json's answers with
script_check.py (they were never classified at generation time).

Output:
  results_clean/script_adherence.csv
    one row per case (30 rows): case_id, variant,
    instruction_strength_used, attempts_needed, output_script_classified,
    adherent

  Printed summary:
    adherence rate per variant under ORIGINAL vs STRICT, and the delta.

Run:  python scripts/script_adherence_report.py
"""
import os
import sys
import csv
import json
import logging

if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from script_check import classify, is_script_adherent  # noqa: E402

ORIGINAL_JSON_PATH = os.path.join(REPO_ROOT, "results.json")
CLEAN_JSON_PATH = os.path.join(REPO_ROOT, "results_clean.json")
OUTPUT_CSV_PATH = os.path.join(REPO_ROOT, "results_clean", "script_adherence.csv")

VARIANTS = ["en", "hi", "hinglish"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("script_adherence_report")

ORIGINAL_PROMPT_LABEL = "original_same_language"
STRICT_PROMPT_LABEL = "strict_roman_forbidden_devanagari"


def build_rows():
    with open(ORIGINAL_JSON_PATH, encoding="utf-8") as f:
        original_rows = {r["case_id"]: r for r in json.load(f)}
    with open(CLEAN_JSON_PATH, encoding="utf-8") as f:
        clean_rows = {r["case_id"]: r for r in json.load(f)}

    all_case_ids = sorted(set(original_rows) | set(clean_rows))
    rows = []

    for case_id in all_case_ids:
        variant = case_id.split("__")[-1]
        if variant not in VARIANTS:
            # defensive: case_id format is always <task_id>__<variant>
            parts = case_id.split("__")
            variant = parts[-1] if parts[-1] in VARIANTS else "unknown"

        orig = original_rows.get(case_id)
        if orig is not None:
            orig_answer = orig.get("answer")
            orig_script = classify(orig_answer)
            orig_adherent = is_script_adherent(orig_answer, variant)
            rows.append({
                "case_id": case_id,
                "variant": variant,
                "instruction_strength_used": ORIGINAL_PROMPT_LABEL,
                "attempts_needed": 1,  # run_experiment.py made exactly one attempt per case
                "output_script_classified": orig_script,
                "adherent": orig_adherent,
            })

        clean = clean_rows.get(case_id)
        if clean is not None:
            rows.append({
                "case_id": case_id,
                "variant": variant,
                "instruction_strength_used": STRICT_PROMPT_LABEL,
                "attempts_needed": clean.get("generation_attempts"),
                "output_script_classified": clean.get("output_script"),
                "adherent": clean.get("script_adherent"),
            })

    return rows


def main():
    rows = build_rows()

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "variant", "instruction_strength_used",
                      "attempts_needed", "output_script_classified", "adherent"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s (%d rows)", OUTPUT_CSV_PATH, len(rows))

    print_summary(rows)


def print_summary(rows):
    from collections import defaultdict

    by_prompt_variant = defaultdict(list)
    for r in rows:
        by_prompt_variant[(r["instruction_strength_used"], r["variant"])].append(r)

    print("\n" + "=" * 90)
    print("SCRIPT ADHERENCE: ORIGINAL vs STRICT system prompt, per variant")
    print("=" * 90)
    hdr = f"{'variant':10s} {'n_original':>10s} {'adherent_original':>18s} {'rate_original':>14s} " \
          f"{'n_strict':>9s} {'adherent_strict':>16s} {'rate_strict':>12s} {'delta_rate':>11s}"
    print(hdr)
    print("-" * len(hdr))

    for v in VARIANTS:
        orig_rows = by_prompt_variant[(ORIGINAL_PROMPT_LABEL, v)]
        strict_rows = by_prompt_variant[(STRICT_PROMPT_LABEL, v)]

        n_orig = len(orig_rows)
        adherent_orig = sum(1 for r in orig_rows if r["adherent"])
        rate_orig = adherent_orig / n_orig if n_orig else 0.0

        n_strict = len(strict_rows)
        adherent_strict = sum(1 for r in strict_rows if r["adherent"])
        rate_strict = adherent_strict / n_strict if n_strict else 0.0

        delta = rate_strict - rate_orig

        print(f"{v:10s} {n_orig:>10d} {adherent_orig:>18d} {rate_orig:>14.3f} "
              f"{n_strict:>9d} {adherent_strict:>16d} {rate_strict:>12.3f} {delta:>11.3f}")

    # Overall (all variants pooled)
    orig_all = [r for r in rows if r["instruction_strength_used"] == ORIGINAL_PROMPT_LABEL]
    strict_all = [r for r in rows if r["instruction_strength_used"] == STRICT_PROMPT_LABEL]
    n_orig, n_strict = len(orig_all), len(strict_all)
    rate_orig = sum(1 for r in orig_all if r["adherent"]) / n_orig if n_orig else 0.0
    rate_strict = sum(1 for r in strict_all if r["adherent"]) / n_strict if n_strict else 0.0
    print("-" * len(hdr))
    print(f"{'ALL':10s} {n_orig:>10d} {sum(1 for r in orig_all if r['adherent']):>18d} {rate_orig:>14.3f} "
          f"{n_strict:>9d} {sum(1 for r in strict_all if r['adherent']):>16d} {rate_strict:>12.3f} "
          f"{rate_strict - rate_orig:>11.3f}")


if __name__ == "__main__":
    main()
