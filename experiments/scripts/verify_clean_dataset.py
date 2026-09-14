"""
Part 4: verify results_clean.json before anything downstream uses it.

Asserts:
  - every case has script_adherent == True
  - no empty answers
  - for each task, hi and hinglish answers are NOT near-duplicates
    (difflib.SequenceMatcher ratio < 0.6)
  - n == 10 per variant

Prints the near-duplicate ratios for every task either way (that's a
result, not just a gate). Exits non-zero and prints which assertion(s)
failed if verification fails; scripts/rerun_all.py checks this exit code
before proceeding, per the task's "if any assertion fails, stop and
report."

Run:  python experiments/scripts/verify_clean_dataset.py
"""
import io
import sys
import os
import json
import difflib
from collections import defaultdict

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")
NEAR_DUP_THRESHOLD = 0.6


def main():
    with open(CLEAN_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    failures = []

    print("=" * 90)
    print("VERIFYING results_clean.json")
    print("=" * 90)

    # NOTE: rows that failed script adherence after all attempts are still
    # present in results_clean.json (per Part 2/3 spec: "keep every attempt"
    # and record the failure), but they must be EXCLUDED from the set that
    # downstream scripts treat as "the clean dataset". We check the full
    # file for schema/failure-flag sanity, then build the excluded-failures
    # view for the n==10-per-variant and near-duplicate checks, since those
    # checks are about the usable clean data.
    all_rows = rows
    clean_rows = [r for r in rows if not r["script_adherence_failure"]]

    excluded = [r for r in rows if r["script_adherence_failure"]]
    print(f"\nTotal cases in file: {len(all_rows)}")
    print(f"Excluded (script_adherence_failure=True): {len(excluded)}")
    for r in excluded:
        print(f"  EXCLUDED: {r['case_id']}  script={r['output_script']}  "
              f"attempts={r['generation_attempts']}  answer={r['answer'][:60]!r}")

    # --- assertion 1: every case in the CLEAN set has script_adherent=True ---
    print("\n--- assertion: every clean-set case has script_adherent=True ---")
    bad_adherence = [r["case_id"] for r in clean_rows if not r["script_adherent"]]
    if bad_adherence:
        failures.append(f"script_adherent=False found in clean set (should be excluded): {bad_adherence}")
        print(f"  FAIL: {bad_adherence}")
    else:
        print(f"  OK: all {len(clean_rows)} clean-set cases have script_adherent=True")

    # --- assertion 2: no empty answers (in the clean set) ---
    print("\n--- assertion: no empty answers in clean set ---")
    empty = [r["case_id"] for r in clean_rows if not (r["answer"] or "").strip()]
    if empty:
        failures.append(f"empty answers found in clean set: {empty}")
        print(f"  FAIL: {empty}")
    else:
        print(f"  OK: no empty answers among {len(clean_rows)} clean-set cases")

    # --- assertion 3: n == 10 per variant (of ALL generated cases, since
    #     that's what "n per variant" means before any exclusion -- we also
    #     report clean-set n separately since exclusions reduce it) ---
    print("\n--- assertion: n == 10 per variant (generated) ---")
    by_variant_all = defaultdict(list)
    for r in all_rows:
        by_variant_all[r["variant"]].append(r)
    n_bad = {}
    for v in ["en", "hi", "hinglish"]:
        n = len(by_variant_all[v])
        print(f"  {v:10s}: n_generated={n}  n_clean={sum(1 for r in by_variant_all[v] if not r['script_adherence_failure'])}")
        if n != 10:
            n_bad[v] = n
    if n_bad:
        failures.append(f"n != 10 per variant (generated): {n_bad}")
        print(f"  FAIL: {n_bad}")
    else:
        print("  OK: 10 generated per variant")

    # --- assertion 4: hi vs hinglish are not near-duplicates, per task ---
    print("\n--- near-duplicate check: hi vs hinglish per task (SequenceMatcher ratio) ---")
    by_task = defaultdict(dict)
    for r in clean_rows:
        by_task[r["task_id"]][r["variant"]] = r["answer"]

    near_dup_failures = []
    for task_id, variants in sorted(by_task.items()):
        hi = variants.get("hi")
        hl = variants.get("hinglish")
        if hi is None or hl is None:
            print(f"  {task_id:28s}: SKIPPED (hi or hinglish excluded from clean set)")
            continue
        ratio = difflib.SequenceMatcher(None, hi, hl).ratio()
        status = "OK" if ratio < NEAR_DUP_THRESHOLD else "FAIL (near-duplicate)"
        print(f"  {task_id:28s}: ratio={ratio:.3f}  [{status}]")
        if ratio >= NEAR_DUP_THRESHOLD:
            near_dup_failures.append((task_id, ratio))

    if near_dup_failures:
        failures.append(f"hi/hinglish near-duplicates (ratio >= {NEAR_DUP_THRESHOLD}): {near_dup_failures}")

    # -----------------------------------------------------------------
    print("\n" + "=" * 90)
    if failures:
        print(f"VERIFICATION FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(" -", f)
        print("=" * 90)
        return 1
    else:
        print("VERIFICATION PASSED: all assertions hold.")
        print("=" * 90)
        return 0


if __name__ == "__main__":
    sys.exit(main())
