"""
Milestone 1.6: validate the vindex port against the original finding.

experiments/scripts/script_adherence_report.py established the headline
result this repo is built around: 20% Hinglish script adherence under the
original ("reply in the same language and script") system prompt, 100%
under the strict, script-forbidding one. That script called
script_check.py's is_script_adherent(answer, variant) directly.

If src/vindex/script.py's port of that same logic (Milestone 1.1/1.2) is
correct, running it over the same two files must reproduce the same
numbers exactly. If it doesn't, the port broke something -- this is the
reason experiments/results.json (the contaminated file) stays in the
repo instead of being deleted: without it, this check has nothing to
run against.

Two independent checks, both against the same two files:

  1. Variant-driven (methodologically identical to the original report):
     vindex.script.is_script_adherent(answer, variant), where variant
     comes from the case_id / "variant" field, exactly as
     script_adherence_report.py did it.

  2. Prompt-driven (Milestone 1.4's public metric): vindex.metric.
     script_adherence(question, answer). Both files' "question" field is
     already written in the case's variant (Hindi question in Devanagari
     for _hi, Romanized Hindi question for _hinglish), so it can stand in
     as the "prompt" the new two-string metric expects. This exercises
     the actual shipped public API, not just the ported classify().

Both must agree with each other and with the original 20%/100% figure.

Run:  python experiments/scripts/validate_vindex_port.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from collections import defaultdict

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))  # src-layout vindex package

from vindex.metric import script_adherence  # noqa: E402
from vindex.script import is_script_adherent  # noqa: E402

# This script's whole job is the before/after comparison of the two system
# prompts, so it legitimately reads the contaminated file on purpose -- see
# Correction 1 in ARCHITECTURE.md.
ORIGINAL_JSON_PATH = os.path.join(EXPERIMENTS_DIR, "results.json")
CLEAN_JSON_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")

VARIANTS = ["en", "hi", "hinglish"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("validate_vindex_port")

# The numbers this whole script exists to reproduce.
EXPECTED_HINGLISH_RATE_ORIGINAL = 0.20
EXPECTED_HINGLISH_RATE_STRICT = 1.00


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def variant_of(row: dict) -> str:
    variant = row.get("variant")
    if variant in VARIANTS:
        return variant  # type: ignore[return-value]
    parts = row["case_id"].split("__")
    return parts[-1] if parts[-1] in VARIANTS else "unknown"


def variant_driven_rates(rows: list[dict]) -> dict[str, float]:
    """Reproduces script_adherence_report.py's methodology exactly:
    is_script_adherent(answer, variant), grouped by variant."""
    by_variant: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        variant = variant_of(row)
        adherent = is_script_adherent(row.get("answer"), variant)
        by_variant[variant].append(adherent)
    return {v: (sum(vals) / len(vals) if vals else 0.0) for v, vals in by_variant.items()}


def prompt_driven_rates(rows: list[dict]) -> dict[str, float]:
    """Exercises the shipped Milestone 1.4 metric: script_adherence(prompt,
    response), using each row's "question" field as the prompt -- it is
    already written in that case's variant script/language."""
    by_variant: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        variant = variant_of(row)
        result = script_adherence(row.get("question"), row.get("answer"))
        by_variant[variant].append(result.passed)
    return {v: (sum(vals) / len(vals) if vals else 0.0) for v, vals in by_variant.items()}


def print_table(title: str, original_rates: dict[str, float], strict_rates: dict[str, float]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    hdr = f"{'variant':10s} {'rate_original':>14s} {'rate_strict':>12s} {'delta':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        ro = original_rates.get(v, 0.0)
        rs = strict_rates.get(v, 0.0)
        print(f"{v:10s} {ro:>14.3f} {rs:>12.3f} {rs - ro:>8.3f}")


def main() -> None:
    original_rows = load_rows(ORIGINAL_JSON_PATH)
    clean_rows = load_rows(CLEAN_JSON_PATH)
    log.info("Loaded %d rows from %s", len(original_rows), ORIGINAL_JSON_PATH)
    log.info("Loaded %d rows from %s", len(clean_rows), CLEAN_JSON_PATH)

    variant_original = variant_driven_rates(original_rows)
    variant_strict = variant_driven_rates(clean_rows)
    print_table("CHECK 1: variant-driven (vindex.script.is_script_adherent)", variant_original, variant_strict)

    prompt_original = prompt_driven_rates(original_rows)
    prompt_strict = prompt_driven_rates(clean_rows)
    print_table("CHECK 2: prompt-driven (vindex.metric.script_adherence)", prompt_original, prompt_strict)

    failures = []

    def check(desc: str, actual: float, expected: float) -> None:
        if abs(actual - expected) > 1e-9:
            failures.append(f"FAIL: {desc}: got {actual!r}, expected {expected!r}")
        else:
            print(f"ok:   {desc}: {actual!r}")

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)
    check(
        "CHECK 1 (variant-driven) hinglish rate, original prompt",
        variant_original.get("hinglish", 0.0),
        EXPECTED_HINGLISH_RATE_ORIGINAL,
    )
    check(
        "CHECK 1 (variant-driven) hinglish rate, strict prompt",
        variant_strict.get("hinglish", 0.0),
        EXPECTED_HINGLISH_RATE_STRICT,
    )
    check(
        "CHECK 2 (prompt-driven) hinglish rate, original prompt",
        prompt_original.get("hinglish", 0.0),
        EXPECTED_HINGLISH_RATE_ORIGINAL,
    )
    check(
        "CHECK 2 (prompt-driven) hinglish rate, strict prompt",
        prompt_strict.get("hinglish", 0.0),
        EXPECTED_HINGLISH_RATE_STRICT,
    )

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S) -- the vindex port does not reproduce the original finding:")
        for f in failures:
            print(" ", f)
        raise SystemExit(1)
    print("All checks passed. vindex reproduces the 20%/100% Hinglish adherence finding.")


if __name__ == "__main__":
    main()
