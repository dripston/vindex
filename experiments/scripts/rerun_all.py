"""
Orchestrator: runs encoder_comparison.py, discrimination.py, and
minimal_edit.py against data/results_clean.json (their shared default now
that Correction 1 gave each an explicit DATASET_PATH), writing outputs to
experiments/results_clean/. No monkeypatching -- each module takes its
dataset path as an explicit main(dataset_path) argument.

Also runs verify_clean_dataset.py FIRST and refuses to proceed if it
fails, and, after all reruns, builds the before/after contamination-impact
comparison (which itself reads experiments/results/ and
experiments/results_clean/, both already on disk).

Run:  python experiments/scripts/rerun_all.py
"""
import io
import sys
import os
import subprocess
import importlib

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.dirname(__file__)
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

RESULTS_CLEAN_DIR = os.path.join(EXPERIMENTS_DIR, "results_clean")
CLEAN_JSON_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")


def run_verification_gate():
    """Refuse to proceed if results_clean.json fails its invariants."""
    print("=" * 90)
    print("STEP 0: verifying data/results_clean.json")
    print("=" * 90)
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "verify_clean_dataset.py")],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        print("\nVerification FAILED. Stopping before the rerun, as required.", flush=True)
        sys.exit(1)
    print("\nVerification passed. Proceeding.\n", flush=True)


def run_encoder_comparison_on_clean_data():
    print("=" * 90)
    print("STEP 1: encoder_comparison.py on data/results_clean.json -> experiments/results_clean/")
    print("=" * 90)
    import encoder_comparison as ec
    importlib.reload(ec)
    os.makedirs(RESULTS_CLEAN_DIR, exist_ok=True)
    ec.main(CLEAN_JSON_PATH)


def run_discrimination_on_clean_data():
    print("\n" + "=" * 90)
    print("STEP 2: discrimination.py on data/results_clean.json -> experiments/results_clean/")
    print("=" * 90)
    import discrimination as disc
    importlib.reload(disc)
    disc.main(CLEAN_JSON_PATH)


def run_minimal_edit_on_clean_data():
    print("\n" + "=" * 90)
    print("STEP 3: minimal_edit.py on data/results_clean.json -> experiments/results_clean/")
    print("=" * 90)
    import minimal_edit as me
    importlib.reload(me)
    me.main(CLEAN_JSON_PATH)


def main():
    run_verification_gate()
    run_encoder_comparison_on_clean_data()
    run_discrimination_on_clean_data()
    run_minimal_edit_on_clean_data()

    print("\n" + "=" * 90)
    print("STEP 4: before/after contamination-impact comparison")
    print("=" * 90)
    import contamination_impact
    importlib.reload(contamination_impact)
    contamination_impact.main()


if __name__ == "__main__":
    main()
