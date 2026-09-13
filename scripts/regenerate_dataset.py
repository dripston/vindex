"""
*** DO NOT RE-RUN THIS AGAINST results_clean.json. ***
results_clean.json is FROZEN as of the pinned generation run (see the git
commit that introduced it). The model (openai/gpt-oss-20b via Groq) is NOT
deterministic even at temperature=0 -- re-running this script produces
different answer text on ~1/3 of cases (confirmed empirically: 11/30 cases
changed wording across two identical regeneration runs). Any dataset
labelling, hand-authored negatives (testcases.py, minimal_edit.py's EDITS
table), or scoring done against the current results_clean.json would be
silently invalidated by regenerating it. If the dataset genuinely needs to
change, produce results_clean_v2.json (or similar) as a NEW, separately
reviewed artifact -- never overwrite the pinned file in place.

Regenerate the 30-case answer dataset with script adherence ENFORCED at
generation time, fixing the contamination in results.json where the model
answered Romanized-Hinglish prompts in Devanagari script (8/10 cases) and
one Hindi case silently kept an empty answer.

Does NOT edit run_experiment.py or results.json. Produces results_clean.json
alongside it, with the same schema plus:
  output_script            : script_check.classify() label of the final answer
  script_adherent          : bool
  generation_attempts      : int, how many attempts it took
  script_adherence_failure : bool -- true if all 5 attempts failed; such
                              cases are EXCLUDED from the "clean" set used
                              downstream, and never hand-replaced.

Checkpointed to disk after every case (results_clean.checkpoint.json) so a
crash mid-run does not lose completed work; reruns resume from there.

Run:  python scripts/regenerate_dataset.py
"""
import io
import sys
import os
import json
import time

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# --- load .env (same pattern as run_experiment.py) ---
ENV = {}
with open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, val = line.split("=", 1)
            ENV[k.strip()] = val.strip()
os.environ.setdefault("GROQ_API_KEY", ENV.get("GROQ_API_KEY", ""))

from groq import Groq
from testcases import build_cases
from script_check import classify, is_script_adherent

MODEL_UNDER_TEST = "openai/gpt-oss-20b"
MAX_ATTEMPTS = 5
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "results_clean.checkpoint.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "results_clean.json")

client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ---------------------------------------------------------------------------
# System prompts, escalating in strength across attempts. Attempt index is
# 0-based; SYSTEM_PROMPTS[i] is used on attempt i+1. The last prompt is
# reused for any attempt beyond the list length.
# ---------------------------------------------------------------------------
def system_prompt_for(variant, attempt_index):
    """attempt_index: 0-based attempt number (0 = first try)."""
    common_suffix = (
        " Give a short answer of one or two sentences that states the fact "
        "and a brief bit of context."
    )

    if variant == "en":
        prompts = [
            "You are a helpful assistant. Reply in English, written in the Latin/Roman alphabet." + common_suffix,
            "You must reply in English using ONLY Latin/Roman letters. Do not use any other script." + common_suffix,
            "STRICT REQUIREMENT: your entire reply must be English text in the Latin alphabet (A-Z). "
            "No Devanagari, no other script, under any circumstances." + common_suffix,
        ]
    elif variant == "hi":
        prompts = [
            "You are a helpful assistant. Reply in Hindi, written in the Devanagari script." + common_suffix,
            "You must reply in Hindi using ONLY the Devanagari script (देवनागरी). Do not use Latin/Roman letters "
            "for the answer." + common_suffix,
            "STRICT REQUIREMENT: your entire reply must be in Hindi written in the Devanagari script "
            "(देवनागरी), e.g. 'मुंबई महाराष्ट्र की राजधानी है।'. Do not answer in English or Romanized text "
            "under any circumstances." + common_suffix,
        ]
    elif variant == "hinglish":
        prompts = [
            "You are a helpful assistant. Reply in Hinglish: Hindi words written in the Latin/Roman alphabet "
            "(Romanized), e.g. 'Mumbai Maharashtra ki rajdhani hai.' Do NOT use Devanagari script." + common_suffix,
            "You must reply in ROMANIZED HINGLISH: Hindi sentence structure and vocabulary but spelled out "
            "using ONLY Latin/Roman letters (A-Z), like typing Hindi on an English keyboard, e.g. "
            "'Mumbai Maharashtra ki rajdhani hai.' Using Devanagari script (देवनागरी) anywhere in your reply "
            "is NOT allowed and will be rejected." + common_suffix,
            "STRICT REQUIREMENT: your entire reply must use ONLY Latin/Roman alphabet characters (A-Z, a-z). "
            "Write Hindi words phonetically in Roman letters (Romanized Hinglish), for example: "
            "'Mumbai Maharashtra ki rajdhani hai, aur ye India ka sabse bada shहर bhi hai' is WRONG because "
            "it contains a Devanagari character -- every single character must be Latin. Devanagari script "
            "(देवनागरी) of ANY kind, even one character, is FORBIDDEN and will be rejected and regenerated." + common_suffix,
        ]
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    idx = min(attempt_index, len(prompts) - 1)
    return prompts[idx]


def call_model(system_prompt, question):
    """Single Groq call with retry/backoff on transient API errors (mirrors
    run_experiment.py's get_answer retry pattern). This retries API
    failures, not script-adherence failures -- those are handled by the
    caller via a fresh call with a stronger prompt."""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_UNDER_TEST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=250,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    [API retry {attempt}] {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Groq API call failed after 5 retries")


def generate_with_validation(case):
    """Generate an answer for one case, retrying up to MAX_ATTEMPTS times
    with escalating system prompts until script_check says the output
    matches the expected script. Returns a dict describing the outcome,
    including the full attempt log."""
    variant = case["variant"]
    question = case["question"]

    attempts_log = []
    final_answer = None
    final_script = None
    adherent = False

    for i in range(MAX_ATTEMPTS):
        sys_prompt = system_prompt_for(variant, i)
        raw = call_model(sys_prompt, question)
        script_label = classify(raw)
        ok = is_script_adherent(raw, variant)

        attempts_log.append({
            "attempt": i + 1,
            "system_prompt": sys_prompt,
            "raw_response": raw,
            "script_classification": script_label,
            "adherent": ok,
        })

        print(f"    attempt {i+1}/{MAX_ATTEMPTS}: script={script_label:10s} "
              f"adherent={ok}  -> {raw[:70]!r}", flush=True)

        if ok:
            final_answer = raw
            final_script = script_label
            adherent = True
            break

        time.sleep(1)

    if not adherent:
        # Keep the LAST attempt's raw text as the recorded (non-adherent)
        # answer for transparency, but this case will be marked as a
        # failure and excluded from the clean set -- never hand-replaced.
        final_answer = attempts_log[-1]["raw_response"]
        final_script = attempts_log[-1]["script_classification"]

    return {
        "case_id": case["case_id"],
        "task_id": case["task_id"],
        "variant": variant,
        "question": question,
        "gold": case["gold"],
        "answer": final_answer,
        "output_script": final_script,
        "script_adherent": adherent,
        "generation_attempts": len(attempts_log),
        "script_adherence_failure": not adherent,
        "attempts_log": attempts_log,
    }


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(done_by_case_id):
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(done_by_case_id, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)


def main():
    cases = build_cases()
    done = load_checkpoint()
    print(f"Resuming with {len(done)}/{len(cases)} cases already checkpointed.", flush=True)

    for i, case in enumerate(cases, 1):
        if case["case_id"] in done:
            print(f"[{i}/{len(cases)}] {case['case_id']} -- already done, skipping.", flush=True)
            continue

        print(f"[{i}/{len(cases)}] {case['case_id']} (variant={case['variant']})", flush=True)
        result = generate_with_validation(case)
        done[case["case_id"]] = result
        save_checkpoint(done)

    # Final output: same schema as results.json plus the new fields.
    # Order matches build_cases() order for readability.
    final_rows = [done[c["case_id"]] for c in cases]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUTPUT_PATH} ({len(final_rows)} rows)", flush=True)

    print_summary(final_rows)


def print_summary(rows):
    from collections import defaultdict

    print("\n" + "=" * 90)
    print("SCRIPT ADHERENCE SUMMARY")
    print("=" * 90)
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    print(f"{'variant':10s} {'n_generated':>12s} {'n_adherent':>11s} {'n_failed':>9s} {'mean_attempts':>14s}")
    print("-" * 60)
    for v in ["en", "hi", "hinglish"]:
        rs = by_variant[v]
        n = len(rs)
        n_ok = sum(1 for r in rs if r["script_adherent"])
        n_fail = sum(1 for r in rs if r["script_adherence_failure"])
        mean_att = sum(r["generation_attempts"] for r in rs) / n if n else 0.0
        print(f"{v:10s} {n:>12d} {n_ok:>11d} {n_fail:>9d} {mean_att:>14.2f}")

    print("\nCases that FAILED script adherence after all attempts (excluded from clean set):")
    failures = [r for r in rows if r["script_adherence_failure"]]
    if not failures:
        print("  (none)")
    else:
        for r in failures:
            print(f"  {r['case_id']:40s} expected={sorted(_expected(r['variant']))} "
                  f"got={r['output_script']}  answer={r['answer'][:80]!r}")


def _expected(variant):
    from script_check import expected_script
    return expected_script(variant)


if __name__ == "__main__":
    main()
