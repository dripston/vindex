"""ARCHIVED -- DO NOT RUN. Deeper read of results.json for the 0.1 writeup.

Superseded by scripts/contamination_impact.py (the before/after comparison)
and scripts/discrimination.py (the proper discrimination scoring across
encoders). Kept for the record: it answered a real question in week one,
it is not broken, it was just replaced by a better version of the same
idea. See experiments/README.md.
"""
import io, os, sys, json
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_here = os.path.dirname(os.path.abspath(__file__))
results = json.load(open(os.path.join(_here, "results.json"), encoding="utf-8"))
by_var = defaultdict(list)
for r in results:
    by_var[r["variant"]].append(r)

print("PER-CASE TABLE (human-correct only, one row per task)\n")
tasks = sorted(set(r["task_id"] for r in results))
print(f"{'task':26s} | {'EN sim':>7s} {'HI sim':>7s} {'HL sim':>7s} | {'EN jNR':>7s} {'HI jNR':>7s} {'HL jNR':>7s}")
print("-" * 90)
for t in tasks:
    row = {r["variant"]: r for r in results if r["task_id"] == t}
    def s(v, k):
        return f"{row[v][k]:.2f}" if v in row and row[v][k] is not None else "  - "
    print(f"{t:26s} | {s('en','similarity'):>7s} {s('hi','similarity'):>7s} {s('hinglish','similarity'):>7s} "
          f"| {s('en','judge_noref_score'):>7s} {s('hi','judge_noref_score'):>7s} {s('hinglish','judge_noref_score'):>7s}")

print("\n\n=== AGGREGATE (human-correct answers only) ===\n")
for v in ["en", "hi", "hinglish"]:
    rows = [r for r in by_var[v] if r["human_correct"]]
    n = len(rows)
    jref = [r["judge_score"] for r in rows]
    jnr = [r["judge_noref_score"] for r in rows]
    sims = [r["similarity"] for r in rows]
    ems = [r["exact_match"] for r in rows]
    print(f"{v:9s} n={n}")
    print(f"   judge (with reference)  : avg {sum(jref)/n:.3f}  | pass@0.5 {sum(x>=0.5 for x in jref)}/{n}")
    print(f"   judge (no reference)    : avg {sum(jnr)/n:.3f}  | pass@0.5 {sum(x>=0.5 for x in jnr)}/{n}")
    print(f"   exact-match vs gold     : {sum(ems)}/{n} = {sum(ems)/n:.0%}")
    print(f"   embedding similarity    : avg {sum(sims)/n:.3f}  min {min(sims):.3f}  max {max(sims):.3f}")
    print(f"   sim >= 0.5 (typical thr): {sum(s>=0.5 for s in sims)}/{n}")
    print()

print("=== EXACT-MATCH: would ANY threshold-style string metric pass? ===")
print("Across all 30 cases, exact_match True count:", sum(r["exact_match"] for r in results), "/ 30")
print("(gold strings are short canonical forms; every model answer adds context/script)\n")

print("=== SIMILARITY GAP: English vs Hinglish vs Hindi, same fact ===")
en = [r["similarity"] for r in by_var["en"] if r["human_correct"]]
hi = [r["similarity"] for r in by_var["hi"] if r["human_correct"]]
hl = [r["similarity"] for r in by_var["hinglish"] if r["human_correct"]]
print(f"  mean similarity  EN={sum(en)/len(en):.3f}  HI={sum(hi)/len(hi):.3f}  HL={sum(hl)/len(hl):.3f}")
print(f"  HI is {(1-(sum(hi)/len(hi))/(sum(en)/len(en)))*100:.0f}% lower than EN")
print(f"  HL is {(1-(sum(hl)/len(hl))/(sum(en)/len(en)))*100:.0f}% lower than EN")

print("\n=== JUDGE FAILURE / DRIFT NOTES ===")
for r in results:
    jr = (r["judge_reason"] or "").lower()
    # detect the judge silently mistranslating the Hindi/Hinglish answer
    flags = []
    if r["variant"] in ("hi", "hinglish"):
        if "sea floor" in jr or "sea-floor" in jr or "seafloor" in jr:
            flags.append("judge mistranslated 'समुद्र तल' (sea level) as 'sea floor'")
    jnrr = (r["judge_noref_reason"] or "").lower()
    if r["human_correct"] and r["judge_noref_score"] is not None and r["judge_noref_score"] < 0.5:
        flags.append(f"NO-REF JUDGE FAILED a human-correct answer (score {r['judge_noref_score']})")
    if r["human_correct"] and r["judge_score"] is not None and r["judge_score"] < 0.5:
        flags.append(f"REF JUDGE FAILED a human-correct answer (score {r['judge_score']})")
    for f in flags:
        print(f"  [{r['variant']}] {r['task_id']}: {f}")
