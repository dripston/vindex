"""
0.1 Empirical validation: does an English LLM judge penalize Hinglish?
Does string/similarity scoring break across scripts?

Pipeline:
  1. For each of 30 cases, get an answer from the model under test (Groq llama-3.3-70b).
  2. Score each answer 4 ways:
       - human_correct   : language-agnostic fact check (ground truth)
       - judge_score      : DeepEval GEval "Correctness" with an English LLM judge
       - exact_match      : normalized exact match vs gold
       - similarity       : sentence-transformers cosine vs gold
  3. Dump everything to results.json + print a summary table.

Run:  python experiments/run_experiment.py
"""
import io
import sys
import os
import json
import time
import re

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# --- load .env ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # testcases.py lives alongside this file
ENV = {}
with open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, val = line.split("=", 1)
            ENV[k.strip()] = val.strip()
os.environ.setdefault("GROQ_API_KEY", ENV.get("GROQ_API_KEY", ""))
# DeepEval's OpenAI-compatible judge talks to Groq's OpenAI endpoint.
os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"

from groq import Groq
from testcases import build_cases, human_is_correct

MODEL_UNDER_TEST = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def get_answer(question):
    """Answer the question. Neutral instruction: reply in the same language,
    give one or two sentences (not just a bare word) so the judge has real text
    to parse -- this is where script/language bias tends to surface."""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL_UNDER_TEST,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Reply in the SAME language and script the user used. Give a short answer of one or two sentences that states the fact and a brief bit of context."},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                max_tokens=250,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [retry {attempt}] answer error: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("answer failed after retries")


# ---------------- string scorers ----------------
def normalize(s):
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(answer, gold):
    a = normalize(answer)
    g = normalize(gold)
    # "exact match" as eval tools apply it: gold must equal answer, or be the whole answer
    return a == g or a == g + "." or g == a


_st_model = None
def _load_st():
    global _st_model
    if _st_model is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use local cache only
        print("  loading sentence-transformers model (cached)...", flush=True)
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  model ready", flush=True)
    return _st_model


def similarity(answer, gold):
    m = _load_st()
    from sentence_transformers import util
    emb = m.encode([answer or "", gold or ""], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))


# ---------------- LLM judge ----------------
# This is a G-Eval / promptfoo-llm-rubric style judge: an English rubric prompt,
# chain-of-thought, a 1-5 integer score normalized to 0-1. It is the "default
# English LLM judge" that eval tools ship with. We DON'T translate the rubric or
# tell the judge that Hinglish is acceptable -- that's the whole point.
#
# (We verified this matches DeepEval's GEval GPTModel output on sample cases;
#  calling the judge directly avoids DeepEval's telemetry/first-run stdin hang
#  when stdout is redirected on Windows.)

JUDGE_REF_RUBRIC = """You are an expert evaluator. Evaluate the CORRECTNESS of an AI assistant's answer.

Question:
{question}

Reference correct answer:
{gold}

AI assistant's answer:
{answer}

Evaluation criteria:
Determine whether the assistant's answer correctly and factually answers the
question, using the reference answer as the ground truth. Focus only on factual
correctness of the core answer.

Think step by step, then output a JSON object exactly like:
{{"reasoning": "<your step-by-step reasoning>", "score": <integer 1-5, where 5 = fully correct, 1 = completely wrong>}}
Output only the JSON object."""

JUDGE_NOREF_RUBRIC = """You are an expert evaluator. Judge the QUALITY of an AI assistant's answer.

Question:
{question}

AI assistant's answer:
{answer}

Evaluation criteria:
Judge whether the answer is a correct, clear, and helpful answer to the question.
A good answer is factually accurate and directly addresses the question.

Think step by step, then output a JSON object exactly like:
{{"reasoning": "<your step-by-step reasoning>", "score": <integer 1-5, where 5 = excellent, 1 = poor>}}
Output only the JSON object."""


def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def _judge_call(prompt):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw = r.choices[0].message.content
            data = json.loads(_extract_json(raw))
            sc = int(round(float(data["score"])))
            sc = max(1, min(5, sc))
            return (sc - 1) / 4.0, str(data.get("reasoning", ""))[:400]
        except Exception as e:
            print(f"  [retry {attempt}] judge error: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None, "judge failed"


def judge_ref(question, answer, gold):
    return _judge_call(JUDGE_REF_RUBRIC.format(question=question, answer=answer, gold=gold))


def judge_noref(question, answer, gold):
    return _judge_call(JUDGE_NOREF_RUBRIC.format(question=question, answer=answer))


def main():
    cases = build_cases()
    _load_st()  # warm up embedding model before the loop
    results = []

    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {c['case_id']}", flush=True)
        ans = get_answer(c["question"])
        hc = human_is_correct(c["task"], ans)
        js_ref, jr_ref = judge_ref(c["question"], ans, c["gold"])
        js_nr, jr_nr = judge_noref(c["question"], ans, c["gold"])
        em = exact_match(ans, c["gold"])
        sim = similarity(ans, c["gold"])
        row = {
            "case_id": c["case_id"],
            "task_id": c["task_id"],
            "variant": c["variant"],
            "question": c["question"],
            "gold": c["gold"],
            "answer": ans,
            "human_correct": hc,
            "judge_score": js_ref,            # reference-based (primary)
            "judge_reason": jr_ref,
            "judge_noref_score": js_nr,       # reference-free rubric
            "judge_noref_reason": jr_nr,
            "exact_match": em,
            "similarity": sim,
        }
        results.append(row)
        print(f"    human={hc}  judge_ref={js_ref}  judge_noref={js_nr}  exact={em}  sim={sim:.3f}", flush=True)
        print(f"    answer: {ans[:120]}", flush=True)
        time.sleep(1)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nWrote results.json", flush=True)

    summarize(results)


def summarize(results):
    from collections import defaultdict
    by_var = defaultdict(list)
    for r in results:
        by_var[r["variant"]].append(r)

    print("\n" + "=" * 90)
    print("SUMMARY BY LANGUAGE VARIANT")
    print("=" * 90)
    hdr = (f"{'variant':10s} {'n':>3s} {'human_ok':>9s} {'jRef_avg':>9s} {'jRef_pass':>10s} "
           f"{'jNoRef_avg':>11s} {'jNoRef_pass':>12s} {'exact_avg':>10s} {'sim_avg':>8s}")
    print(hdr)
    print("-" * 90)
    for v in ["en", "hi", "hinglish"]:
        rows = by_var[v]
        n = len(rows)
        human_ok = sum(r["human_correct"] for r in rows)
        jr_s = [r["judge_score"] for r in rows if r["judge_score"] is not None]
        jr_avg = sum(jr_s) / len(jr_s) if jr_s else 0
        jr_pass = sum(1 for r in rows if (r["judge_score"] or 0) >= 0.5)
        jn_s = [r["judge_noref_score"] for r in rows if r["judge_noref_score"] is not None]
        jn_avg = sum(jn_s) / len(jn_s) if jn_s else 0
        jn_pass = sum(1 for r in rows if (r["judge_noref_score"] or 0) >= 0.5)
        exact_avg = sum(r["exact_match"] for r in rows) / n
        sim_avg = sum(r["similarity"] for r in rows) / n
        print(f"{v:10s} {n:>3d} {human_ok:>9d} {jr_avg:>9.3f} {jr_pass:>10d} "
              f"{jn_avg:>11.3f} {jn_pass:>12d} {exact_avg:>10.3f} {sim_avg:>8.3f}")

    print("\n" + "=" * 90)
    print("KEY QUESTION 1: judge penalizing Hinglish when human says CORRECT?")
    print("=" * 90)
    for label, key, rkey in [("reference-based", "judge_score", "judge_reason"),
                             ("reference-free ", "judge_noref_score", "judge_noref_reason")]:
        print(f"\n  --- {label} judge ---")
        for v in ["en", "hi", "hinglish"]:
            rows = [r for r in by_var[v] if r["human_correct"]]
            if not rows:
                continue
            js = [r[key] for r in rows if r[key] is not None]
            avg = sum(js) / len(js) if js else 0
            passed = sum(1 for r in rows if (r[key] or 0) >= 0.5)
            print(f"    {v:10s}: {len(rows)} human-correct -> judge avg {avg:.3f}, "
                  f"passes {passed}/{len(rows)}")
        print(f"    Human-correct answers this judge marked WRONG (< 0.5):")
        any_wrong = False
        for r in results:
            if r["human_correct"] and (r[key] or 0) < 0.5:
                any_wrong = True
                print(f"      [{r['variant']}] {r['task_id']}: judge={r[key]:.2f}")
                print(f"          answer: {r['answer'][:100]}")
                print(f"          reason: {(r[rkey] or '')[:220]}")
        if not any_wrong:
            print("      (none)")

    print("\n" + "=" * 78)
    print("KEY QUESTION 2: does exact-match / similarity break across scripts?")
    print("=" * 78)
    for v in ["en", "hi", "hinglish"]:
        rows = by_var[v]
        hc = [r for r in rows if r["human_correct"]]
        em_on_hc = sum(r["exact_match"] for r in hc)
        sim_on_hc = [r["similarity"] for r in hc]
        savg = sum(sim_on_hc) / len(sim_on_hc) if sim_on_hc else 0
        print(f"  {v:10s}: of {len(hc)} human-correct, exact-match catches {em_on_hc}, "
              f"avg similarity {savg:.3f}")


if __name__ == "__main__":
    main()
