"""
Targeted probe: the main run showed the model answering 'Hinglish' prompts in
Devanagari. Here we hand the judge KNOWN-CORRECT answers written in real
Romanized Hinglish (and code-mixed), plus a few longer conversational ones,
and see how the English no-reference judge scores them vs the English original.

No model-under-test here -- we control the answers so the ONLY variable is
the language/script of a factually-correct response.
"""
import io, sys, os, json, time, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = {}
for l in open(os.path.join(REPO_ROOT, ".env"), encoding="utf-8"):
    if "=" in l and not l.strip().startswith("#"):
        k, v = l.strip().split("=", 1); ENV[k.strip()] = v.strip()
from groq import Groq
client = Groq(api_key=ENV["GROQ_API_KEY"])
JUDGE = "openai/gpt-oss-120b"

RUBRIC = """You are an expert evaluator. Judge the QUALITY of an AI assistant's answer.

Question:
{q}

AI assistant's answer:
{a}

Evaluation criteria:
Judge whether the answer is a correct, clear, and helpful answer to the question.
A good answer is factually accurate and directly addresses the question.

Think step by step, then output JSON: {{"reasoning": "...", "score": <integer 1-5>}}
Output only the JSON object."""

# (question, {variant: known-correct answer}) -- every answer is factually correct.
PROBES = [
    ("What is the capital of Maharashtra?", {
        "english": "The capital of Maharashtra is Mumbai.",
        "hinglish_roman": "Maharashtra ki rajdhani Mumbai hai.",
        "hinglish_codemix": "Maharashtra ka capital Mumbai hai, jo ki state ka sabse bada city bhi hai.",
        "hinglish_convo": "Arre haan, Maharashtra ki rajdhani toh Mumbai hi hai na. Wahi financial capital bhi hai India ka.",
    }),
    ("How many planets are in the solar system?", {
        "english": "There are eight planets in the solar system.",
        "hinglish_roman": "Solar system mein aath planets hain.",
        "hinglish_codemix": "Solar system mein 8 planets hote hain, Pluto ko toh dwarf planet bana diya tha 2006 mein.",
        "hinglish_convo": "Dekho, pehle nau maante the but ab aath hi planets hain kyunki Pluto count nahi hota ab.",
    }),
    ("Who is the Father of the Nation in India?", {
        "english": "Mahatma Gandhi is the Father of the Nation in India.",
        "hinglish_roman": "Bharat ke rashtrapita Mahatma Gandhi hain.",
        "hinglish_codemix": "India ke Father of the Nation Mahatma Gandhi hain, unhe Bapu bhi bolte hain.",
        "hinglish_convo": "Gandhi ji ko hi rashtrapita kehte hain yaar, unhone hi non-violence se aazadi dilayi thi.",
    }),
    ("What is the chemical formula of water?", {
        "english": "The chemical formula of water is H2O.",
        "hinglish_roman": "Paani ka chemical formula H2O hai.",
        "hinglish_codemix": "Water ka formula H2O hota hai - do hydrogen aur ek oxygen atom.",
        "hinglish_convo": "H2O hota hai bhai, do hydrogen ek oxygen. School mein padha tha na?",
    }),
    ("In which direction does the sun rise?", {
        "english": "The sun rises in the east.",
        "hinglish_roman": "Suraj purab disha mein ugta hai.",
        "hinglish_codemix": "Sun east direction mein rise karta hai, Earth ki rotation ki wajah se.",
        "hinglish_convo": "Purab mein ugta hai na suraj, ye toh basic hai.",
    }),
]

VARIANTS = ["english", "hinglish_roman", "hinglish_codemix", "hinglish_convo"]


def score(q, a):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=JUDGE,
                messages=[{"role": "user", "content": RUBRIC.format(q=q, a=a)}],
                temperature=0.0, max_tokens=800,
                response_format={"type": "json_object"},
            )
            d = json.loads(r.choices[0].message.content)
            s = max(1, min(5, int(round(float(d["score"])))))
            return (s - 1) / 4.0, str(d.get("reasoning", ""))[:300]
        except Exception as e:
            print("  retry", attempt, e, flush=True)
            time.sleep(4 * (attempt + 1))
    return None, "failed"


rows = []
for q, ans in PROBES:
    for v in VARIANTS:
        s, reason = score(q, ans[v])
        rows.append({"q": q, "variant": v, "answer": ans[v], "score": s, "reason": reason})
        print(f"[{v:17s}] score={s}  {ans[v][:70]}", flush=True)
        time.sleep(1)

json.dump(rows, open("hinglish_probe_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("MEAN NO-REFERENCE JUDGE SCORE by variant (all answers factually correct)")
print("=" * 70)
from collections import defaultdict
bv = defaultdict(list)
for r in rows:
    if r["score"] is not None:
        bv[r["variant"]].append(r["score"])
for v in VARIANTS:
    xs = bv[v]
    print(f"  {v:18s}: mean {sum(xs)/len(xs):.3f}  pass@0.5 {sum(x>=0.5 for x in xs)}/{len(xs)}  "
          f"perfect(1.0) {sum(x==1.0 for x in xs)}/{len(xs)}")

print("\nCases where judge scored a CORRECT answer below 1.0:")
for r in rows:
    if r["score"] is not None and r["score"] < 1.0:
        print(f"  [{r['variant']}] {r['q']}")
        print(f"     answer: {r['answer']}")
        print(f"     score:  {r['score']}")
        print(f"     reason: {r['reason']}")
