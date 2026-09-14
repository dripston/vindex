# 0.1 Empirical Validation — Does the Hinglish-eval problem exist?

**Date:** 2026-09-10
**Question:** Before building a Hinglish bot-evaluation dataset, prove empirically
whether (1) an English LLM judge penalizes correct Hinglish answers, and
(2) string / similarity scoring breaks across scripts.

## Setup

| Piece | Choice |
|---|---|
| Eval tool | DeepEval 4.2.2 (G-Eval style rubric) |
| Model under test | Groq `openai/gpt-oss-20b` |
| LLM judge | Groq `openai/gpt-oss-120b`, temp 0, English rubric, 1–5 → 0–1 |
| Similarity | `sentence-transformers/all-MiniLM-L6-v2` cosine vs gold |
| Exact match | normalized (lowercase, strip punctuation) vs gold |
| Test cases | 10 factual-QA tasks × 3 variants (English / Devanagari Hindi / Romanized Hinglish) = 30 |
| Ground truth | language-agnostic human check — fact is right in *any* script |

Two judge conditions:
- **Reference-based** — judge sees the gold answer (`Correctness`).
- **Reference-free** — judge must decide correctness from its own knowledge
  (`AnswerQuality`; the promptfoo `llm-rubric` / answer-correctness style).

Plus a **controlled probe** (`hinglish_probe.py`): 5 questions, hand-written
known-correct answers in English vs Romanized Hinglish vs code-mixed Hinglish vs
conversational Hinglish (20 judgements) — the only variable is language/script.

## Results

### Main run (30 cases), human-correct answers only

| variant | n | judge w/ ref (avg / pass@0.5) | judge no-ref (avg / pass@0.5) | exact-match | embed similarity (avg) | sim ≥ 0.5 |
|---|---|---|---|---|---|---|
| English  | 10 | 1.000 / 10 | 0.975 / 10 | 0/10 | 0.490 | 5/10 |
| Hindi    |  9 | 1.000 / 9  | 0.889 / **8** | 0/9  | **0.242** | **0/9** |
| Hinglish | 10 | 1.000 / 10 | 1.000 / 10 | 0/10 | **0.337** | **1/10** |

### Controlled probe (20 judgements, every answer factually correct)

| variant | mean no-ref judge score | perfect (1.0) |
|---|---|---|
| English            | 1.000 | 5/5 |
| Hinglish (Roman)   | 1.000 | 5/5 |
| Hinglish (code-mix)| 1.000 | 5/5 |
| Hinglish (convo)   | 1.000 | 5/5 |

### Similarity threshold sweep (human-correct answers)

| threshold | English | Hindi | Hinglish |
|---|---|---|---|
| sim ≥ 0.3 | 9/10 | 3/9 | 5/10 |
| sim ≥ 0.4 | 6/10 | 0/9 | 3/10 |
| sim ≥ 0.5 | 5/10 | 0/9 | 1/10 |
| sim ≥ 0.6 | 4/10 | 0/9 | 0/10 |

## Answers to the two key questions

### 1. Does the judge score Hinglish lower even when a human says correct?

**Mostly NO for this judge and this task, with one real failure and one systematic drift.**

- **Reference-based judge: no bias.** 29/29 human-correct answers passed, avg 1.000
  across all three languages. Giving the judge the gold string neutralizes language.
- **Reference-free judge: small but real gap.** English/Hinglish 1.000, Hindi 0.889
  with **1 hard failure**:
  - `boiling_point_water_c__hi`: answer "समुद्र तल पर पानी 100 °C पर उबलता है" (correct)
    → **score 0.0**. The judge's reasoning: *"the question asks for the boiling
    temperature of water at the **sea floor** … so the boiling point is significantly
    above 100 °C."* The judge **mistranslated** `समुद्र तल` ("sea level") as "sea
    floor" and marked a correct answer completely wrong. This is a
    comprehension/translation error on Devanagari, not a style penalty.
- **Systematic drift (subtle):** on several Hindi/Hinglish cases the
  reference-based judge's *reasoning text* silently rewrites the answer
  ("The assistant responded that water at the **sea floor** boils at 100 °C,
  matching the reference") — it reached the right verdict here only because the
  gold string bailed it out. Without a reference, this drift becomes a wrong score.
- **Controlled probe: judge is robust on short facts.** Romanized, code-mixed and
  conversational Hinglish ("Arre haan…", "H2O hota hai bhai") all scored 5/5,
  identical to English. No penalty for Roman script or informal register per se.

### 2. Does exact-match / similarity scoring break across scripts?

**YES — comprehensively. This is the strong, unambiguous finding.**

- **Exact match: 0/30.** Completely dead. Gold is a short canonical form
  ("Mumbai", "1947", "H2O"); every real answer adds context or uses another
  script. Not script-specific — it fails in English too — but it means any
  eval suite leaning on exact/normalized match is measuring nothing here.
- **Embedding similarity: collapses on non-English, and the gap is script-driven.**
  - Same fact, mean cosine vs the English gold: **EN 0.490 · Hinglish 0.337 (−31%) · Hindi 0.242 (−51%)**.
  - At the usual 0.5 pass line: English 5/10, **Hinglish 1/10, Hindi 0/9**.
  - `all-MiniLM-L6-v2` is a predominantly-English model; a correct Devanagari or
    Romanized answer lands far from an English gold string in its embedding space.
  - There is **no threshold that works for all three**: anything that passes a
    reasonable fraction of Hindi answers (≤ 0.3) also passes near-misses in English.
  - A Hindi-gold or a multilingual embedding model would shift absolute numbers,
    but the cross-script comparison (English gold vs Hinglish answer, which is the
    realistic dataset situation) stays broken.

## Verdict — does the thesis hold?

**Partly, and the useful half is solid.**

- **String/similarity scoring across scripts is definitively broken** (finding 2).
  Any Hinglish eval dataset that ships exact-match or English-embedding-similarity
  metrics will silently fail correct answers. This alone justifies building
  Hinglish-aware scoring (Hindi/multilingual gold, transliteration-normalized
  match, or judge-only).
- **English-LLM-judge bias is real but narrower than the strong claim.** A capable
  judge (120B) with a reference answer handled Hinglish fine on clean facts. The
  failure mode is **comprehension of Devanagari / code-mixed text in the
  reference-free setting** — mistranslation leading to a confidently wrong 0.0 —
  not a stylistic penalty on Romanized or informal Hinglish. Expect this to get
  worse with: weaker/cheaper judges, longer free-text answers, reasoning traces,
  ambiguous questions, and no gold reference — i.e. exactly the conditions a real
  bot-eval dataset has.

**Recommendation:** proceed, but scope the dataset's contribution as:
(a) transliteration/script-robust scoring — clear win;
(b) a judge-bias probe set that stresses the reference-free + Devanagari/code-mix +
long-answer conditions where the effect showed up, rather than claiming blanket
judge bias on all Hinglish. The half-day check says the problem is worth a
dataset; it also says be precise about which part of the problem you're solving.

## Files

- `testcases.py` — 30 cases + language-agnostic human checker
- `run_experiment.py` — main pipeline → `results.json`
- `analyze.py` — aggregates + drift detection
- `hinglish_probe.py` — controlled judge probe → `hinglish_probe_results.json`
- `run.log` — full run transcript
