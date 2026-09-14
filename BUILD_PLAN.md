# BUILD PLAN — Complete Product

An Indic-aware metrics library for evaluating LLM output.

Five milestones. Roughly 10 weeks at 12 to 15 hours a week.

---

## THE PRODUCT IN ONE PARAGRAPH

Every LLM evaluation framework in use today (DeepEval, Ragas, promptfoo, Phoenix) assumes English. Indian companies ship vernacular LLM features and evaluate them with these tools, which silently fail correct answers, pass wrong ones, and cannot tell whether the model even replied in the right script. Sarvam has built the ASR half of this problem. Nobody has built the LLM-answer half. This library fills that gap with five metrics and plugs into the frameworks people already use.

**The evidence behind it**, all from your own runs: exact match scores 0/30 across scripts. Default similarity thresholds perform at chance in 13 of 15 encoder-language configurations. MuRIL, the obvious Indic encoder, scores 0.99 on everything and cannot discriminate at all. A model told to match the user's script ignores that instruction 80% of the time on Romanized input. And an LLM judge silently mistranslated Devanagari inside its own reasoning and scored a correct answer 0.0.

---

## GROUND RULES

**Ship M1 before starting M2.** One metric live beats four half-built. This is the rule most likely to be broken and most costly to break.

**Deterministic by default.** No LLM call unless nothing else works. Your own data shows the model produces different wording a third of the time at temperature 0, so an LLM-based metric inherits non-determinism. "Your evaluation metric should not itself be non-deterministic" is a line for the README.

**Align-then-judge, when an LLM is unavoidable.** Sarvam's shape: diff first, send only the mismatched segments to the model, rescore. Bounds cost, latency and variance to the subset that actually disagrees.

**Nothing deleted.** Experiment scripts get archived, never removed. `git mv`, history preserved.

**Every claim carries its sample size.** 10 cases per cell, one model family, Hindi-dominant. Overclaiming gets found out in a week.

**No GPU.** Nothing in this plan needs one.

---

## MILESTONE 0 — CLEAN AND SCAFFOLD
**Week 1, first half.**

### 0.1 Audit
`ARCHITECTURE.md`: one row per file. What it does, exploratory or reusable, dependencies, whether outputs are still valid. Flag everything still reading the contaminated `results.json`.

### 0.2 Reorganise

```
experiments/          one-off scripts and outputs, moved as-is
experiments/README.md what each asked, found, and whether it survived
data/                 results_clean.json (pinned), trap words, NOTICE.md
src/<package>/        the package
tests/
docs/
```

`experiments/README.md` must record what was superseded: the encoder rerun replaced the original script-gap claim, the contamination fix invalidated every pre-fix Hinglish figure. Future-you will not remember.

### 0.3 Name it
Check PyPI and GitHub. **You choose**, not your agent. Claim it the same day.

### 0.4 Scaffold
`pyproject.toml`, src layout, MIT licence. ruff + mypy + pytest clean. GitHub Actions on every push. Green badge before any metric exists.

### 0.5 The result object
Every metric returns the same shape. Get it right once:

```python
result.score    # float 0-1
result.passed   # bool
result.label    # metric-specific string
result.reason   # one human sentence
result.detail   # dict, raw evidence
```

`reason` is what appears in a red CI job at 11pm. Write it for that person.

**Exit:** `pip install -e .` works, CI green, result object tested.

---

## MILESTONE 1 — `script_adherence`
**Week 1, second half, into week 2.**

Did the response come back in the script the user wrote in?

Nobody else has this. Sarvam checks transcription fidelity against a reference; this checks a property of the answer itself, with no reference needed.

### 1.1 Port the engine
`script_check.py` becomes `script.py`. **Keep the character-counting logic exactly as written** — it's tested and validated on 90 real responses. Clean the API around it, don't touch the core. Your agent will want to refactor it. Don't let it.

### 1.2 Extend to 8 scripts
Devanagari, Kannada, Tamil, Telugu, Bengali, Gujarati, Malayalam, Odia, Gurmukhi. One test per script, real sentences.

### 1.3 Latin-script language detection
Script can't separate English from Romanized Hindi; both are Latin. Function-word heuristic to start (*hai, hain, kya, nahi, mera, aap, ka, ki, ke, se, mein, tha, hoga, raha*). No heavy dependency yet. Mark it `v0` in the docstring and state its limits honestly.

### 1.4 The metric
```python
script_adherence(prompt, response) -> MetricResult
```
- native-script prompt → native or mixed = pass
- Romanized prompt → Roman = pass
- code-mixed prompt → Roman or mixed = pass

Labels: `matched`, `script_mismatch`, `language_mismatch`, `mixed`, `empty`.

### 1.5 Tests
Every script, every label, empty, whitespace-only, numerals-only, emoji, punctuation-only, mixed-script response, third-language response.

### 1.6 Validate — the real test
Run over both `results.json` and `results_clean.json`. Must reproduce **20% Hinglish adherence under the original instruction, 100% under the strict one**. If it doesn't, the port is broken. This is why the contaminated file stays in the repo.

### 1.7 Ship
README: install, one example, the 20%/100% table, limitations. No marketing copy — the table is the argument. Test PyPI, then PyPI, tag `v0.1.0`.

**Exit:** `pip install <package>` works from a clean environment.

---

## MILESTONE 2 — `script_normalized_match`
**Week 3.**

Transliterate both sides to a common representation before comparing, so "namaskara", "ನಮಸ್ಕಾರ" and "namaskār" stop being three different strings.

SN-WER (arXiv 2606.02548) established this principle for ASR. You apply it to answer scoring and cite them.

### 2.1 Pick a backend
Evaluate `indic_transliteration_py` and IndicXlit on your own 30 cases. Pick one, document why, wrap it behind an interface so it can be swapped. **Do not write your own transliterator.**

### 2.2 Normalization pipeline
Transliterate → lowercase → strip diacritics → normalize whitespace and punctuation → reconcile Indic versus Arabic numerals.

### 2.3 Three match modes
Exact-after-normalization, token-level F1, character-level similarity.

### 2.4 Beat the baseline
Your exact-match baseline is 0/30. Report the improvement with the number attached.

### 2.5 Document the failures honestly
Transliteration is many-to-many. The ट्यून versus तूने case is a genuine ambiguity that Jio's own in-house layer doesn't fully resolve — their engineer said so directly. Put it in the docs.

**One place you beat Sarvam:** their loanword problem ("वह doctor" versus "वह डॉक्टर") is solved with an LLM call. A transliteration lookup solves it deterministically, for free, reproducibly. Say so.

---

## MILESTONE 3 — `calibrated_similarity`
**Week 4.**

Multilingual encoder similarity with per-encoder, per-script thresholds instead of a default 0.5 that performs at chance.

### 3.1 Ship the calibration table
From your discrimination run: per encoder, per script, the threshold that maximised accuracy. State plainly it came from 10 cases per cell.

### 3.2 Sensible defaults, loud warnings
Default to mpnet or e5. **Warn loudly if someone passes MuRIL**, with a link to the HindiWiC finding: 55% zero-shot, chance level, reaching 90% only after Hindi-specific fine-tuning. That's the trap an Indian engineer walks into first, because MuRIL is the obvious Indic pick.

### 3.3 `calibrate()`
Let users fit thresholds on their own labelled data. Your table is a starting point, not a truth.

### 3.4 Wire in caching
`encoder_cache.py` already exists. Nobody has a GPU; encoding is the slow part.

### 3.5 The argument for the metric
13 of 15 configurations at chance accuracy with a default 0.5 threshold. Tuned, the good encoders reach 0.85 to 0.95. That table goes in the docs.

---

## MILESTONE 4 — INTEGRATIONS AND LAUNCH
**Week 5.**

Small work, disproportionate return. You stop competing with DeepEval and start extending it.

### 4.1 Three adapters
DeepEval custom metric (subclass `BaseMetric`). Promptfoo Python assertion plus a YAML example. Ragas custom metric. Roughly 50 lines each.

### 4.2 Copy-pasteable examples
In each tool's idiom, not yours.

### 4.3 Go upstream
Open an issue or discussion on each project offering the integration or asking to be listed in their docs. Costs them nothing, expands their coverage, drives real installs.

### 4.4 The writeup
Title it after the finding, not the tool: *"Your LLM eval suite is broken for Indian languages. Here's the data."*

1. Indian companies ship vernacular LLM features; eval tools assume English
2. Script adherence: 20% versus 100%, same model, one word changed
3. Exact match: 0/30 across scripts
4. Default thresholds: chance accuracy in 13 of 15 configurations
5. MuRIL, the obvious pick, cannot discriminate at all
6. Why: cite HindiWiC on zero-shot polysemy failure, SN-WER on script bias, Script Gap on comprehension failure, Sarvam on the ASR half
7. Limitations, plainly
8. The tool, last

### 4.5 Distribution
**Day 1:** LinkedIn with charts posted natively — for an Indian audience this outperforms HN. Then r/LocalLLaMA, r/MachineLearning.
**Day 2:** Hacker News (Tue–Thu, 8–10am ET, one shot, no vote-rallying). Indian AI Discord and Slack communities.
**Day 3:** Email Minakshi, who asked to see the results. Rudra Murthy. The DeepEval, promptfoo and Ragas communities, framed as "we added Indic support." AI4Bharat and Indic NLP groups. **Sarvam** — you extended their framework to a layer they don't cover, and they're in Bangalore.

Answer every comment. The comments build more credibility than the post.

---

## THE DECISION POINT
**Weeks 6 to 7.**

Ship, post, then watch. Installs, issues, questions, language requests.

**If people use it:** build M5, add languages by request, keep going.
**If the post lands but nobody installs:** you have a finding, a writeup, a published package and an interview story, and you stopped at five weeks instead of ten.

Both outcomes are fine. Decide with data rather than hope.

During these two weeks: answer every issue within 24 hours, ship what people ask for rather than what you find interesting, and get to three named users you can talk to.

---

## MILESTONE 5 — `judge_trace_check`
**Weeks 8 to 10. The differentiator.**

Inspect the judge's own reasoning for mistranslation of the source. The thing Minakshi called the worrying one, because the error happens inside the model's head before any output exists, so neither WER nor a transliteration layer can catch it.

### 5.1 Dictionary first, not LLM
You have the HindiWiC inventory: 60 polysemous Hindi nouns, plus your own compounds and number words. Fill in `reading_a` / `reading_b` by hand — about an hour, and it's your work, not your agent's.

The check then becomes mechanical: source contains समुद्र तल, trace says "sea floor" instead of "sea level" → flag. Source has उत्तर meaning north, trace says "answer" → flag.

Deterministic, free, instant, reproducible. No meta-judge, no kappa study.

### 5.2 The honest claim
"Detects mistranslation of N known ambiguous terms," not "detects mistranslation." Smaller claim, true claim. Grow the dictionary from user reports.

### 5.3 Validate with ~60 traces
Not a publishable corpus — enough to report precision and recall honestly.

Generate ~30 trap-seeded questions, model answers from a pinned run, judge verdicts **reference-free** with full reasoning traces captured.

You grade all 60. One other fluent Hindi speaker grades all 60 independently. ₹1,000, 90 minutes. Compute agreement.

**Bias protocol, mandatory since you're both annotator and builder:**
- Commit labels to git **before** writing a line of the metric
- Blind the sheet: no model identities, no cross-visibility between annotators
- Freeze the rubric before grading; if it changes, re-grade from the start
- Hold out 30%, never look at it while tuning
- Disclose the self-annotation in the docs

### 5.4 Optional LLM fallback
If the dictionary misses a category it structurally cannot catch, add an LLM mode using Sarvam's align-then-judge shape: diff first, model only on mismatches, default to flagging when ambiguous. Two modes, clearly documented, user chooses.

### 5.5 Publish the number whatever it is
A metric catching 60% of mistranslations at a 15% false positive rate, honestly reported, beats a vague claim of working well.

### 5.6 v0.2 and the second writeup
Second posts outperform first posts, because you now have an audience.

---

## WHAT YOU ALREADY HAVE

| Asset | Becomes |
|---|---|
| `script_check.py`, 30 tests, 90 responses validated | M1's engine |
| Adherence result, 20% vs 100% | M1's headline and launch hook |
| Encoder comparison, 5 encoders | M3's calibration table |
| Discrimination run, ROC AUC, thresholds | M3's chance-level finding |
| `encoder_cache.py` | M3's performance layer |
| `results_clean.json`, pinned and committed | the benchmark set |
| HindiWiC trap inventory + own additions | M5's dictionary |
| Nine cited papers | the writeup's related work |
| Two named practitioner confirmations | the problem statement |

Three weeks of experiments produced the evidence base. None of it is wasted.

---

## TIMELINE

| Week | Build |
|---|---|
| 1 | M0 clean and scaffold, M1 starts |
| 2 | **M1 ships, v0.1.0 on PyPI** |
| 3 | M2 normalized match |
| 4 | M3 calibrated similarity |
| 5 | M4 integrations, **launch** |
| 6–7 | **Decision point.** Issues, users, signal |
| 8–10 | M5 judge trace check, **v0.2** |

---

## HONEST RISKS

**M1 and M2 aren't very differentiating alone.** Script detection and normalized matching are a competent week's work for anyone. What defends them is your calibration data and the writeup. M5 is the real moat.

**Nobody has said they'd install this.** Interest is not demand. One engineer called the problem worrying; none said they'd add a dependency. `script_adherence` might be a five-line inline check rather than a package. The decision point exists to find that out cheaply.

**The audience is smaller than a global dev tool's.** Two named users at Indian AI companies beats 500 stars here. Don't measure against the wrong number.

**Someone could ship this while you build.** The gap is visible to anyone who looks. If it happens: read theirs, find what they got wrong, write about it. Second with better analysis is still a career.

**Your samples are small.** State it everywhere.

---

## THE ONE-LINE VERSION

Clean the repo, scaffold, ship one metric you already have the engine for, prove it reproduces 20% versus 100%, publish, post, let the response decide whether you build the other four.
