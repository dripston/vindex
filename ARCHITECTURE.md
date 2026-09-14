# ARCHITECTURE.md — file-by-file audit (Milestone 0.1)

One row per file that currently exists in the repo (excluding `.git/`,
`__pycache__/`, `encoder_cache/`, and `data/external/HindiWiC/`, which is
a gitignored clone of third-party data — see `data/trap_words/NOTICE.md`).

## Corrections applied in Milestone 0.2 (reorganisation)

The table below is the original 0.1 audit and describes the **pre-reorg**
repo layout (root-level `results.json`, `scripts/`, `results/`, etc.).
Three corrections were applied on top of it during the 0.2 reorganisation;
read these first, then treat every path in the table below as historical
— the current layout is `experiments/`, `data/`, `src/vindex/`, `tests/`,
`docs/`, with `script_check.py` kept at repo root. See
`experiments/README.md` for the current, path-accurate file notes.

**Correction 1 — the monkeypatch is gone.** `scripts/rerun_all.py` used
to run the clean-data pipeline by monkeypatching `json.load` at runtime,
so `encoder_comparison.py`, `discrimination.py`, and `minimal_edit.py`
(via `testcases.py`) would only read clean data when invoked *through*
the orchestrator — running any of those four scripts directly regenerated
contaminated output silently. Fixed: each of the four now defaults to
`data/results_clean.json` via an explicit module constant or function
parameter (`--data PATH` CLI override on the three standalone-runnable
ones), `rerun_all.py` passes the path explicitly instead of patching
anything, and there is no implicit redirection left anywhere in the repo.
`scripts/contamination_impact.py` and `scripts/script_adherence_report.py`
are the two scripts whose entire job is the before/after diff, so they
read `results.json` on purpose — each now names the path explicitly with
a comment saying why, in its own code.

**Correction 2 — the aborted adherence run was deleted, not kept.**
`results_adherence/checkpoint.jsonl` (552 of a planned 1800 rows) was
abandoned mid-run during a model swap (`qwen/qwen3.6-27b` hit a hard
output-token-per-minute cap; swapped to `qwen/qwen3.8-27b`). Deleted
rather than moved — half a run reads as a finished one to anyone who
finds it later. The full 150×4×3 sweep has not been rerun; it is on the
roadmap, not in the evidence base. This does **not** affect the 20%/100%
headline finding, which comes from the separate, already-complete
`scripts/script_adherence_report.py` over the 30-case set. See
`experiments/README.md`.

**Correction 3 — `analyze.py` reclassified.** Previously described below
as "broken, no clean-data path." Corrected: it is not broken, it is
**archived — do not run.** It answered a real question in week one and
was superseded by `scripts/contamination_impact.py` and
`scripts/discrimination.py`, not invalidated by a bug. The distinction
matters when this file is read months from now.

**Columns:**
- **What it does** — one line.
- **Kind** — `exploratory` (one-off, answered a question, not meant to be
  reused as a library) or `reusable` (intended to survive into `vindex`
  or is load-bearing infrastructure other files depend on).
- **Depends on** — other repo files it imports or reads at runtime. Blank
  if self-contained (stdlib + third-party packages only).
- **Reads `results.json`?** — Y/N/indirect. `results.json` is the
  **contaminated** file: `run_experiment.py`'s original run asked the
  model to "reply in the same language and script" and the model answered
  Romanized-Hindi ("Hinglish") prompts in Devanagari script 8 times out of
  10. `results_clean.json` (pinned, committed, frozen — see its own
  header comment and `scripts/regenerate_dataset.py`'s guard) is the fix.
  Any file marked Y or "indirect" here produced numbers that are only
  valid for the *comparison itself* (old vs. new), never as a standalone
  finding, unless it explicitly reads the clean file too.
- **Outputs still valid?** — whether what this file *produced* (if
  anything) is still something we'd point to today.

---

## Root-level files

| File | What it does | Kind | Depends on | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|---|
| `BUILD_PLAN.md` | The 5-milestone plan for turning this repo into the `vindex` package (named `pramana` in early planning; `pramana` was found to be taken on PyPI during Milestone 0.3 and the package was renamed to `vindex`). Not code. | reusable (planning doc) | — | N | Yes — this is the plan being executed. |
| `FINDINGS.md` | Write-up of the very first experiment (0.1): does exact-match/similarity break across scripts, does an English LLM judge penalize Hinglish. Uses `run_experiment.py` + `results.json` + `all-MiniLM-L6-v2` only. | exploratory | `run_experiment.py`, `results.json` (as data, referenced) | indirect (documents results computed from it) | **Partially superseded.** The "does judge bias exist" and "does exact-match break" conclusions stand on their own logic. But the specific EN/HI/Hinglish similarity numbers it quotes came from (a) an English-only encoder (superseded by the multi-encoder rerun — the "script gap" turned out to be an encoder artifact, not a script property) and (b) pre-contamination-fix data (the Hinglish column was often secretly Devanagari). See `experiments/README.md`. |
| `analyze.py` | Standalone script: reads `results.json` directly and prints a deeper per-case breakdown (similarity gaps, threshold sweep, judge mistranslation flags) for the `FINDINGS.md` write-up. | **archived — do not run** (Correction 3: reclassified from "broken"; it answered a real question in week one and was superseded, not broken) | `results.json` (hardcoded, unconditional, by design — this file's whole job was a `results.json`-only deep-dive) | **Y** (hardcoded; archived rather than given a clean-data path, since it's superseded) | No — contaminated input only. Superseded by `scripts/contamination_impact.py` and `scripts/discrimination.py`. |
| `run_experiment.py` | The **original** experiment: generates 30 model answers (EN/HI/Hinglish × 10 tasks) with the "same language and script" system prompt, scores them with a DeepEval LLM judge + exact-match + MiniLM similarity, **writes `results.json`**. | exploratory | Groq API, `testcases.py`, DeepEval, sentence-transformers | **writes** it (not a reader) | The *generation event* that produced the contaminated file. Its own analysis (printed summary) is superseded by later runs; the file it produces (`results.json`) is kept on purpose as the "before" side of every contamination comparison. |
| `hinglish_probe.py` | Standalone control experiment: hand-writes 5 known-correct Q&A pairs in EN/Roman-Hinglish/code-mix/conversational style and scores them with the same LLM judge, to isolate whether judge bias is about *script* or about *correctness confound*. | exploratory | Groq API only | N | Yes, still valid as a small controlled probe — result: no judge penalty for pure script/register once correctness is held constant. Superseded in scope (not in validity) by the later `discrimination.py` / `minimal_edit.py` work, which does this properly at n=90–145 instead of n=20. |
| `hinglish_probe_results.json` | Output of `hinglish_probe.py`. | exploratory (data) | — | N | Yes, small-n but valid for what it is. |
| `results.json` | **The contaminated dataset.** 30 model answers, EN/HI/Hinglish, generated under the "same language and script" instruction. 8/10 Hinglish cases came back in Devanagari. | reusable *as a fixed comparison artifact only* | produced by `run_experiment.py` | — (this *is* the file) | Kept deliberately as the "before" half of the contamination story. Never to be treated as a clean benchmark. |
| `results_clean.checkpoint.json` | Resumability checkpoint written by `scripts/regenerate_dataset.py` during generation (append-as-you-go, in case of a crash). | exploratory (transient artifact) | — | N | Redundant with `results_clean.json` (the final, complete output). Gitignored already. No independent value. |
| `results_clean.json` | **The pinned, frozen, fixed dataset.** Same 30 questions, regenerated with a strict script-enforcing prompt + retry loop. 29/30 script-adherent; one case (`father_of_nation_india__hi`) excluded as a genuine empty-answer failure, not a script failure. Committed to git (commit `cbf516d`) and explicitly marked DO NOT REGENERATE (the model is non-deterministic even at temperature 0 — confirmed empirically, ~1/3 of cases reword on a re-run). | **reusable — the benchmark set** | produced by `scripts/regenerate_dataset.py` | — (this *is* the fix) | Yes. This is the frozen ground truth going forward. |
| `script_adherence_cases.py` | 150 hand-authored prompts (50 questions × devanagari/roman/codemix) for the scaled-up script-adherence experiment. Independent of the contamination-experiment files by design. | **reusable — test fixture** | — | N | Yes. Intended as Milestone 1's validation fixture (`1.6` in BUILD_PLAN.md) alongside `results.json`/`results_clean.json`. |
| `script_check.py` | The script classifier: `count_scripts()`, `classify()`, `expected_script()`, `is_script_adherent()`. Devanagari-vs-Latin character counting + heuristic thresholds. Has its own 30-assertion unit-test block (`_run_tests()`). | **reusable — the core engine of Milestone 1** | — (stdlib `re` only) | N (one comment references `results.json` as a *docstring example*, does not read it) | Yes — this is the validated, load-bearing piece. **BUILD_PLAN.md 1.1 says: port unchanged. Do not refactor in Milestone 0.** |
| `testcases.py` | Defines the 10 base tasks (question × EN/HI/Hinglish), gold answers (short + full-sentence), hand-authored `wrong_hard`/`wrong_subtle` negatives, and case-builder functions (`build_cases`, `build_discrimination_cases`) that **join in real model answers by reading `results.json` directly** (hardcoded path, no clean-data parameter). | exploratory (test-fixture-as-code) | reads `results.json` (hardcoded) | **Y** (hardcoded) | The hand-authored question/gold/negative *data* is still good. But `build_discrimination_cases()`'s hardcoded `results.json` read is the reason `scripts/rerun_all.py` has to monkeypatch `json.load` at runtime to point it at clean data instead — this file itself was never edited to take a `dataset_path` parameter. |

## `scripts/`

| File | What it does | Kind | Depends on | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|---|
| `scripts/contamination_impact.py` | Reads **both** `results/discrimination_summary.csv` (old) and `results_clean/discrimination_summary.csv` (new) and produces the before/after comparison CSV + chart. Pure CSV arithmetic, no re-encoding. | **reusable — the comparison report** | `results/`, `results_clean/` (both, by design) | **indirect** (reads a CSV derived from it, on purpose, as the "old" side of a diff) | Yes — this *is* the contamination write-up. |
| `scripts/discrimination.py` | Scores correct-vs-wrong-answer discrimination (ROC AUC, separation, accuracy) across 5 encoders × gold configs × labels. Calls `testcases.build_discrimination_cases()` for its data. | reusable *(pending contamination fix at the call site)* | `encoder_comparison.py`, `testcases.py` (→ hardcoded `results.json`) | **indirect via `testcases.py`** | Its logic/metrics are reusable, but every direct invocation without `rerun_all.py`'s monkeypatch reads contaminated data. The clean-data run exists (`results_clean/discrimination_summary.csv`, produced via `rerun_all.py`) and *that* output is valid. |
| `scripts/discrimination_run.log` | Captured stdout from a `discrimination.py` run. | exploratory (log) | — | — | Gitignored (`*.log`). No action needed. |
| `scripts/encoder_cache.py` | On-disk embedding cache keyed by `(encoder, is_query, sha256(text))`. Built but not yet wired into any script (no script currently imports it). | **reusable — earmarked for Milestone 3 (3.4)** | — (numpy only) | N | Untested in anger (no cache hits/misses observed in a real run yet), but the logic was unit-tested standalone and works. |
| `scripts/encoder_comparison.py` | Defines `EncoderWrapper` (the shared 5-encoder abstraction: MiniLM, LaBSE, multilingual-e5, MuRIL, paraphrase-mpnet — handles e5's query/passage prefixing and MuRIL's manual mean-pooling) and a `main()` that scores encoder-vs-gold similarity. `load_answers()` defaults to reading `results.json`. | **reusable — `EncoderWrapper` is load-bearing** for `discrimination.py` and `minimal_edit.py`, which import it | — | **Y by default** (`load_answers()`'s path is a module-level global, only overridden by `rerun_all.py`'s monkeypatch) | The wrapper class is solid and reused everywhere. Direct default-config runs are contaminated; the clean run exists via `rerun_all.py` → `results_clean/encoder_comparison.csv`. |
| `scripts/encoder_comparison_run.log` | Captured stdout. | exploratory (log) | — | — | Gitignored. No action needed. |
| `scripts/extract_trap_words.py` | Loads the (gitignored, locally-cloned) HindiWiC CSVs, extracts a word-only inventory (no sentence text — see licensing note in the file and `data/trap_words/NOTICE.md`), builds a hand-seeded "own additions" word list, writes both + a NOTICE.md. | **reusable — feeds Milestone 5's dictionary** | `data/external/HindiWiC/*.csv` (gitignored, not part of this repo) | N | Yes. Independent of the contamination story entirely. |
| `scripts/minimal_edit.py` | Minimal-edit negative-construction method: edits the model's own correct answer text in place (one entity/detail swapped) rather than hand-writing a whole new wrong answer, to isolate whether discrimination scores were detecting *correctness* or *authorship style*. `DATASET_PATH` defaults to `results.json`. | reusable *(pending contamination fix at the call site)* | `encoder_comparison.py`, `discrimination.py` (for `roc_auc`/threshold helpers), `testcases.py` (for `TASKS`/`VARIANTS`) | **Y by default**, overridden by `rerun_all.py` | Logic is reusable; default-config output is contaminated, clean output exists at `results_clean/minimal_edit_*.csv` via `rerun_all.py`. |
| `scripts/regenerate_dataset.py` | **The fix.** Regenerates the 30 answers with an escalating, explicit, script-forbidding system prompt + up-to-5-attempt retry, classifying each attempt with `script_check.py`. Produces `results_clean.json`. Carries an explicit top-of-file warning against re-running it (model non-determinism at temp 0, confirmed empirically). | **reusable — historical record of the fix**, not meant to run again | `script_check.py`, `testcases.py` (for the original 30 questions only, not for its `results.json` read) | N (writes `results_clean.json`, doesn't read `results.json`) | Frozen by design — this *produced* the pin, it should not be re-run. |
| `scripts/rerun_all.py` | Orchestrator: runs `encoder_comparison.py` → `discrimination.py` → `minimal_edit.py` against `results_clean.json` by monkeypatching `RESULTS_DIR`/`DATASET_PATH` module globals and redirecting `json.load` for `testcases.py`'s hardcoded read — **without editing any of those files**. Then calls `contamination_impact.main()`. | **reusable — the glue that makes the clean rerun possible** | `verify_clean_dataset.py`, `encoder_comparison.py`, `discrimination.py`, `minimal_edit.py`, `contamination_impact.py`, `testcases.py` | N directly; its entire job is to *redirect other files away from* `results.json` | Yes — this is why `results_clean/` contains valid numbers at all, without having to rewrite three other scripts. |
| `scripts/script_adherence.py` | The scaled, independent script-adherence experiment: 150 prompts × 4 conditions × 3 Groq models = 1800 calls, Wilson CI, two-proportion z-test, stacked confusion chart. Explicitly does not touch `results.json`/`results_clean.json`/`testcases.py`. | **reusable — the real validation data for Milestone 1 (1.6)** | `script_check.py`, `script_adherence_cases.py`, Groq API | N (by design) | **Incomplete.** The run was aborted partway (see `results_adherence/checkpoint.jsonl` below) after discovering `qwen/qwen3.6-27b` was unworkable (visible `<think>` reasoning eating the whole token budget, and a hard 1000 output-tokens/minute account limit). Swapped to `qwen/qwen3.8-27b` in the code, but the full 1800-call run has not completed. **This needs a fresh full run before its numbers can be trusted or published.** |
| `scripts/script_adherence_report.py` | Compares script adherence under the **original** prompt (from `results.json`, classified after the fact with `script_check.py` — it was never classified at generation time) vs the **strict** prompt (from `results_clean.json`, already classified during generation). Reproduces the headline 20% → 100% Hinglish number BUILD_PLAN.md cites. | **reusable — produced the 20%/100% headline finding** | `script_check.py`, `results.json`, `results_clean.json` (both, by design — this *is* the before/after) | **Y and indirect** (reads both, intentionally, as the point of the file) | Yes — `results_clean/script_adherence.csv` is the validated output. This is the small-n (10 questions × 1 model) predecessor to the scaled `script_adherence.py` experiment; BUILD_PLAN.md 1.6 asks for exactly this comparison to be reproduced by the ported engine. |
| `scripts/verify_clean_dataset.py` | Asserts `results_clean.json` invariants: every non-excluded case is script-adherent, no empty answers, n=10/10/9 per variant, hi/hinglish near-duplicate ratio < 0.6 (`difflib.SequenceMatcher`). Exit-code gate used by `rerun_all.py`. | **reusable — the dataset's regression test** | `results_clean.json` only | N | Yes, passes cleanly as of the last recorded run. |

## `results/` (contaminated-data outputs)

| Path | What it is | Kind | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|
| `results/encoder_comparison.csv`, `results/encoder_comparison.png`, `results/encoder_summary.csv` | Output of `encoder_comparison.py` run against `results.json` directly (default config, no monkeypatch). | exploratory (data) | Y (this *is* a `results.json`-derived artifact) | **No, not standalone** — only valid as the "old" half of `scripts/contamination_impact.py`'s comparison. Never cite these numbers alone. |
| `results/discrimination_per_case.csv`, `results/discrimination_summary.csv`, `results/auc_by_encoder.png`, `results/score_distributions.png` | Output of `discrimination.py` run against `results.json`-derived cases (via `testcases.py`). | exploratory (data) | Y | Same as above — "old" comparison half only. |

## `results_clean/` (clean-data outputs)

| Path | What it is | Kind | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|
| `results_clean/encoder_comparison.csv`, `.png`, `encoder_summary.csv` | Output of `encoder_comparison.py` run via `rerun_all.py` against `results_clean.json`. | reusable (data) | N | Yes. |
| `results_clean/discrimination_per_case.csv`, `discrimination_summary.csv`, `auc_by_encoder.png`, `score_distributions.png` | Output of `discrimination.py` via `rerun_all.py`, clean data. | reusable (data) | N | Yes. |
| `results_clean/minimal_edit_per_case.csv`, `minimal_edit_summary.csv`, `authorship_check.csv`, `auc_handwritten_vs_minimal.png` | Output of `minimal_edit.py` via `rerun_all.py`, clean data. | reusable (data) | N | Yes. |
| `results_clean/contamination_impact.csv`, `before_after_auc.png` | Output of `scripts/contamination_impact.py` — reads both old and new summaries by design. | reusable (data) | indirect (compares against it) | Yes — this is the headline before/after artifact. |
| `results_clean/script_adherence.csv` | Output of `scripts/script_adherence_report.py` — the small-n (n=30/condition) 20%→100% finding. | reusable (data) | indirect (compares against it) | Yes — but see `scripts/script_adherence.py` above: a larger, independent replication (150×4×3) was started and aborted, not yet complete. |

## `results_adherence/` (in-progress / incomplete)

**DELETED in Milestone 0.2 — see "Correction 2" at the top of this file.**
`results_adherence/checkpoint.jsonl` (552 of a planned 1800 rows) was
deleted outright rather than kept or moved: it was gitignored, never
committed, and half a run reads as a finished one to anyone who finds it
later. The row below is kept for the historical record of what existed
at the time of the 0.1 audit.

| Path | What it is | Kind | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|
| `results_adherence/checkpoint.jsonl` *(deleted)* | Partial checkpoint (552 of a planned 1800 rows) from `scripts/script_adherence.py`, aborted mid-run when `qwen/qwen3.6-27b` proved unworkable and the code was changed to `qwen/qwen3.8-27b` mid-flight. | exploratory (incomplete artifact) | N | **No — incomplete and partly generated under a model choice the code no longer uses.** Deleted per Correction 2; the full 150×4×3 sweep is on the roadmap, not in the evidence base. |

## `data/trap_words/`

| Path | What it is | Kind | Reads `results.json`? | Outputs still valid? |
|---|---|---|---|---|
| `data/trap_words/hindiwic_inventory.csv` | 60-word inventory extracted from the (gitignored, locally-cloned) HindiWiC dataset — word, instance count, inferred sense-cluster count, split presence. No sentence text (licensing constraint — see NOTICE.md). Manual columns (`suggested_reading_a/b`, `trap_viability`, `domain_fit`) intentionally blank. | **reusable — Milestone 5's dictionary seed** | N | Yes. |
| `data/trap_words/own_additions.csv` | 15 hand-picked words covering categories HindiWiC doesn't include (misleading compounds, tense-flipping time adverbs, fractional numbers, Indian large-number words). Same blank manual columns. | **reusable — Milestone 5's dictionary seed** | N | Yes. |
| `data/trap_words/NOTICE.md` | Source, citation, license-absence statement, and exactly what was/wasn't extracted from HindiWiC, and why. | reusable (compliance doc) | N | Yes. |

---

## Summary: every file that reads `results.json` (the contamination flag)

Per the audit instruction — flagged, not fixed:

| File | How it reads it | Fix already applied elsewhere? |
|---|---|---|
| `analyze.py` | Hardcoded, unconditional | No — this script has no clean-data path at all. |
| `testcases.py` (`build_cases`, `build_discrimination_cases`) | Hardcoded path, no parameter | Yes, but only via `scripts/rerun_all.py`'s runtime `json.load` monkeypatch — the file itself is unchanged. |
| `scripts/encoder_comparison.py` (`load_answers`) | Default `REPO_ROOT/results.json`, module-level global | Yes, via `rerun_all.py` overriding `RESULTS_DIR`/patching `json.load`. |
| `scripts/minimal_edit.py` (`DATASET_PATH`) | Default `REPO_ROOT/results.json`, module-level global | Yes, via `rerun_all.py` setting `me.DATASET_PATH = CLEAN_JSON_PATH` directly (real path, no monkeypatch needed here). |
| `scripts/discrimination.py` | Indirect, via `testcases.build_discrimination_cases()` | Yes, inherits the `testcases.py` monkeypatch fix via `rerun_all.py`. |
| `scripts/contamination_impact.py` | Reads `results/discrimination_summary.csv`, itself derived from `results.json` | **By design** — this file's entire purpose is the old-vs-new comparison. Not a bug. |
| `scripts/script_adherence_report.py` | Reads `results.json` directly, classifies it after the fact | **By design** — this file's entire purpose is the original-vs-strict-prompt comparison. Not a bug. |
| `run_experiment.py` | Writes it (not a reader) | N/A |

**Net effect (as of the 0.1 audit, before this fix):** every number sitting in `results_clean/` was clean. Every number in `results/` was contaminated and only meaningful as the "before" side of a diff. `analyze.py` and the raw, unmonkeypatched invocation of `testcases.py`/`encoder_comparison.py`/`discrimination.py`/`minimal_edit.py` were the loose threads — they defaulted to contaminated data and nothing stopped a future `python scripts/discrimination.py` (run directly, not via `rerun_all.py`) from silently reproducing the contaminated numbers again.

**Resolved in Milestone 0.2 — see "Correction 1" at the top of this file.** Every script listed in this table's "Fix already applied elsewhere?" column as monkeypatch-dependent now defaults to clean data on its own, with no monkeypatch anywhere in the repo. `analyze.py` is archived rather than fixed (Correction 3), since it was superseded rather than broken.
