# experiments/ — what each asked, what it found, whether it survived

Everything here is exploratory: it answered a question, it is not meant
to be imported as a library. `src/pramana/` is where validated logic
gets ported once a milestone says to port it. Nothing in this directory
is deleted, even when superseded -- superseded results stay on the
record so nobody re-derives a wrong number from scratch.

## The two things that must be understood before reading any number below

**The multi-encoder rerun SUPERSEDED the original "script gap" claim.**
The first experiment (`run_experiment.py` / `FINDINGS.md`, scored with
`all-MiniLM-L6-v2`, an English-only encoder) found a large EN-vs-Hindi
similarity gap and read it as evidence that the *script* itself was the
problem. `scripts/encoder_comparison.py`, rerun across five multilingual
encoders, showed this was mostly an artifact of picking an English-only
encoder: the EN-minus-Hindi gap fell from 0.248 (MiniLM) to 0.015
(LaBSE) and 0.015 (e5) on the register-matched full-sentence gold. The
gap is a property of encoder choice, not of the script being Hindi.

**The contamination fix INVALIDATED every pre-fix Hinglish figure.**
`run_experiment.py`'s system prompt ("reply in the same language and
script") was ambiguous enough that the model answered Romanized-Hindi
("Hinglish") prompts in Devanagari script 8 times out of 10. Every
`hinglish`-variant number computed from `results.json` before the fix is
therefore not measuring what it claims to measure -- in several tasks
the "hinglish" answer text was 74-96% character-identical to the "hi"
answer text, because the model silently ignored the requested script.
`scripts/regenerate_dataset.py` fixed this with an escalating,
script-forbidding prompt, producing `data/results_clean.json`. Any
number computed from `results.json` alone, without the word
"contamination" or "before/after" attached, should not be trusted.

## The two other things worth stating plainly

**`data/results_clean.json` is PINNED and must never be regenerated.**
The model is not deterministic even at temperature 0 -- confirmed
empirically, roughly a third of cases reword on a re-run. Regenerating
it would silently invalidate every hand-authored negative and every
score computed against it. See `data/NOTICE.md`.

**The 150x4x3 adherence sweep is incomplete, not a result.**
`scripts/script_adherence.py` is a *second*, independent, larger
script-adherence experiment (150 hand-authored prompts x 4 instruction
conditions x 3 Groq models = 1800 calls), separate from the 30-case
before/after in `scripts/script_adherence_report.py`. It was started and
aborted mid-run (552 of 1800 rows) when `qwen/qwen3.6-27b` proved
unworkable -- it emits visible `<think>...</think>` reasoning that
regularly exceeds the account's 1000 output-tokens/minute cap for that
model. The code was swapped to `qwen/qwen3.8-27b` mid-flight, but the
full run was never redone under the new model choice, and the partial
checkpoint (`results_adherence/checkpoint.jsonl`, 552 rows, mixed
between the two model choices) has been deleted rather than kept as a
half-result -- half a run reads as a finished one to anyone who finds it
later. This sweep is on the roadmap, not in the evidence base.

**This does not affect the headline 20%/100% finding.** That number
comes from `scripts/script_adherence_report.py`, a different, already-
complete script: it compares the original `experiments/results.json`
(10 questions x 3 variants, one model, unscored for adherence at
generation time) against `data/results_clean.json` (the same 30
questions, regenerated under the strict prompt). See
`results_clean/script_adherence.csv`.

## Findings that DID survive

- **Exact match scores 0/30 across all scripts.** No script-normalized
  comparison exists yet (Milestone 2); naive string equality fails
  completely across Devanagari/Latin/mixed answers.
- **Similarity thresholds do not transfer across encoders.** At a
  default 0.5 cosine-similarity cutoff, 13 of 15 encoder x language
  configurations perform at chance accuracy for correct-vs-wrong
  discrimination (`scripts/discrimination.py`,
  `results_clean/discrimination_summary.csv`).
- **MuRIL cannot discriminate at all.** `google/muril-base-cased` scores
  ~0.99 similarity on nearly everything -- correct answers, wrong
  answers, different languages -- with std ~0.002. It is the encoder an
  Indian-language project would reach for first, and it is the one that
  fails hardest.
- **Script adherence: 20% under the original instruction, 100% under
  the strict one.** Same model, same 30 questions, one system-prompt
  change (`scripts/script_adherence_report.py`,
  `results_clean/script_adherence.csv`).

## File-by-file notes not already covered above

- **`analyze.py`** -- ARCHIVED, do not run. Deeper per-case read of
  `results.json` for the week-one `FINDINGS.md` writeup. Not broken:
  superseded by `scripts/contamination_impact.py` (the proper
  before/after comparison) and `scripts/discrimination.py` (proper
  discrimination scoring across five encoders instead of one). Kept for
  the record because it answered a real question in week one.
- **`hinglish_probe.py`** / **`hinglish_probe_results.json`** -- a small
  (n=20) controlled probe: does the LLM judge penalize pure
  script/register once correctness is held constant? Answer: no. Still
  valid as a small probe; superseded in *scope* (not validity) by the
  much larger `scripts/discrimination.py` / `scripts/minimal_edit.py`
  work (n=90-145).
- **`script_adherence_cases.py`** -- 150 hand-authored prompts, the
  fixture for `scripts/script_adherence.py`'s (incomplete) sweep above.
  Independent of the contamination-experiment files by design.
- **`testcases.py`** -- the 10 base tasks, gold answers, and
  hand-authored wrong-answer negatives shared by most scripts in this
  directory. `build_discrimination_cases()` now takes an explicit
  `data_path` (see Correction 1 below); it is not itself a result.
- **`results/`** -- output of `encoder_comparison.py` /
  `discrimination.py` run against the *contaminated* dataset. Never
  cite these numbers standalone; they exist only as the "before" half
  of `results_clean/contamination_impact.csv`.
- **`results_clean/`** -- output of the same scripts run against
  `data/results_clean.json`. These numbers are the ones to cite.
- **`results_adherence/`** -- output directory for the incomplete
  150x4x3 sweep above. Currently empty (checkpoint deleted).

## Correction 1 (this reorganisation): the monkeypatch is gone

`scripts/rerun_all.py` used to run the clean-data pipeline by
monkeypatching `json.load` at runtime so that `encoder_comparison.py`,
`discrimination.py`, and `minimal_edit.py` (via `testcases.py`) would
read `data/results_clean.json` even though their own code still
defaulted to `results.json`. That meant running any of those four
scripts directly, without the orchestrator, silently regenerated
contaminated output -- exactly the class of bug that cost a week of
work the first time.

Fixed: `testcases.py`, `scripts/encoder_comparison.py`,
`scripts/discrimination.py`, and `scripts/minimal_edit.py` now each
default to `data/results_clean.json` via an explicit module constant or
function parameter, with a `--data PATH` CLI override on the three
runnable-standalone scripts. `scripts/rerun_all.py` now passes the path
explicitly instead of patching anything. `scripts/contamination_impact.py`
and `scripts/script_adherence_report.py` are the two scripts whose job
*is* the before/after diff, so they read `results.json` on purpose --
each names the path explicitly in its own code, with a comment saying
why.
