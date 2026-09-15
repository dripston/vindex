# experiments/ — what each asked, what it found, whether it survived

Everything here is exploratory: it answered a question, it is not meant
to be imported as a library. `src/vindex/` is where validated logic
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
  default 0.5 cosine-similarity cutoff, most encoder x language
  configurations perform at or near chance accuracy for correct-vs-
  wrong discrimination (`scripts/discrimination.py`,
  `results_clean/discrimination_summary.csv`). The original estimate
  here was "13 of 15" -- Milestone 3.1's exact count, computed
  directly from the english_gold/full_sentence/hard-negative slice
  used to build `vindex.calibration`'s shipped table, is **11 of 15
  at exactly <=0.5 accuracy** (12 of 15 at <=0.55). Both numbers say
  the same thing -- a 0.5 default is close to a coin flip on most
  cells -- the exact count just depends on which slice of the 2x2x3
  (gold_mode x gold_length x variant) grid you read it from. See the
  Milestone 3 table below for the full picture, calibrated and
  uncalibrated side by side.
- **MuRIL cannot discriminate at all.** `google/muril-base-cased` scores
  ~0.99 similarity on nearly everything -- correct answers, wrong
  answers, different languages -- with std ~0.002. It is the encoder an
  Indian-language project would reach for first, and it is the one that
  fails hardest.
- **Script adherence: 20% under the original instruction, 100% under
  the strict one.** Same model, same 30 questions, one system-prompt
  change (`scripts/script_adherence_report.py`,
  `results_clean/script_adherence.csv`).

## Milestone 2 beats the exact-match baseline -- with the number, and its limit

`scripts/beat_the_baseline.py` (Milestone 2.4) re-runs the same 30
(answer, gold) pairs from `data/results_clean.json` that produced the
0/30 finding above, through `vindex.match`'s three modes, using each
task's register-matched full-sentence gold
(`gold_en_full`/`gold_hi_full`/`gold_hinglish_full` in
`testcases.py`) instead of the bare-entity gold the 0/30 baseline used.

The honest result: **`exact_match_score` is ALSO 0/30.** This is not a
vindex bug and not a script problem -- it is a property of the task.
Real model answers add context beyond the fact ("Mumbai is the capital
of Maharashtra, the most populous state in India" against a gold of
"The capital of Maharashtra is Mumbai."), so no answer in this dataset
is ever a pure string match against any gold, however well-normalized.
Exact match cannot beat this baseline, on this data, and no amount of
script/diacritic/numeral normalization changes that -- normalization
fixes *spelling* disagreement, not *phrasing* disagreement.

The real improvement is the two modes exact match cannot express at
all:

| mode                 | score (mean over 30) |
|-----------------------|----------------------:|
| baseline exact match  | 0/30                  |
| vindex exact match    | 0/30                  |
| vindex token F1       | 0.473                 |
| vindex char similarity| 0.472                 |

Both give partial-credit signal -- a paraphrased-but-correct answer
scores well above zero, a wrong-entity answer scores near it -- exactly
where a binary exact-match check goes silent. Run
`python experiments/scripts/beat_the_baseline.py` for the full
per-row breakdown.

## Documented failures (Milestone 2.5)

**Transliteration is many-to-many, and that is not fully solvable.**
"tune" is a genuine ambiguity: it can mean the loanword "tune" (ट्यून)
or the pronoun+postposition "tune" (तूने, "you [did]"), and these are
unrelated words that happen to share a Roman spelling. This is not a
gap in `indic_transliteration`'s coverage -- a Jio engineer confirmed
directly that their own in-house transliteration layer does not fully
resolve this class of ambiguity either. `vindex.transliterate` picks
one deterministic rendering ("तुने") and does not attempt disambiguation
by context; see `src/vindex/transliterate.py`'s module docstring.

**Where vindex beats Sarvam's approach, for a narrow, real reason.**
Sarvam's published work solves the loanword problem -- "वह doctor"
versus "वह डॉक्टर" -- with an LLM call per case. `vindex.loanwords`
(Milestone 2.5) solves the same class of case with a small, hand-picked
lookup table (10 words) consulted before ITRANS's phonetic
transliteration runs, so a known loanword gets its conventional
spelling instead of a letter-by-letter guess ("doctor" would otherwise
become "दोच्तोर्", not "डॉक्टर"). This is deterministic, free (no API
call), and perfectly reproducible -- for exactly the fixed vocabulary
in the table, and no further. It is a genuine advantage over an
LLM-based normalizer for this narrow case, not a general claim that
lookup beats LLM judging -- see `src/vindex/loanwords.py`'s module
docstring for the table's own v0 limitations (10 words, Devanagari
only, no inflections, a loanword outside the table falls straight
through to ITRANS and is not fixed by this module at all).

## Milestone 3: calibrated_similarity beats the 0.5 default -- the table

`vindex.calibration.CALIBRATION_TABLE` ships thresholds derived from
`results_clean/discrimination_summary.csv`, sliced at
`gold_mode=english_gold, gold_length=full_sentence`, scored against
hard-negative discrimination (correct vs. a different-entity wrong
answer). **Each cell is calibrated from 10 cases** -- stated plainly,
per Milestone 3.1, because this is a small-sample calibration, not a
large validation study; see `src/vindex/calibration.py`'s module
docstring for exactly why this slice was chosen over the other
`gold_mode`/`gold_length` combinations in the same CSV, and use
`vindex.calibrate()` (Milestone 3.3) to fit your own threshold on your
own labelled data instead of trusting this table as ground truth.

| encoder                          | language | accuracy @ 0.5 (uncalibrated) | calibrated threshold | accuracy @ threshold |
|-----------------------------------|----------|-------------------------------:|----------------------:|-----------------------:|
| all-MiniLM-L6-v2                  | en       | 0.500                          | 0.863                 | 0.650                  |
| all-MiniLM-L6-v2                  | hi       | 0.526                          | 0.072                 | 0.632                  |
| all-MiniLM-L6-v2                  | hinglish | 0.600                          | 0.351                 | 0.750                  |
| LaBSE                             | en       | 0.500                          | 0.553                 | 0.500                  |
| LaBSE                             | hi       | 0.474                          | 0.867                 | 0.526                  |
| LaBSE                             | hinglish | 0.650                          | 0.423                 | 0.750                  |
| multilingual-e5-base               | en       | 0.500                          | 0.878                 | **0.900**              |
| multilingual-e5-base               | hi       | 0.474                          | 0.880                 | 0.632                  |
| multilingual-e5-base               | hinglish | 0.500                          | 0.845                 | 0.750                  |
| muril-base-cased                  | en       | 0.500                          | 0.996                 | 0.500                  |
| muril-base-cased                  | hi       | 0.474                          | 0.995                 | 0.526                  |
| muril-base-cased                  | hinglish | 0.500                          | 0.991                 | 0.600                  |
| paraphrase-multilingual-mpnet-v2  | en       | 0.500                          | 0.818                 | **0.850**              |
| paraphrase-multilingual-mpnet-v2  | hi       | 0.474                          | 0.871                 | 0.737                  |
| paraphrase-multilingual-mpnet-v2  | hinglish | 0.700                          | 0.412                 | 0.800                  |

Reading it straight: at the naive 0.5 default, 11 of these 15 cells
score at or below 0.5 accuracy -- worse than useless as a threshold,
since a coin flip also gets 0.5. Calibrated, 10 of 15 cells clear 0.6
(all 3 MuRIL cells and both non-Hinglish LaBSE cells are the five that
don't), and the best cell (multilingual-e5-base, English) reaches
0.90. MuRIL
never clears 0.6 even calibrated, for the reason `MURIL_WARNING`
documents: it scores ~0.99 on nearly everything regardless of
correctness, so there is no threshold that separates its correct
answers from its wrong ones -- calibration cannot fix an encoder that
does not encode the distinction in the first place. `paraphrase-
multilingual-mpnet-base-v2` is the most consistently strong performer
across all three languages, which is why `vindex.calibration.
DEFAULT_ENCODER` picks it over the single-best-cell winner
(multilingual-e5-base, which is excellent on English but weaker on
Hindi and Hinglish).

Reproduce this table: `python experiments/scripts/discrimination.py`
(re-scores against the pinned `data/results_clean.json`; needs
sentence-transformers, transformers, and torch --
`pip install vindex[similarity]`).

## The vindex port is validated against this finding

`scripts/validate_vindex_port.py` (Milestone 1.6) re-runs the 20%/100%
Hinglish adherence check above through `src/vindex/`'s ported and shipped
code instead of `script_check.py` directly -- both the ported
`is_script_adherent(answer, variant)` (methodologically identical to
`script_adherence_report.py`) and the new prompt-driven
`vindex.script_adherence(question, answer)` public metric (Milestone
1.4). Both reproduce 20% under the original prompt and 100% under the
strict one, exactly. This is why `experiments/results.json` stays in the
repo rather than being deleted once `data/results_clean.json` existed:
it is the fixture this regression check runs against.

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
