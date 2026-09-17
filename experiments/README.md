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

## Milestone 5: indic_judge does not repeat the समुद्र तल error -- confirmed, not assumed

Phase 0 (`experiments/FINDINGS.md`) found `run_experiment.py`'s English
rubric scoring a correct Hindi answer 0.0 -- asked to evaluate "समुद्र
तल पर पानी 100 °C पर उबलता है" (water boils at 100 C at sea level,
correct), the judge's own English-language reasoning silently
mistranslated समुद्र तल ("sea level") as "sea floor" mid-thought, then
marked the answer wrong for not accounting for undersea pressure.

`src/vindex/judge_rubric.py`'s Hindi rubric instructs the judge to
reason in Hindi when the content is in Hindi, specifically so there is
no translation step for that error to hide inside -- and includes the
समुद्र तल case itself as a worked few-shot example the judge must not
repeat. This was tested against the exact same case, with a real Groq
call (not a mock): the Hindi-rubric judge scores it 1.0, high
confidence, and its own reasoning explicitly identifies समुद्र तल as
"समुद्र सतह/समुद्र स्तर" (sea surface/sea level) -- see
`tests/test_judge.py::test_indic_judge_does_not_mistranslate_samudra_tal`,
which runs this exact regression check on every test run that has a
`GROQ_API_KEY` available (skipped otherwise, including in CI, which has
no key configured).

Also tested for real, same case-by-case method: a genuinely wrong
answer (भारत की राजधानी मुंबई है -- "Mumbai" for a question about
India's capital) is still correctly flagged, confirming the Hindi
rubric isn't simply biased toward passing everything; a Romanized-Hindi
("Hinglish") question/answer pair passes without script penalty,
confirming Milestone 5.2's script-aware prompting; and the align-then-
judge exact-match shortcut (Milestone 5.4) measurably skips the LLM
call entirely (0.0s) when `gold` matches `answer` exactly at the word
level.

**Milestone 5.8 (validate against human labels, vs an English-rubric
baseline) has been run, using the Milestone 6.3 data.** An
English-rubric baseline judge (`vindex.judge_rubric_en_baseline`,
structurally identical to `judge_rubric.py`'s Hindi rubric -- same
instructions, same four few-shot examples including the समुद्र तल
case translated, same JSON contract and conservative scoring
threshold, only the rubric language differs) was run against the same
62 pinned traces from the 6.3 study
(`experiments/scripts/run_english_rubric_baseline.py`), then compared
against the same two human graders' verdicts
(`experiments/scripts/compute_english_baseline_agreement.py`).

```
--- ALL (62 traces) ---
  grader1 vs indic_judge (Hindi rubric):   56/62 = 90.3%
  grader1 vs english_baseline (En rubric): 58/62 = 93.5%
  grader2 vs indic_judge (Hindi rubric):   56/62 = 90.3%
  grader2 vs english_baseline (En rubric): 58/62 = 93.5%
```

**Honest result: the English-rubric baseline agreed with humans
slightly MORE than `indic_judge`'s Hindi rubric on this set** --
93.5% (58/62) vs 90.3% (56/62). This is the opposite of the motivating
hypothesis from Phase 0 (that a Hindi rubric avoids the समुद्र तल-style
mistranslation an English-reasoning judge is prone to), and it is
reported as-is, not adjusted or re-run to get a different answer.

The 4 traces where the two judges disagree (T023 डाक, T025 गुरु, T033
चारा, T051 निकासी, all `answer_type=correct`) are not the English
judge catching a Hindi misread, or the reverse -- inspecting the raw
reasoning shows `indic_judge` being **more conservative about
completeness**, not less accurate about comprehension: in 3 of the 4
cases (T023, T025, T051) the Hindi rubric's own chain-of-thought
correctly identifies the answer as factually right, then flags it
anyway for omitting a secondary detail the question didn't explicitly
ask for (e.g. T023: the answer correctly says a sealed envelope and
registration fee must be submitted for registered mail, and the Hindi
judge's reasoning agrees this is correct, but flags it for not also
mentioning ordinary postage). The English-rubric run and both human
graders accepted the same answers as correct. Only T033 goes the other
direction (Hindi rubric passes, English rubric flags the same kind of
completeness gap).

**What this does and doesn't show.** It doesn't show the Hindi rubric
is worse at comprehension -- zero of the 4 disagreements involve a
mistranslation or a misread; all 4 are the conservative-completeness
behavior that Milestone 5.5 deliberately built in, just triggering
slightly more often under the Hindi rubric's phrasing than the English
one's. It does show that "a Hindi rubric agrees with humans more than
an English one" was, until this run, an assumption motivated by one
qualitative Phase 0 incident, not a demonstrated result -- and on this
62-trace sample, the measured direction is the opposite. Small sample,
4 disagreements total -- not strong evidence either way about which
rubric is better in general, but real evidence against treating the
Hindi-rubric choice as already validated by agreement numbers.
Disclosure per `docs/annotation/BIAS_PROTOCOL.md` applies identically
here: the human grades being compared against are the same ones from
the 6.3 study, graded by the project owner and one independent grader.

## Milestone 6: judge_trace_check catches the error class it was built for, with a real 70-entry dictionary

`src/vindex/judge_trace_check.py`'s `check_trace()` is the mechanism
Minakshi named as the worrying one: the समुद्र तल error happens inside
a judge's reasoning before any output exists, so nothing downstream
(script_adherence, a transliteration layer, WER) can catch it. Tested
against the real dictionary (not an injected fixture): the exact
समुद्र तल case is correctly flagged when a trace substitutes "sea
floor" for "sea level"; a trace that reads it correctly passes; a
trace that mentions both readings while correctly reasoning through
the ambiguity is correctly NOT flagged (a judge working through an
ambiguity is not the same as a judge misreading one). The उत्तर
("north" vs "answer") case from BUILD_PLAN.md 6.1 was also tested and
correctly caught. See `tests/test_judge_trace_check.py` and
`tests/test_trap_words.py`.

**The shipped dictionary (`data/trap_words/`) has 70 usable entries.**
60 HindiWiC words plus 15 hand-authored terms (misleading compounds,
tense-flip time adverbs, fractional numbers, Indian large-number
words), with `suggested_reading_a`/`suggested_reading_b` filled in by
hand -- per BUILD_PLAN.md 6.1's instruction that this is the project
owner's own work, not something to automate ("about an hour, and it's
your work, not your agent's"). In practice: drafted with help from a
Hindi-fluent LLM (Sarvam's chatbot, fed the real inventory CSV) for
speed, then reviewed before committing -- disclosed here, not silently
presented as unassisted manual work. 5 of the 60 HindiWiC words (तेल,
धन, डब्बा, संबंध, थान) were marked "no good trap" and left blank on
purpose; their secondary senses aren't realistically confusable enough
to be worth a forced pair.

**One real bug found and fixed during this fill-in.** The first draft
used "ocean floor (seabed)" for समुद्र तल's reading_b -- a real
synonym of the actual Phase 0 wording ("sea floor"), but
`check_trace`'s substring-match logic required the literal reading
text, so "ocean floor" in a trace matched while the historically-exact
"sea floor" did not, and vice versa. Root-caused to
`_trace_says_wrong_reading`'s design: dictionary readings can carry a
parenthetical clarifying gloss (e.g. "ocean floor (seabed)") meant for
a human filling in the CSV, but a real trace only uses the primary
term. Fixed with `_primary_gloss()`, which strips the parenthetical
before matching -- and separately, समुद्र तल's reading_b was corrected
to "sea floor" to match the actual documented case exactly, since
that's the wording with real evidence behind it. See
`test_check_trace_matches_reading_with_parenthetical_gloss_stripped`.

**Milestone 6.4 (optional LLM fallback)** was tested for real, using a
negation-scope drop (a trace that drops a "not" entirely) -- once कल
("kal", tense-ambiguous "yesterday"/"tomorrow") became a real
dictionary entry, it stopped being a valid fallback test case (it's
now a dictionary hit, which is strictly better: free and instant
instead of an LLM call). The negation case is genuinely outside any
fixed word-pair dictionary's reach, and `check_trace_llm_fallback()`
correctly caught it via a real Groq call, using align-then-judge
(Sarvam's shape) to send only the mismatched segment, not the full
text. The exact-alignment shortcut (no LLM call when trace and source
already match) was also confirmed: 0.001s, no call made.

**Milestone 6.3 (validate with ~60 traces) has been run.** 31
trap-seeded questions (correct + wrong answer each = 62 traces) were
generated (`experiments/scripts/generate_trace_study_data.py`), judged
reference-free with `indic_judge`, and the ground truth was committed
to git (`11fd38b`) before either grader opened the blinded sheet. Two
people -- the project owner and one other fluent Hindi speaker --
graded all 62 independently, on a blinded sheet (no ground truth, no
judge verdict, no trap word visible), against a rubric frozen before
grading started, with a 19-trace (30.6%) holdout selected by a
documented random seed and never looked at while tuning. Full protocol
and checklist: `docs/annotation/BIAS_PROTOCOL.md`.

**Milestone 6.5 (publish the number): 90.3% agreement (56/62) between
the two human graders and `indic_judge`'s own verdict.** Inter-annotator
agreement (the two humans against each other) was 100%. Split by set:
88.4% (38/43) on the tuning set, 94.7% (18/19) on the untouched
holdout -- no meaningful gap, no sign the graders drifted as they
learned the judge's patterns. Every disagreement was on a trace the
judge itself flags conservatively (a correct answer marked "flagged"
over an edge-case nuance unrelated to the core mistranslation the
metric targets, e.g. कमान, डाक, गुरु, हार, निकासी, लाख from the earlier
findings above) -- not a missed mistranslation, which is the failure
mode this metric exists to avoid.

**Required disclosure (BIAS_PROTOCOL.md Step 5):** one of the two
graders is the project owner, who also built `indic_judge` and
`check_trace` -- the metric being evaluated. That is a real conflict
of interest. The mitigations are the bias protocol itself (ground
truth committed before grading, blinded sheet, frozen rubric,
documented-random holdout) plus the second grader: an independent
fluent Hindi speaker with no stake in the project's result, grading
from the same blinded sheet with no discussion until both submitted.
The 90.3% figure should not be cited without this paragraph, or an
equivalent, next to it.

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
