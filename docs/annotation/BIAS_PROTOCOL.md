# Bias protocol for Milestone 6.3's human-agreement study

This document exists because the project owner is BOTH the builder of
`indic_judge`/`check_trace` and one of the two human graders in this
study. That is a conflict of interest by definition -- someone
grading how well their own metric performs has every incentive,
conscious or not, to grade generously. This protocol is the mitigation.
Follow it in order. Do not skip steps because they feel like
overhead -- the whole point of a bias protocol is that it binds you
even when it's inconvenient.

## Step 0 — Before any grading starts

- [x] `data/trace_study_traces.json` has been generated
      (`experiments/scripts/generate_trace_study_data.py`) and is
      treated as PINNED from this point forward -- do not regenerate
      it once grading begins, even if you notice something you'd want
      to fix in the source questions. Fix it for a v2 dataset later;
      this run is frozen. (62 traces, generated and committed at
      `11fd38b`.)
- [x] This rubric (below, "Grading rubric") is frozen as of the commit
      that adds this file (`11fd38b`). If the rubric changes after
      grading starts, **the entire study restarts from Step 1** --
      every grade already assigned is discarded, not adjusted. This is
      expensive on purpose: it removes any incentive to tweak the
      rubric mid-grade to match a result you're hoping for.
- [x] `data/trace_study_grading_sheet.csv` (the blinded sheet) has
      been generated and contains ONLY: trace_id, question, answer,
      judge_reasoning, plus two empty columns for your verdict and
      notes. It does NOT contain: answer_type (correct/wrong ground
      truth), judge_score, judge_passed, judge_label, or trap_word --
      verified by inspection before committing. Confirm this again
      before sending the sheet to the second grader (re-check the
      column headers yourself; don't just trust this note).

## Step 1 — Commit ground truth BEFORE grading

- [ ] `data/trace_study_traces.json` (which DOES contain
      `answer_type`, the ground truth, and the judge's own verdict) is
      committed to git NOW, before either grader opens the blinded
      sheet. This is the commit that makes it impossible to quietly
      adjust the ground-truth labels after seeing how grading goes --
      the git history is the record.
- [x] Note the commit hash here once made: `11fd38b`

## Step 2 — Split into holdout

- [ ] Before grading, split the 62 traces into two sets: ~70% (a
      "tuning" set you can look at and discuss) and ~30% (a holdout,
      roughly 18-19 traces) that NEITHER grader looks at, discusses,
      or uses to adjust their judgment, until both full sets of grades
      are already submitted.
- [x] Record which trace_ids are in the holdout here, before grading
      starts: `T001,T002,T003,T006,T007,T008,T016,T023,T029,T038,T043,T045,T046,T047,T049,T050,T052,T057,T062`
      (19 of 62 = 30.6%, selected via `random.seed(1234)` over the
      sorted trace_id list -- not hand-picked, so the choice of which
      traces are held out cannot itself be influenced by which ones
      look "safe" or "risky.")
- [ ] The holdout exists to catch a specific failure mode: grading the
      first 70% teaches you the judge's patterns, and you
      (consciously or not) start grading the last 30% differently
      because you now expect a certain kind of trace. Comparing
      agreement on the holdout vs. the tuning set tells you whether
      that happened.

## Step 3 — Grade independently, blind

- [ ] You grade all 62 rows in `trace_study_grading_sheet.csv`,
      writing "correct" or "wrong" in `grader_verdict_correct_or_wrong`
      based ONLY on: does this answer actually, factually answer the
      question correctly? (Not: does it match what I expect the judge
      to have said. Not: does the reasoning look sophisticated.)
- [ ] The second grader (a fluent Hindi speaker, NOT told which model
      generated anything, NOT shown your grades, NOT shown
      `judge_score`/`judge_passed`/`judge_label`) grades their own
      copy of the same blinded sheet, independently, with no
      discussion between you until both are submitted.
- [ ] Neither grader sees the other's answers until both files are
      saved and dated.

## Step 4 — Compute agreement

- [ ] Compare: your verdicts vs. the second grader's verdicts
      (inter-annotator agreement -- are two humans consistent with
      each other?).
- [ ] Compare: each grader's verdict vs. `indic_judge`'s own
      `judge_passed` from `trace_study_traces.json` (this is the
      actual number the project needs -- does the judge agree with
      humans?).
- [ ] Compare: agreement on the tuning 70% vs. the holdout 30%
      separately. A large gap between them is itself a finding, not
      noise to average away.
- [ ] Report precision/recall/agreement honestly, whatever the number
      is -- see BUILD_PLAN.md 6.5: "A metric catching 60% of
      mistranslations at a 15% false positive rate, honestly reported,
      beats a vague claim of working well."

## Step 5 — Disclose

- [ ] `experiments/README.md` and `README.md`'s Limitations section
      state plainly: one of the two graders is the project owner and
      also the builder of the metric being evaluated. State the
      second grader's role (independent, fluent Hindi speaker, no
      stake in the result) without necessarily naming them if they'd
      rather not be named.
- [ ] Do not describe the result as "validated" without this
      disclosure sitting next to the number, every time the number is
      cited.

## Grading rubric (frozen once this file is committed)

For each trace, read the question and the answer text (the
`judge_reasoning` column is provided as context but is NOT what you're
grading -- you are grading whether the ANSWER is correct, independent
of what the judge said about it).

Write **"correct"** if the answer factually and directly answers the
question as a fluent Hindi speaker would understand it.

Write **"wrong"** if the answer is factually incorrect, uses the wrong
sense of an ambiguous word, or does not actually answer the question
(even if it's fluent, confident-sounding Hindi).

If genuinely unsure, write "unsure" rather than guessing -- an unsure
verdict is itself useful data (it may mean the question was
ambiguously worded), and forcing a guess corrupts the agreement
number more than an honest "unsure" does.
