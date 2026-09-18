# Changelog

## v0.3.0

**Fixes and one dictionary correction found by a fourth independent
outside review** (installed 0.2.3, read all ~3000 lines of source and
the full README, ran five adversarial batteries against all four
public functions):

- **`check_trace`'s उत्तर entry was backwards for this library's own
  primary use case.** reading_a="north" (treated as correct),
  reading_b="answer" (treated as a mistranslation to flag) -- but this
  library evaluates question-answering, where उत्तर meaning "answer"
  is correct far more often than "north." A judge trace correctly
  saying "the answer is X" for a Hindi question containing उत्तर was
  flagged as a misread 100% of the time. Removed from the dictionary
  (69 entries now, not 70) rather than shipped with either polarity,
  since no single reading_a/reading_b assignment is right for both
  "उत्तर की ओर" (north) and "सही उत्तर" (the correct answer).
- **`check_trace`'s source-side term matching had no word boundary at
  all.** `word.term not in source` was a raw substring test -- कल
  matched inside कलम ("pen"), दर matched inside चादर ("bedsheet"), मूल
  matched inside मूल्य ("price"). The gloss side already had a
  word-boundary fix (v0.2.2); the source side never did, and a plain
  regex `\b` boundary does not work correctly for Devanagari (matras
  and the virama are `\w` characters with no boundary before them, and
  consecutive consonant letters with no vowel sign between them are
  still one word). Fixed with a character-position-based boundary
  check instead of regex.
- **Two documentation bugs in `check_trace`'s own docstring and the
  README, now corrected:** a stale "N=0, dictionary ships blank"
  sentence left in after the dictionary was actually filled in
  (contradicting the "25 of 70 entries" paragraph two sections later
  in the same file); and a README bullet headed
  "`check_trace`/`indic_judge` human-agreement study" whose 90.3%
  result is actually `indic_judge`-only -- `docs/annotation/BIAS_PROTOCOL.md`'s
  own comparison is explicitly "each grader's verdict vs.
  `indic_judge`'s own [verdict]"; `check_trace` was never part of that
  study despite using the same 62 traces as source material. No
  precision/recall study of `check_trace` itself has been run.
  **Also newly documented (not previously disclosed): every dictionary
  gloss is ASCII English, and `judge_rubric.py` instructs the judge to
  reason in Hindi -- so a Hindi-language judge trace can never trigger
  `check_trace` at all,** regardless of whether it misread anything.
  `check_trace`/`indic_judge` do not currently compose safely on
  `indic_judge`'s own recommended (Hindi-rubric) output for this
  reason.
- **`calibrated_similarity` silently returned `score=1.0, passed=True`
  for a NaN cosine similarity.** `max(0.0, min(1.0, float("nan")))`
  returns `1.0` in Python (every comparison against NaN is False), so
  a corrupted embedding (e.g. a bad cached `.npy` file, fp16 overflow)
  produced the single most confident possible result from a
  numerically undefined comparison. Fixed: raises `ValueError`
  instead. `calibrate()` had the matching hole -- a NaN silently
  behaved as "always wrong" with no warning, and the out-of-[-1,1]
  range check couldn't catch it (`nan < -1.0` and `nan > 1.0` are both
  False) -- now raises immediately on any NaN input.
- **`indic_judge`'s parser silently accepted a boolean `score`.** `bool`
  is a subclass of `int` in Python, so `float(True) == 1.0` succeeded;
  `{"score": true}` was accepted as a valid score of 1. Now rejected
  explicitly as malformed, consistent with every other invalid shape.
- **`_extract_json` used a greedy regex** (`\{.*\}` with DOTALL), so a
  judge response mentioning any brace in prose before its real JSON
  answer had the prose glued into the "JSON" and failed to parse.
  Rewritten to find every balanced `{...}` span (brace-counted,
  string-literal-aware) and return the first one that is actually
  valid JSON.
- **A judge sending `confidence: "medium"` had its raw value silently
  overwritten by the normalized `"low"` everywhere, including in
  `detail`** -- so the audit trail claimed the judge said "low" when
  it said "medium" (the conservative low-confidence *behavior* was
  correct; only the reporting was wrong). `_parse_judge_response` now
  returns both the normalized confidence (used for gating) and the
  raw value (for `detail["raw_confidence"]` when they differ).
- **A non-string, non-`None` text argument crashed every metric.**
  `script_adherence(float("nan"), "x")` raised `AttributeError:
  'float' object has no attribute 'strip'` -- pandas puts `nan`, not
  `None` or `""`, in an empty dataframe cell, and `nan` is truthy in
  Python so the existing `value or ""` coercion didn't catch it. Added
  `vindex.result.coerce_text()`, used by `script_adherence`,
  `check_trace`, `calibrated_similarity`, and `indic_judge` -- a
  non-string, non-`None` value now degrades to `label="empty"` like a
  genuinely empty string, instead of crashing.
- **`_family`'s same-model self-enhancement check misses the most
  common real pairing.** "gpt-4o" vs "gpt-4o-mini" -- probably the
  single most common self-judging pair in production -- is not caught
  ("mini" is not a bare size digit, so neither name gets stripped).
  Documented as a known limitation rather than patched with a fragile
  pattern expansion that would still miss the next naming convention.
- Smaller documentation fixes: `language.py` said "~16 function
  words," there are 14 (matches what the README already said);
  `_encoder_instances` (similarity.py) and the embedding cache key
  (encoder.py) both now document real, disclosed limitations
  (unbounded process-lifetime growth; no model-revision pinning);
  `judge_rubric.py` now discloses the prompt-injection surface from
  formatting untrusted answer text directly into the grading prompt
  with no delimiters.
- Checked and found NOT reproducing: the same review's claim that the
  README's own `language_mismatch` examples ("Se7en", "ka in Egyptian
  belief") don't actually trigger the failure they're describing. Both
  examples, and three additional ones the review proposed, all
  reproduce exactly as documented -- verified directly, no change made.

## v0.2.3

**Fixes found by a third independent outside review** (installed 0.2.2
fresh from PyPI, diffed against 0.2.1, drove all four public functions
with adversarial input again):

- **Indic-script digit-only responses defeated `no_script_signal`.**
  The v0.2.1 fix made emoji/CJK/ASCII-digit/punctuation-only responses
  correctly fail as `no_script_signal` -- but every Indic script's own
  decimal digits live inside that script's Unicode block (e.g.
  Devanagari ०-९ are in the same U+0900-U+097F block as the letters),
  so a response that was purely Devanagari digits (`"१४०००००००००"`)
  was NOT caught: `count_scripts()` counted the digits as real script
  characters, and `classify()` confidently returned `"devanagari"`,
  the same verdict as a real sentence. Same gap for Bengali, Tamil,
  Kannada, and every other Indic script's own digits, and for a digit
  paired with a lone danda (`"१।"`, which slipped past the
  danda-only check too, since the digit is a non-danda character in
  the block). Fixed: a script's characters now only count as real
  script signal if at least one of them is an actual letter -- not a
  decimal digit (Unicode category Nd) and not a danda/double danda.
  A real sentence containing Indic digits is unaffected.
- **`calibrated_similarity`'s shipped `CALIBRATION_TABLE` never
  surfaced a `.warnings`, even for cells that would trip calibrate()'s
  own guard.** `CalibratedThreshold.warnings` (added in v0.2.2) only
  ever populated for a caller's own `calibrate()` call -- the 15
  hand-authored shipped cells were written before `.warnings` existed
  and never carried any, including `LaBSE`/en and `LaBSE`/hi and
  `MuRIL`/en and `MuRIL`/hi, whose own fitted accuracy is at or below
  chance by the exact same guard. Fixed in two parts: (1) the whole
  table is now regenerated by a new script
  (`experiments/scripts/generate_calibration_table.py`) that runs the
  real `calibrate()` on each cell's real per-case similarity scores
  from `discrimination_per_case.csv`, instead of being hand-typed --
  this also corrected `n_cases` for the "hi" cells (9, not the
  previously-assumed 10) and produced real `.warnings` for the 4
  degenerate cells; (2) `calibrated_similarity()` now copies a used
  cell's `.warnings` into both `detail["calibration_warnings"]` and
  `reason`, so a caller sees it without separately reading the
  README's AUC table. Documented plainly: this only catches the most
  degenerate cells (same-sample fitted accuracy at or below chance) --
  it does not catch every cell with weak real AUC (e.g.
  multilingual-e5-base/hi fits above chance in-sample but has real AUC
  0.500), which remains the README's AUC table's job, not this
  warning's.

Everything reported fixed in v0.2.1 and v0.2.2 (AUC table accuracy,
judge `OverflowError`, `calibrate()` degenerate-input warnings,
`check_trace` word-boundary matching) was re-verified this round and
confirmed still correct -- see the review's own findings for the full
verification detail, not reproduced here.

## v0.2.2

**Fixes found by a second independent outside review** (this one installed
`vindex==0.2.1` fresh from PyPI, diffed it against 0.2.0, and drove
`indic_judge`/`check_trace`/`script_adherence`/`calibrated_similarity` with
adversarial input -- no context that this was our own project):

- **The "AUC (threshold-free)" column added in v0.2.1 was, in all 15 rows,
  actually a copy of the calibrated-threshold column** -- a hand-transcription
  error, not a `similarity.py`/`calibration.py` bug (the code was correct in
  both 0.2.0 and 0.2.1; only the README table was wrong). The analysis built
  on the wrong column was wrong too: it named `multilingual-e5-base`/hi
  (real AUC 0.500, exactly chance) as one of only two strong cells, and
  described `LaBSE`/en (real AUC 0.070, strongly inverted) as "barely above
  chance." Regenerated from `experiments/results_clean/discrimination_summary.csv`
  by a new script (`experiments/scripts/generate_auc_table.py`) instead of
  hand-transcribed -- real numbers: 9 of 15 cells above chance, only 3
  strong (>=0.75), mean AUC 0.525. Re-run that script before ever editing
  this table by hand again.
- **`check_trace()` false-positives on ordinary English words.** 25 of the
  70 trap-word dictionary entries have a `reading_b` gloss that is itself a
  common English word with no connection to the Hindi term on its own (e.g.
  उत्तर's wrong-reading gloss is "answer"; अंग's is "organ"). Since judge
  reasoning traces are themselves English prose about correctness, a
  trace's ordinary, unrelated use of "answer" was indistinguishable from a
  genuine misread of उत्तर. Fixed the narrower bug in the same code path
  (substring match let "answer" match inside "unanswerable") by switching
  to word-boundary matching -- this does NOT fix the broader false-positive
  rate on isolated legitimate word use, which has no cheap fix without a
  transliteration/context anchor `TrapWord` doesn't have; documented
  plainly in `_trace_says_wrong_reading`'s docstring instead of silently
  left as a surprise.
- **`indic_judge` crashed with an uncaught `OverflowError`** on a judge
  response containing `{"score": Infinity, ...}`. `json.loads` accepts the
  non-standard `Infinity`/`-Infinity` tokens by default, and `float()`
  accepts them too, but `round()`/`int()` on an infinite float raises
  `OverflowError` -- not one of the exception types `_parse_judge_response`
  or its caller already caught, so a malformed judge response crashed the
  whole call instead of degrading to `judge_error` like every other
  malformed shape. Fixed: caught explicitly, re-raised as `ValueError`.
- **`calibrate()` accepted degenerate input silently.** `calibrate([0.5],
  [0.5])` returned a threshold as if it were a real fit off one point per
  side; `calibrate([50.0], [-50.0])` reported "accuracy 1.0" off values
  outside cosine similarity's valid range with no indication anything was
  wrong; `calibrate()` on data where "correct" scores are lower than
  "wrong" scores returned a threshold with no signal the ranking is
  inverted. Fixed: `calibrate()` still always returns a result (never
  refuses to fit small real-world data), but `CalibratedThreshold` gained
  a `.warnings` tuple that flags too-few-cases (<5 per side),
  out-of-[-1,1]-range scores, and at-or-below-chance fitted accuracy.
  Always empty on the shipped `CALIBRATION_TABLE`.
- **`language_mismatch`'s hard-fail-on-one-match problem was investigated
  and NOT fixed** -- tried downgrading to a soft label when the prompt has
  exactly one Hindi-function-word match (the "Se7en"/"ka" false-positive
  case), but this package's own canonical example, `"Mumbai kahan hai?"`
  (which SHOULD hard-fail an all-English response), also has exactly one
  match. Match count cannot distinguish the two cases -- the same wall a
  2+-match threshold hit earlier and was reverted for. No code change;
  documented as a real, unresolved trade-off in `metric.py` and
  README.md's Limitations section, same as before.

## v0.2.1

**Packaging fix, not a feature release.** The `v0.2.0` package published to
PyPI on 2026-09-15 was built from the wrong commit -- the version string had
been bumped to `0.2.0` but the actual tree checked out at publish time was
`03a7ec6` (the v0.1.0 commit). The published wheel exported only
`MetricResult`, `calibrated_similarity`, and `script_adherence`; `indic_judge`,
`check_trace`, and `check_trace_llm_fallback` were entirely absent, despite
being documented in that same version's README and CHANGELOG entry below.
Anyone who ran `pip install vindex` got a package that silently didn't match
its own docs. `0.2.0` cannot be fixed in place (PyPI never allows
overwriting a version's files) -- `0.2.1` is a from-scratch rebuild off the
current `master`, containing everything the `v0.2.0` entry below describes,
for real this time.

Also corrected in this release: the `v0.2.0` entry below cited the 90.3%
human-agreement number with no disclosure attached. `docs/annotation/BIAS_PROTOCOL.md`
requires that disclosure sit next to the number every time it's cited --
added below, and will not be omitted again.

**Real bugs found by an independent outside review and fixed in this release**
(this package's `check_trace` and `calibrated_similarity` were reviewed by an
AI given no context that this was our own project -- full findings not
reproduced here, fixes are):

- `check_trace()` silently reported `passed=True, score=1.0, dictionary_size=0`
  on every call when installed via pip. The trap-word CSVs were loaded from a
  repo-relative filesystem path that only exists in a git checkout -- an
  installed copy had no dictionary at all, and nothing said so loudly. Fixed:
  the CSVs now ship as real package data, loaded via `importlib.resources`.
- `CACHE_ROOT` (the on-disk embedding cache used by `calibrated_similarity`)
  resolved to a path inside the Python install directory when installed via
  pip, and the cache-write call had no error handling -- a `PermissionError`
  on first real use for most installs. Fixed: uses `$VINDEX_CACHE_DIR` or a
  real platform user-cache directory, and a cache-write failure now degrades
  to no caching instead of crashing.
- `script_adherence` treated ANY response `classify()` calls `"mixed"` as a
  pass for romanized/code-mixed prompts -- including emoji-only, CJK-only,
  digit-only, or punctuation-only responses with zero real script content.
  Fixed with a distinct `no_script_signal` label (always `passed=False`).
  Same fix applied to a response that is purely a lone Devanagari danda
  (`।`), which previously classified as a confident `"devanagari"` match.
- `calibrated_similarity`'s README table showed only the fitted-threshold
  accuracy, which can only look better than the naive 0.5 baseline by
  construction (same 10-19 points, no holdout). Added the threshold-
  independent ROC AUC for all 15 cells: only 6 have genuine above-chance
  discrimination, and only 2 are strong. The other 9 -- several with a
  "good-looking" calibrated accuracy -- have AUC at or below chance, an
  inverted ranking, or come from an encoder (MuRIL) that cannot discriminate
  at all. See README.md's "argument for calibrated_similarity" section.
- `vindex.calibrate()` didn't exist -- the README told users to call it, but
  the function was only reachable as `vindex.calibration.calibrate`. Now
  exported from the top-level package as documented.
- `indic_judge`'s out-of-range score handling, `check_trace`'s paraphrase
  blind spots, and a same-model self-enhancement check bypass were also
  found and are documented (the paraphrase and bypass issues are inherent
  heuristic limitations, corrected in the docs rather than "fixed") --
  see `src/vindex/judge.py`, `judge_trace_check.py`, and README.md's
  Limitations section for the full detail on each.

Two things this same review checked and found NOT overclaimed: the
`is_only_danda_punctuation` fix doesn't touch real-sentence classification,
and a code-mixed prompt answered in pure Devanagari correctly fails --
that's this package's own headline finding working as designed, not a bug
(see `metric.py`'s module docstring).

## v0.2.0

**indic_judge**: an LLM judge built for Indic scripts, not an English rubric pointed at Hindi (Milestone 5.1-5.7).

**judge_trace_check / check_trace**: dictionary-first mechanical detection of mistranslated ambiguous Hindi terms inside a judge's own reasoning trace, plus an optional LLM fallback mode (Milestone 6.1, 6.2, 6.4).

**Validation**: 62-trace human-agreement study under a documented bias protocol (`docs/annotation/BIAS_PROTOCOL.md`) — ground truth committed before grading, blind sheet, frozen rubric, 30% holdout. Result: 90.3% agreement between two independent human graders and `indic_judge`'s own verdict, 100% inter-annotator agreement, no tuning-vs-holdout drift (Milestone 6.3/6.5). **Disclosure required by the protocol, every time this number is cited**: one of the two graders is the project owner, who also built the metric being evaluated; the second grader is an independent fluent Hindi speaker with no stake in the result. A follow-up study (Milestone 5.8) found an English-rubric baseline agreed with the same human graders slightly *more* (93.5%) on this same set — see `experiments/README.md` for the full breakdown; the Hindi rubric's human-agreement advantage is not yet a demonstrated result.

**Normalization and matching** (carried in from pre-v0.2 work, first released here): transliteration-aware normalization pipeline, three match modes (exact, token F1, char similarity), calibrated per-encoder similarity thresholds (Milestone 2, 3).

**Integrations**: DeepEval, promptfoo, and Ragas (Milestone 4.1/4.2).

Known limitation, stated plainly: `check_trace`'s dictionary currently covers 70 trap-word entries. It catches mistranslation of *known* ambiguous terms — not mistranslation in general. Grows from user reports.

## v0.1.0

Initial release: `script_adherence` check, 30 tests, 90 responses validated. 20% vs 100% adherence result as the launch headline.
