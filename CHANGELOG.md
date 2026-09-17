# Changelog

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
