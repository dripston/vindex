# Changelog

## v0.2.0

**indic_judge**: an LLM judge built for Indic scripts, not an English rubric pointed at Hindi (Milestone 5.1-5.7).

**judge_trace_check / check_trace**: dictionary-first mechanical detection of mistranslated ambiguous Hindi terms inside a judge's own reasoning trace, plus an optional LLM fallback mode (Milestone 6.1, 6.2, 6.4).

**Validation**: 62-trace human-agreement study under a documented bias protocol (`docs/annotation/BIAS_PROTOCOL.md`) — ground truth committed before grading, blind sheet, frozen rubric, 30% holdout. Result: 90.3% agreement between two independent human graders and `indic_judge`'s own verdict, 100% inter-annotator agreement, no tuning-vs-holdout drift (Milestone 6.3/6.5).

**Normalization and matching** (carried in from pre-v0.2 work, first released here): transliteration-aware normalization pipeline, three match modes (exact, token F1, char similarity), calibrated per-encoder similarity thresholds (Milestone 2, 3).

**Integrations**: DeepEval, promptfoo, and Ragas (Milestone 4.1/4.2).

Known limitation, stated plainly: `check_trace`'s dictionary currently covers 70 trap-word entries. It catches mistranslation of *known* ambiguous terms — not mistranslation in general. Grows from user reports.

## v0.1.0

Initial release: `script_adherence` check, 30 tests, 90 responses validated. 20% vs 100% adherence result as the launch headline.
