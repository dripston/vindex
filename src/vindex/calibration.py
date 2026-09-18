"""
Calibration table for calibrated_similarity (Milestone 3.1-3.3).

A default 0.5 cosine-similarity threshold performs at or near chance on
most encoder/language combinations -- see experiments/README.md and
experiments/results_clean/discrimination_summary.csv. This module ships
a starting-point calibration table derived from that experiment, plus
calibrate() for fitting your own thresholds on your own labelled data.

WHERE THE TABLE COMES FROM (Milestone 3.1)

experiments/scripts/discrimination.py ran 5 encoders against the 30
cases in data/results_clean.json (10 tasks x 3 language variants),
each case scored against a correct answer and two wrong answers
(wrong_hard: different entity/fact; wrong_subtle: same entity, one
detail wrong). For each (encoder, language) cell, the best threshold
found is the cosine-similarity cutoff that maximizes correct-vs-
wrong_hard discrimination accuracy over that cell's cases.

STATED PLAINLY: each cell is calibrated from 10 cases (10 correct + 10
wrong_hard answers, one task per case, 3 language variants x 10 tasks =
30 cases total, split by variant into 3 cells of 10 per encoder). This
is a small-sample calibration, not a large validation study. Treat
these thresholds as a documented starting point, not ground truth --
see calibrate() below to fit your own on your own labelled data, and
see experiments/results_clean/discrimination_summary.csv for the full
underlying numbers (including subtle-negative discrimination, and the
english_gold/same_language_gold and short/full_sentence variants this
table does not use).

This table uses the english_gold + full_sentence slice specifically:
english_gold because a single canonical gold answer (rather than a
per-language gold) is the realistic production setup most callers have,
and full_sentence because real model answers are full sentences, not
bare entities (see experiments/scripts/beat_the_baseline.py's Milestone
2.4 finding on why bare-entity golds under-score full-sentence
answers). Hard-negative discrimination (not subtle) is used because it
is the more defensible, less noisy signal to calibrate a general-
purpose threshold against; subtle-negative discrimination is a harder,
noisier task better suited to a dedicated correctness judge (see
BUILD_PLAN.md's indic_judge milestone) than to a similarity threshold.

encoder x language -> (threshold, accuracy_at_threshold, accuracy_at_0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MURIL_MODEL_NAME = "google/muril-base-cased"

MURIL_WARNING = (
    "google/muril-base-cased scores ~0.99 cosine similarity on nearly "
    "everything -- correct answers, wrong answers, even different "
    "languages -- with std ~0.002 (see "
    "experiments/results_clean/discrimination_summary.csv). It cannot "
    "discriminate correct from wrong at any threshold. This is the "
    "encoder an Indian-language project reaches for first, because "
    "'Multilingual Representations for Indian Languages' sounds like "
    "exactly the right tool -- it is the trap. See the HindiWiC "
    "finding (Dairkee & Dubossarsky, 2024, 'Strengthening the WiC: New "
    "polysemy dataset in Hindi and lack of cross lingual transfer', "
    "LREC-COLING 2024, https://github.com/haimdub/HindiWiC -- also "
    "cited in this repo's data/trap_words/NOTICE.md): MuRIL scores 55% "
    "zero-shot on Hindi word-sense disambiguation, chance level for a "
    "2-way task, reaching 90% only after Hindi-specific fine-tuning it "
    "does not ship with. Use mpnet or e5 instead, or fine-tune MuRIL "
    "yourself before relying on it."
)

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass(frozen=True, slots=True)
class CalibratedThreshold:
    """One (encoder, language) calibration cell.

    threshold          : cosine-similarity cutoff that maximized
                          correct-vs-wrong_hard discrimination accuracy
                          on the 10 cases in this cell.
    accuracy_at_threshold : discrimination accuracy at `threshold`.
    accuracy_at_default : discrimination accuracy at the naive default
                          of 0.5, for comparison -- this is the number
                          that makes the case for calibrating at all.
    n_cases             : number of (correct, wrong_hard) case pairs
                          this cell was calibrated from. Always 10 for
                          the shipped table -- see module docstring.
    warnings            : populated only by calibrate() (always empty,
                          `()`, on the shipped CALIBRATION_TABLE
                          entries above). Non-fatal signals that this
                          particular fit may not be trustworthy -- see
                          calibrate()'s docstring for exactly what is
                          checked and why these are warnings, not
                          raised errors.
    roc_auc             : threshold-independent ROC AUC for this cell,
                          from experiments/results_clean/
                          discrimination_summary.csv's roc_auc_hard
                          column (0.5 = chance, <0.5 = inverted). Added
                          after repeated independent outside review kept
                          finding the same gap: accuracy_at_threshold is
                          an in-sample argmax fit and can look fine even
                          when a cell has no real discrimination (e.g.
                          multilingual-e5-base/hi fits at 0.632 accuracy
                          here but has real AUC exactly 0.500, chance).
                          calibrate()'s own .warnings guard cannot catch
                          this case -- it only checks same-sample fitted
                          accuracy, not AUC. See
                          calibrated_similarity()'s docstring for how
                          this is now used to warn callers directly,
                          not just documented in README.md's AUC table.
                          None only for a cell with no shipped AUC
                          (never happens for CALIBRATION_TABLE's 15
                          shipped cells; present for completeness).
    """

    threshold: float
    accuracy_at_threshold: float
    accuracy_at_default: float
    n_cases: int = 10
    warnings: tuple[str, ...] = ()
    roc_auc: float | None = None


# Generated by experiments/scripts/generate_calibration_table.py from
# experiments/results_clean/discrimination_per_case.csv,
# gold_mode=english_gold, gold_length=full_sentence, by running the
# real calibrate() function on each cell's real per-case correct/
# wrong_hard similarity scores -- not hand-typed. Re-run that script
# and paste its output here if the underlying data or calibrate()'s
# guard logic ever changes; do not hand-edit these values or their
# `warnings`.
#
# WHY SOME CELLS HAVE WARNINGS AND SOME DON'T (read before trusting an
# empty warnings tuple as "this cell is fine"): calibrate()'s guard
# checks the SAME-SAMPLE argmax-fitted accuracy against chance
# (<=0.5), not the threshold-independent ROC AUC. A cell can fit above
# chance in-sample (so calibrate() stays quiet) while still having
# AUC at or near chance on a proper discrimination measure -- e.g.
# multilingual-e5-base/hi fits at 0.632 accuracy here (no warning) but
# has real AUC 0.500 (exactly chance) per
# discrimination_summary.csv's roc_auc_hard column. An empty
# `warnings` tuple here means "not degenerate by calibrate()'s own
# guard," not "has real discrimination" -- see README.md's "argument
# for calibrated_similarity" section for the full AUC picture across
# all 15 cells, which is the actual authority on which cells have
# real signal.
CALIBRATION_TABLE: dict[str, dict[str, CalibratedThreshold]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "en": CalibratedThreshold(
            0.8629738688468933, 0.65, 0.5, n_cases=10, warnings=(), roc_auc=0.6
        ),
        "hi": CalibratedThreshold(
            0.10077689588069916,
            0.631578947368421,
            0.5263157894736842,
            n_cases=9,
            warnings=(),
            roc_auc=0.5888888888888889,
        ),
        "hinglish": CalibratedThreshold(
            0.35428088903427124, 0.75, 0.6, n_cases=10, warnings=(), roc_auc=0.75
        ),
    },
    "sentence-transformers/LaBSE": {
        "en": CalibratedThreshold(
            0.553279459476471,
            0.5,
            0.5,
            n_cases=10,
            warnings=(
                "fitted accuracy (0.50) is at or below chance -- this data may not "
                "separate correct from wrong at all, or the correct/wrong lists may "
                "be swapped.",
            ),
            roc_auc=0.06999999999999998,
        ),
        "hi": CalibratedThreshold(
            0.5384654402732849,
            0.47368421052631576,
            0.47368421052631576,
            n_cases=9,
            warnings=(
                "fitted accuracy (0.47) is at or below chance -- this data may not "
                "separate correct from wrong at all, or the correct/wrong lists may "
                "be swapped.",
            ),
            roc_auc=0.16666666666666666,
        ),
        "hinglish": CalibratedThreshold(
            0.4422089755535126, 0.75, 0.65, n_cases=10, warnings=(), roc_auc=0.7300000000000001
        ),
    },
    "intfloat/multilingual-e5-base": {
        "en": CalibratedThreshold(
            0.8811413049697876, 0.9, 0.5, n_cases=10, warnings=(), roc_auc=0.86
        ),
        "hi": CalibratedThreshold(
            0.8806519508361816,
            0.631578947368421,
            0.47368421052631576,
            n_cases=9,
            warnings=(),
            roc_auc=0.49999999999999994,
        ),
        "hinglish": CalibratedThreshold(
            0.8469548225402832, 0.75, 0.5, n_cases=10, warnings=(), roc_auc=0.68
        ),
    },
    MURIL_MODEL_NAME: {
        "en": CalibratedThreshold(
            0.9957686066627502,
            0.5,
            0.5,
            n_cases=10,
            warnings=(
                "fitted accuracy (0.50) is at or below chance -- this data may not "
                "separate correct from wrong at all, or the correct/wrong lists may "
                "be swapped.",
            ),
            roc_auc=0.0,
        ),
        "hi": CalibratedThreshold(
            0.992067813873291,
            0.47368421052631576,
            0.47368421052631576,
            n_cases=9,
            warnings=(
                "fitted accuracy (0.47) is at or below chance -- this data may not "
                "separate correct from wrong at all, or the correct/wrong lists may "
                "be swapped.",
            ),
            roc_auc=0.1111111111111111,
        ),
        "hinglish": CalibratedThreshold(
            0.9907472133636475,
            0.6,
            0.5,
            n_cases=10,
            warnings=(),
            roc_auc=0.43999999999999995,
        ),
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "en": CalibratedThreshold(
            0.8324964046478271, 0.85, 0.5, n_cases=10, warnings=(), roc_auc=0.9299999999999999
        ),
        "hi": CalibratedThreshold(
            0.8738334774971008,
            0.7368421052631579,
            0.47368421052631576,
            n_cases=9,
            warnings=(),
            roc_auc=0.7222222222222222,
        ),
        "hinglish": CalibratedThreshold(
            0.41423243284225464, 0.8, 0.7, n_cases=10, warnings=(), roc_auc=0.7300000000000001
        ),
    },
}


def get_threshold(encoder_name: str, language: str) -> CalibratedThreshold:
    """Look up the shipped calibration for (encoder_name, language).

    language is one of "en", "hi", "hinglish" (the three variants the
    calibration data covers). Raises KeyError with a clear message if
    either is not in the shipped table -- callers should fall back to
    0.5 explicitly rather than have this silently guess.
    """
    try:
        by_language = CALIBRATION_TABLE[encoder_name]
    except KeyError:
        raise KeyError(
            f"no shipped calibration for encoder {encoder_name!r}. "
            f"Known encoders: {sorted(CALIBRATION_TABLE)}. "
            "Use calibrate() to fit your own threshold."
        ) from None
    try:
        return by_language[language]
    except KeyError:
        raise KeyError(
            f"no shipped calibration for language {language!r} on "
            f"encoder {encoder_name!r}. Known languages: "
            f"{sorted(by_language)}. Use calibrate() to fit your own "
            "threshold."
        ) from None


def calibrate(
    similarities_correct: list[float], similarities_wrong: list[float]
) -> CalibratedThreshold:
    """Fit a threshold on YOUR labelled data (Milestone 3.3).

    Given cosine similarities for known-correct pairs and known-wrong
    pairs, sweep candidate thresholds (every observed similarity value)
    and return the one that maximizes accuracy: correct pairs scoring
    >= threshold, wrong pairs scoring < threshold.

    The shipped CALIBRATION_TABLE is a starting point calibrated on 10
    cases per cell from one dataset (see module docstring) -- it is not
    a substitute for calibrating on your own data, which this function
    exists to make easy.

    GUARDS (added after an independent outside review found this
    function accepted degenerate input silently -- e.g.
    `calibrate([0.5], [0.5])` returned threshold=0.5 as if it were a
    real fit, and `calibrate([0.1, 0.2], [0.9, 0.8])`, an inverted
    input where "correct" scores are lower than "wrong" scores,
    returned a threshold with no indication anything was backwards):
    this still ALWAYS returns a CalibratedThreshold for any structurally
    valid input -- it does not raise just because a fit looks bad,
    since refusing to fit at all would be worse for a caller with
    genuinely small real-world data. Instead, non-fatal problems are
    surfaced in the returned `.warnings` tuple:
      - fewer than 5 cases on either side ("too few cases to trust
        this fit").
      - similarities outside cosine similarity's valid [-1, 1] range
        on either side ("scores outside [-1, 1] -- is this actually
        cosine similarity?").
      - the fitted threshold does not beat a coin flip
        (accuracy_at_threshold <= 0.5) ("fitted accuracy is at or
        below chance -- this data may not separate correct from wrong
        at all, or the correct/wrong lists may be swapped").
    Always check `.warnings` before trusting a threshold from a small
    or unfamiliar dataset; an empty tuple means none of these specific
    checks fired, not that the fit is necessarily good.

    NaN IS DIFFERENT -- IT RAISES, NOT A WARNING (found by an
    independent outside review): `calibrate([float("nan"), 0.9], [0.1,
    0.2])` used to silently behave as if the NaN entry did not exist
    (`s >= threshold` is False for any threshold when `s` is NaN, so a
    NaN in similarities_correct is treated as always "wrong" without
    comment), and the out-of-[-1,1]-range check could not catch it
    either (`nan < -1.0` and `nan > 1.0` are both False). A NaN
    similarity score is not "a low score" or "an out-of-range score"
    that a warning can meaningfully describe -- it means something
    upstream is broken (a corrupted embedding, a bad cache read), so
    this raises immediately rather than silently fitting around it.
    """
    if not similarities_correct or not similarities_wrong:
        raise ValueError(
            "calibrate() needs at least one correct and one wrong "
            "similarity score to fit a threshold."
        )
    if any(math.isnan(s) for s in similarities_correct + similarities_wrong):
        raise ValueError(
            "calibrate() received a NaN similarity score -- this indicates "
            "corrupted input (e.g. a bad cached embedding or fp16 overflow "
            "upstream), not a valid similarity to fit against. Remove NaN "
            "entries and investigate their source before calibrating."
        )

    n = len(similarities_correct) + len(similarities_wrong)
    candidates = sorted(set(similarities_correct) | set(similarities_wrong))

    best_threshold = candidates[0]
    best_correct = sum(1 for s in similarities_correct if s >= best_threshold)
    best_wrong = sum(1 for s in similarities_wrong if s < best_threshold)
    best_accuracy = (best_correct + best_wrong) / n

    for threshold in candidates[1:]:
        n_correct = sum(1 for s in similarities_correct if s >= threshold)
        n_wrong = sum(1 for s in similarities_wrong if s < threshold)
        accuracy = (n_correct + n_wrong) / n
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy

    accuracy_at_default = (
        sum(1 for s in similarities_correct if s >= 0.5)
        + sum(1 for s in similarities_wrong if s < 0.5)
    ) / n

    warnings: list[str] = []
    n_cases = min(len(similarities_correct), len(similarities_wrong))
    if n_cases < 5:
        warnings.append(
            f"only {n_cases} case(s) on the smaller side -- too few cases to "
            "trust this fit; a threshold from this few points can look "
            "perfect by chance."
        )
    if any(s < -1.0 or s > 1.0 for s in similarities_correct + similarities_wrong):
        warnings.append(
            "some scores are outside [-1, 1] -- is this actually cosine "
            "similarity? calibrate() assumes cosine similarity but does "
            "not enforce it."
        )
    if best_accuracy <= 0.5:
        warnings.append(
            f"fitted accuracy ({best_accuracy:.2f}) is at or below chance -- "
            "this data may not separate correct from wrong at all, or the "
            "correct/wrong lists may be swapped."
        )

    return CalibratedThreshold(
        threshold=best_threshold,
        accuracy_at_threshold=best_accuracy,
        accuracy_at_default=accuracy_at_default,
        n_cases=n_cases,
        warnings=tuple(warnings),
    )
