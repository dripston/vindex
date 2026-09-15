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
    """

    threshold: float
    accuracy_at_threshold: float
    accuracy_at_default: float
    n_cases: int = 10


# Derived from experiments/results_clean/discrimination_summary.csv,
# gold_mode=english_gold, gold_length=full_sentence, roc_auc_hard /
# best_threshold_hard / best_accuracy_hard / accuracy_at_0.5_hard
# columns. See experiments/scripts/beat_the_baseline.py-adjacent
# analysis for how this slice was selected.
CALIBRATION_TABLE: dict[str, dict[str, CalibratedThreshold]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "en": CalibratedThreshold(0.8627006411552429, 0.65, 0.5),
        "hi": CalibratedThreshold(0.07167930901050568, 0.631578947368421, 0.5263157894736842),
        "hinglish": CalibratedThreshold(0.35093092918395996, 0.75, 0.6),
    },
    "sentence-transformers/LaBSE": {
        "en": CalibratedThreshold(0.5532784594764709, 0.5, 0.5),
        "hi": CalibratedThreshold(0.8666171298751831, 0.5263157894736842, 0.47368421052631576),
        "hinglish": CalibratedThreshold(0.42336905002593994, 0.75, 0.65),
    },
    "intfloat/multilingual-e5-base": {
        "en": CalibratedThreshold(0.8776235282421112, 0.9, 0.5),
        "hi": CalibratedThreshold(0.8800912499427795, 0.631578947368421, 0.47368421052631576),
        "hinglish": CalibratedThreshold(0.8451241850852966, 0.75, 0.5),
    },
    MURIL_MODEL_NAME: {
        "en": CalibratedThreshold(0.9957676066627502, 0.5, 0.5),
        "hi": CalibratedThreshold(0.995097504688263, 0.5263157894736842, 0.47368421052631576),
        "hinglish": CalibratedThreshold(0.9906428754329681, 0.6, 0.5),
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "en": CalibratedThreshold(0.8179083466529846, 0.85, 0.5),
        "hi": CalibratedThreshold(0.8705282509326935, 0.7368421052631579, 0.47368421052631576),
        "hinglish": CalibratedThreshold(0.41158825159072876, 0.8, 0.7),
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
    """
    if not similarities_correct or not similarities_wrong:
        raise ValueError(
            "calibrate() needs at least one correct and one wrong "
            "similarity score to fit a threshold."
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

    return CalibratedThreshold(
        threshold=best_threshold,
        accuracy_at_threshold=best_accuracy,
        accuracy_at_default=accuracy_at_default,
        n_cases=min(len(similarities_correct), len(similarities_wrong)),
    )
