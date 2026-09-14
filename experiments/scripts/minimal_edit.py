"""
Minimal-edit negatives: for each correct model answer, produce two
corrupted versions by editing the ORIGINAL model-answer text in place
(minimal_hard: swap the key entity; minimal_subtle: swap one secondary
detail), so "correct" and "wrong" differ in exactly one thing -- the fact
-- and not in authorship, style, or length, the way the hand-authored
wrong_hard/wrong_subtle negatives in testcases.py do.

Compares four negative-construction methods against the same correct
answers: wrong_hard, wrong_subtle (hand-authored, from testcases.py) vs
minimal_hard, minimal_subtle (minimal edits of the model's own text,
authored here).

Reuses EncoderWrapper, cosine, e5 prefixing, and MuRIL pooling from
encoder_comparison.py verbatim (imported, not reimplemented). Does not
call an LLM to write the edits (hand-authored find/replace pairs below)
and does not modify the dataset it reads.

Runnable standalone as:  python experiments/scripts/minimal_edit.py [--data PATH]
  (defaults to data/results_clean.json -> experiments/results_clean/)
Or pointed at another dataset by experiments/scripts/rerun_all.py, which
passes DATASET_PATH and RESULTS_DIR explicitly before calling main().

Output:
  results_clean/minimal_edit_per_case.csv
  results_clean/minimal_edit_summary.csv
  results_clean/authorship_check.csv
  results_clean/auc_handwritten_vs_minimal.png
  authorship check table printed to stdout

This script only computes numbers. It does not interpret them.
"""
import os
import sys
import csv
import json
import random
import logging
import difflib
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Determinism (identical setup to encoder_comparison.py / discrimination.py)
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    torch = None

# ---------------------------------------------------------------------------
# Paths / cache -- module-level globals, overridable by rerun_all.py exactly
# like encoder_comparison.py's RESULTS_DIR.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
DATASET_PATH = os.path.join(REPO_ROOT, "data", "results_clean.json")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results_clean")

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("minimal_edit")

from testcases import TASKS, VARIANTS  # noqa: E402
from encoder_comparison import (  # noqa: E402
    EncoderWrapper, cosine, ENCODERS, _hf_cache_has,
)
from discrimination import (  # noqa: E402
    roc_auc, best_threshold_accuracy, accuracy_at_threshold,
)

GOLD_MODES = ["english_gold", "same_language_gold"]
GOLD_LENGTHS = ["short", "full_sentence"]
LABELS = ["correct", "wrong_hard", "wrong_subtle", "minimal_hard", "minimal_subtle"]
NEGATIVE_LABELS = ["wrong_hard", "wrong_subtle", "minimal_hard", "minimal_subtle"]

# High edit-distance-to-length ratio flags a "minimal" edit that wasn't
# actually minimal (see check_minimality()).
MINIMALITY_RATIO_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Part 1: minimal-edit (find, replace) pairs, hand-authored per case_id.
# `find` is normalized-space-tolerant: matched against the real answer with
# both the literal space and U+202F (narrow no-break space, which some
# model outputs use before units/numbers) accepted as equivalent, so the
# same table works whether the source used one or the other.
# ---------------------------------------------------------------------------
EDITS = {
    "capital_maharashtra__en": {
        "minimal_hard": ("Mumbai is the capital", "Chennai is the capital"),
        "minimal_subtle": ("the most populous state", "the second most populous state"),
    },
    "capital_maharashtra__hi": {
        "minimal_hard": ("राजधानी मुंबई है", "राजधानी चेन्नई है"),
        "minimal_subtle": ("सबसे बड़ा और आर्थिक", "दूसरा सबसे बड़ा और आर्थिक"),
    },
    "capital_maharashtra__hinglish": {
        "minimal_hard": ("rajdhani Mumbai hai", "rajdhani Chennai hai"),
        "minimal_subtle": ("India ka financial capital", "India ka doosra financial capital"),
    },
    "planets_count__en": {
        "minimal_hard": ("There are eight planets", "There are twelve planets"),
        "minimal_subtle": ("reclassified as a dwarf planet in 2006", "reclassified as a dwarf planet in 2009"),
    },
    "planets_count__hi": {
        "minimal_hard": ("सौर मंडल में 8 ग्रह", "सौर मंडल में 12 ग्रह"),
        "minimal_subtle": ("2006 में प्लूटो", "2009 में प्लूटो"),
    },
    "planets_count__hinglish": {
        "minimal_hard": ("Saur Mandal mein 8 grah", "Saur Mandal mein 12 grah"),
        "minimal_subtle": ("Uranus, aur Neptune", "Uranus, aur Pluto"),
    },
    "freedom_year__en": {
        "minimal_hard": ("independence in 1947", "independence in 1857"),
        "minimal_subtle": ("nearly 200 years", "nearly 300 years"),
    },
    "freedom_year__hi": {
        "minimal_hard": ("भारत को 1947 में", "भारत को 1857 में"),
        "minimal_subtle": ("15 अगस्त", "26 जनवरी"),
    },
    "freedom_year__hinglish": {
        "minimal_hard": ("Bharat ko 1947 mein", "Bharat ko 1857 mein"),
        "minimal_subtle": ("15 August", "26 January"),
    },
    "water_formula__en": {
        "minimal_hard": ("water is **H₂O**", "water is **CO₂**"),
        "minimal_subtle": ("two hydrogen atoms", "three hydrogen atoms"),
    },
    "water_formula__hi": {
        "minimal_hard": ("सूत्र **H₂O** है", "सूत्र **CO₂** है"),
        "minimal_subtle": ("दो हाइड्रोजन और", "तीन हाइड्रोजन और"),
    },
    "water_formula__hinglish": {
        "minimal_hard": ("sutra H₂O hai", "sutra CO₂ hai"),
        "minimal_subtle": ("do hydrogen atoms", "teen hydrogen atoms"),
    },
    "national_animal_india__en": {
        "minimal_hard": ("is the Bengal tiger", "is the Asiatic lion"),
        "minimal_subtle": ("is the Bengal tiger", "is the Siberian tiger"),
    },
    "national_animal_india__hi": {
        "minimal_hard": ("पशु हाथी है", "पशु एशियाई शेर है"),
        "minimal_subtle": ("1972 में राष्ट्रीय पशु", "1982 में राष्ट्रीय पशु"),
    },
    "national_animal_india__hinglish": {
        "minimal_hard": ("rashtriya pashu bhains", "rashtriya pashu sher"),
        "minimal_subtle": ("rashtriya pashu bhains (water buffalo) hai", "rashtriya pashu bhains (domestic buffalo) hai"),
    },
    "largest_ocean__en": {
        "minimal_hard": ("The Pacific Ocean is the largest", "The Atlantic Ocean is the largest"),
        "minimal_subtle": ("about 63 million square miles", "about 33 million square miles"),
    },
    "largest_ocean__hi": {
        "minimal_hard": ("पैसिफिक महासागर पृथ्वी का सबसे बड़ा", "अटलांटिक महासागर पृथ्वी का सबसे बड़ा"),
        "minimal_subtle": ("लगभग 46.6 मिलियन वर्ग मील", "लगभग 66.6 मिलियन वर्ग मील"),
    },
    "largest_ocean__hinglish": {
        "minimal_hard": ("Sabse bada mahasagar Pacific Ocean hai", "Sabse bada mahasagar Atlantic Ocean hai"),
        "minimal_subtle": ("dharti ke 46% surface area", "dharti ke 66% surface area"),
    },
    "sun_rise_direction__en": {
        "minimal_hard": ("sun rises in the east", "sun rises in the west"),
        "minimal_subtle": ("rotates from west to east", "rotates from east to west"),
    },
    "sun_rise_direction__hi": {
        "minimal_hard": ("सूर्य पूर्व दिशा में", "सूर्य पश्चिम दिशा में"),
        "minimal_subtle": ("सूर्य पूर्व दिशा में", "सूर्य उत्तर-पूर्व दिशा में"),
    },
    "sun_rise_direction__hinglish": {
        "minimal_hard": ("Suraj purvi disha", "Suraj paschimi disha"),
        "minimal_subtle": ("Suraj purvi disha", "Suraj uttar-purvi disha"),
    },
    "days_in_leap_year__en": {
        "minimal_hard": ("leap year has 366 days", "leap year has 400 days"),
        "minimal_subtle": ("leap year has 366 days", "leap year has 365 days"),
    },
    "days_in_leap_year__hi": {
        "minimal_hard": ("वर्ष में 366 दिन", "वर्ष में 400 दिन"),
        "minimal_subtle": ("फरवरी में 29 दिन", "फरवरी में 30 दिन"),
    },
    "days_in_leap_year__hinglish": {
        "minimal_hard": ("leap year mein 366 din", "leap year mein 400 din"),
        "minimal_subtle": ("February mein 29th add", "February mein 30th add"),
    },
    "father_of_nation_india__en": {
        "minimal_hard": ("Mahatma Gandhi is known", "Jawaharlal Nehru is known"),
        "minimal_subtle": ("the “Father of the Nation”", "the “Netaji of the Nation”"),
    },
    # father_of_nation_india__hi is EMPTY in every dataset generated so far
    # -- no entry here; excluded downstream exactly like discrimination.py.
    "father_of_nation_india__hinglish": {
        "minimal_hard": ("Mahatma Gandhi ko Bharat ka rashtrapita", "Jawaharlal Nehru ko Bharat ka rashtrapita"),
        "minimal_subtle": ("Bharat ka rashtrapita mana jata hai", "Bharat ka Netaji mana jata hai"),
    },
    "boiling_point_water_c__en": {
        "minimal_hard": ("Water boils at 100 °C", "Water boils at 50 °C"),
        "minimal_subtle": ("Water boils at 100 °C", "Water boils at 0 °C"),
    },
    "boiling_point_water_c__hi": {
        "minimal_hard": ("पानी 100 डिग्री सेल्सियस पर", "पानी 50 डिग्री सेल्सियस पर"),
        "minimal_subtle": ("पानी 100 डिग्री सेल्सियस पर", "पानी 0 डिग्री सेल्सियस पर"),
    },
    "boiling_point_water_c__hinglish": {
        "minimal_hard": ("samudra tal par 100°C", "samudra tal par 50°C"),
        "minimal_subtle": ("samudra tal par 100°C", "samudra tal par 0°C"),
    },
}


def _space_tolerant_find(haystack, needle):
    """Search for `needle` in `haystack`, treating a regular space (0x20)
    in `needle` as matching either 0x20 or U+202F (narrow no-break space)
    in `haystack`. Returns (start, end, matched_substring) or None."""
    import re
    if " " not in needle:
        idx = haystack.find(needle)
        if idx == -1:
            return None
        return idx, idx + len(needle), needle
    pattern = re.escape(needle).replace(r"\ ", "[  ]")
    m = re.search(pattern, haystack)
    if not m:
        return None
    return m.start(), m.end(), m.group(0)


def apply_edit(original, find, replace):
    """Apply one space-tolerant find/replace, preserving whichever space
    character the original actually used at the matched span. Returns
    (edited_text, matched, n_matches) where matched is the exact substring
    that was found (for edit-distance/minimality bookkeeping) and n_matches
    counts ALL occurrences of `find` in `original` (space-tolerant), so a
    caller can detect an ambiguous (non-unique) anchor."""
    import re
    if " " not in find:
        n_matches = original.count(find)
        pos = original.find(find)
        if pos == -1:
            return None, None, 0
        edited = original[:pos] + replace + original[pos + len(find):]
        return edited, find, n_matches

    pattern = re.escape(find).replace(r"\ ", "[  ]")
    matches = list(re.finditer(pattern, original))
    if not matches:
        return None, None, 0
    m = matches[0]
    matched_text = m.group(0)
    # Substitute the replacement's spaces with whichever space character
    # appears in the matched original span at each corresponding position,
    # best-effort: since find/replace may differ in space count, we simply
    # reuse the first space character found in the match for all spaces in
    # the replacement -- preserves script/formatting intent without
    # requiring a 1:1 character mapping.
    space_char = " "
    for ch in matched_text:
        if ch in (" ", " "):
            space_char = ch
            break
    replace_adjusted = replace.replace(" ", space_char)
    edited = original[:m.start()] + replace_adjusted + original[m.end():]
    return edited, matched_text, len(matches)


def check_minimality(original, edited):
    """Character-level edit distance (Levenshtein via difflib's opcodes,
    which is deterministic and dependency-free) between original and
    edited, plus the ratio relative to the original's length. Flags when
    the ratio exceeds MINIMALITY_RATIO_THRESHOLD, meaning the edit was not
    minimal."""
    sm = difflib.SequenceMatcher(None, original, edited)
    ratio_similarity = sm.ratio()
    # Convert similarity ratio to an edit-distance-like count: total edited
    # chars = chars not matched on either side.
    matches = sum(block.size for block in sm.get_matching_blocks())
    edit_distance = (len(original) - matches) + (len(edited) - matches)
    length = max(len(original), 1)
    ratio = edit_distance / length
    flagged = ratio > MINIMALITY_RATIO_THRESHOLD
    return edit_distance, ratio, flagged


def build_minimal_edit_cases(dataset_path):
    """Loads `dataset_path` (schema: list of {"case_id", "answer", ...}),
    joins with TASKS' hand-authored wrong_hard/wrong_subtle (from
    testcases.py) and this module's EDITS table, and returns the 5-label
    case list: correct, wrong_hard, wrong_subtle, minimal_hard,
    minimal_subtle -- for every task x variant where a correct answer
    exists and is non-empty."""
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    answer_rows = {r["case_id"]: r["answer"] for r in rows
                   if not r.get("script_adherence_failure", False)}

    task_by_id = {t["id"]: t for t in TASKS}
    cases = []
    edit_meta = []  # per (case_id) minimality bookkeeping, for logging

    for task in TASKS:
        for v in VARIANTS:
            case_id = f"{task['id']}__{v}"
            original = answer_rows.get(case_id)
            if not original:
                log.warning("No usable correct answer for %s -- skipping all labels for this case.", case_id)
                continue

            gold_short_same_key = "gold" if v == "en" else f"gold_{v}"
            gold_full_same_key = "gold_en_full" if v == "en" else f"gold_{v}_full"
            golds = {
                ("english_gold", "short"): task["gold"],
                ("english_gold", "full_sentence"): task["gold_en_full"],
                ("same_language_gold", "short"): task[gold_short_same_key],
                ("same_language_gold", "full_sentence"): task[gold_full_same_key],
            }

            answers_by_label = {"correct": original}

            # hand-authored negatives (existing method, for comparison)
            for label in ["wrong_hard", "wrong_subtle"]:
                answers_by_label[label] = task[label][v]

            # minimal-edit negatives (new method)
            edits_for_case = EDITS.get(case_id)
            if edits_for_case is None:
                log.warning("No EDITS entry for %s -- skipping minimal_hard/minimal_subtle for this case.", case_id)
            else:
                for label in ["minimal_hard", "minimal_subtle"]:
                    find, replace = edits_for_case[label]
                    edited, matched, n_matches = apply_edit(original, find, replace)
                    if edited is None:
                        log.error("Anchor %r not found in %s answer -- skipping %s.", find, case_id, label)
                        continue
                    if n_matches > 1 and label == "minimal_hard" and case_id not in (
                        "freedom_year__hi", "freedom_year__hinglish"  # intentionally replace all occurrences (year appears twice)
                    ):
                        log.warning("Anchor %r matches %d times in %s -- using the first match only.",
                                    find, n_matches, case_id)
                    edit_distance, ratio, flagged = check_minimality(original, edited)
                    answers_by_label[label] = edited
                    edit_meta.append({
                        "case_id": case_id,
                        "label": label,
                        "edit_distance": edit_distance,
                        "edit_distance_ratio": ratio,
                        "flagged_non_minimal": flagged,
                        "matched_span": matched,
                    })

            for label in LABELS:
                answer = answers_by_label.get(label)
                if not answer:
                    continue
                cases.append({
                    "case_id": case_id if label == "correct" else f"{case_id}__{label}",
                    "task_id": task["id"],
                    "variant": v,
                    "label": label,
                    "answer": answer,
                    "golds": golds,
                })

    return cases, edit_meta


def _set_offline_mode_if_all_cached():
    all_cached = all(_hf_cache_has(m) for m in ENCODERS)
    if all_cached:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log.info("All %d encoders already cached -- running fully offline.", len(ENCODERS))
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        log.info("Not all encoders cached yet -- allowing network for first-time downloads.")


def main(dataset_path=None):
    _set_offline_mode_if_all_cached()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cases, edit_meta = build_minimal_edit_cases(dataset_path or DATASET_PATH)

    print("\n--- minimality check (edit distance / length) ---")
    for m in edit_meta:
        flag = " <-- FLAGGED NOT MINIMAL" if m["flagged_non_minimal"] else ""
        print(f"  {m['case_id']:42s} {m['label']:16s} dist={m['edit_distance']:>3d} "
              f"ratio={m['edit_distance_ratio']:.3f}{flag}")

    per_case_rows = []
    score_store = {}  # (encoder, gold_mode, gold_length, variant, label) -> [sims]
    self_sim_store = {"minimal_hard": [], "wrong_hard": []}  # authorship probe

    for model_name in ENCODERS:
        log.info("Loading encoder: %s", model_name)
        try:
            enc = EncoderWrapper(model_name).load()
        except Exception as e:
            log.error("Failed to load %s: %s -- skipping this encoder.", model_name, e)
            continue

        # authorship probe: correct vs its own minimal_hard / wrong_hard,
        # DIRECTLY (no gold involved) -- computed once per encoder.
        by_case_id = {}
        for c in cases:
            by_case_id.setdefault(c["task_id"] + "__" + c["variant"], {})[c["label"]] = c["answer"]
        for key, labels in by_case_id.items():
            if "correct" not in labels:
                continue
            correct_emb = enc.encode([labels["correct"]], is_query=True)[0]
            if "minimal_hard" in labels:
                mh_emb = enc.encode([labels["minimal_hard"]], is_query=True)[0]
                self_sim_store["minimal_hard"].append((model_name, cosine(correct_emb, mh_emb)))
            if "wrong_hard" in labels:
                wh_emb = enc.encode([labels["wrong_hard"]], is_query=True)[0]
                self_sim_store["wrong_hard"].append((model_name, cosine(correct_emb, wh_emb)))

        for gold_mode, gold_length in product(GOLD_MODES, GOLD_LENGTHS):
            for c in cases:
                gold_text = c["golds"][(gold_mode, gold_length)]
                try:
                    ans_emb = enc.encode([c["answer"]], is_query=True)[0]
                    gold_emb = enc.encode([gold_text], is_query=False)[0]
                    sim = cosine(ans_emb, gold_emb)
                except Exception as e:
                    log.error("Scoring failed for %s / %s / %s / %s: %s",
                              model_name, gold_mode, gold_length, c["case_id"], e)
                    continue

                key = (model_name, gold_mode, gold_length, c["variant"], c["label"])
                score_store.setdefault(key, []).append(sim)

                per_case_rows.append({
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "gold_length": gold_length,
                    "variant": c["variant"],
                    "label": c["label"],
                    "case_id": c["case_id"],
                    "task_id": c["task_id"],
                    "gold_text": gold_text,
                    "answer": c["answer"],
                    "cosine_similarity": sim,
                })

        del enc
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Part 3: metrics per (encoder, gold_mode, gold_length, variant), for
    # each negative type.
    # -----------------------------------------------------------------
    summary_rows = []
    for model_name in ENCODERS:
        for gold_mode, gold_length in product(GOLD_MODES, GOLD_LENGTHS):
            for v in VARIANTS:
                by_label = {}
                for label in LABELS:
                    key = (model_name, gold_mode, gold_length, v, label)
                    by_label[label] = score_store.get(key, [])
                if not any(by_label.values()):
                    continue

                row = {"encoder": model_name, "gold_mode": gold_mode,
                       "gold_length": gold_length, "variant": v}
                for label in LABELS:
                    scores = by_label[label]
                    row[f"n_{label}"] = len(scores)
                    row[f"mean_{label}"] = float(np.mean(scores)) if scores else None
                    row[f"std_{label}"] = float(np.std(scores, ddof=0)) if scores else None

                correct_scores = by_label["correct"]
                for neg in NEGATIVE_LABELS:
                    neg_scores = by_label[neg]
                    if correct_scores and neg_scores:
                        row[f"separation_{neg}"] = row["mean_correct"] - row[f"mean_{neg}"]
                        row[f"roc_auc_{neg}"] = roc_auc(correct_scores, neg_scores)
                        best_thr, best_acc = best_threshold_accuracy(correct_scores, neg_scores)
                        row[f"best_threshold_{neg}"] = best_thr
                        row[f"best_accuracy_{neg}"] = best_acc
                        row[f"accuracy_at_0.5_{neg}"] = accuracy_at_threshold(correct_scores, neg_scores, 0.5)
                    else:
                        row[f"separation_{neg}"] = None
                        row[f"roc_auc_{neg}"] = None
                        row[f"best_threshold_{neg}"] = None
                        row[f"best_accuracy_{neg}"] = None
                        row[f"accuracy_at_0.5_{neg}"] = None

                summary_rows.append(row)

    # -----------------------------------------------------------------
    # Part 4: the authorship check table.
    # -----------------------------------------------------------------
    authorship_rows = []
    for model_name in ENCODERS:
        for gold_mode, gold_length in product(GOLD_MODES, GOLD_LENGTHS):
            for v in VARIANTS:
                match = [r for r in summary_rows
                         if r["encoder"] == model_name and r["gold_mode"] == gold_mode
                         and r["gold_length"] == gold_length and r["variant"] == v]
                if not match:
                    continue
                r = match[0]
                auc_hw = r.get("roc_auc_wrong_hard")
                auc_min = r.get("roc_auc_minimal_hard")
                delta = (auc_min - auc_hw) if (auc_hw is not None and auc_min is not None) else None
                authorship_rows.append({
                    "encoder": model_name,
                    "gold_mode": gold_mode,
                    "gold_length": gold_length,
                    "variant": v,
                    "auc_handwritten": auc_hw,
                    "auc_minimal": auc_min,
                    "delta": delta,
                })

    # direct probes (no gold involved): correct vs its own minimal_hard /
    # wrong_hard, and mean string length per label per variant.
    direct_probe_rows = []
    for model_name in ENCODERS:
        mh_scores = [s for m, s in self_sim_store["minimal_hard"] if m == model_name]
        wh_scores = [s for m, s in self_sim_store["wrong_hard"] if m == model_name]
        direct_probe_rows.append({
            "encoder": model_name,
            "mean_cosine_correct_vs_own_minimal_hard": float(np.mean(mh_scores)) if mh_scores else None,
            "mean_cosine_correct_vs_handwritten_wrong_hard": float(np.mean(wh_scores)) if wh_scores else None,
        })

    length_rows = []
    for v in VARIANTS:
        for label in LABELS:
            lens = [len(c["answer"]) for c in cases if c["variant"] == v and c["label"] == label]
            length_rows.append({
                "variant": v, "label": label, "n": len(lens),
                "mean_string_length": float(np.mean(lens)) if lens else None,
            })

    # -----------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------
    _write_csv(os.path.join(RESULTS_DIR, "minimal_edit_per_case.csv"), per_case_rows)
    _write_csv(os.path.join(RESULTS_DIR, "minimal_edit_summary.csv"), summary_rows)

    authorship_path = os.path.join(RESULTS_DIR, "authorship_check.csv")
    _write_authorship_csv(authorship_path, authorship_rows, direct_probe_rows, length_rows)

    print_authorship_table(authorship_rows, direct_probe_rows, length_rows)
    make_auc_comparison_chart(summary_rows)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            fieldnames = list(rows[0].keys())
            for r in rows:
                for k in fieldnames:
                    r.setdefault(k, None)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    log.info("Wrote %s (%d rows)", path, len(rows))


def _write_authorship_csv(path, authorship_rows, direct_probe_rows, length_rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# Section 1: AUC delta (handwritten vs minimal-edit negatives), correct vs *_hard"])
        writer.writerow(["encoder", "gold_mode", "gold_length", "variant",
                          "auc_handwritten", "auc_minimal", "delta"])
        for r in authorship_rows:
            writer.writerow([r["encoder"], r["gold_mode"], r["gold_length"], r["variant"],
                              r["auc_handwritten"], r["auc_minimal"], r["delta"]])
        writer.writerow([])
        writer.writerow(["# Section 2: direct probe -- correct vs its own edit, no gold involved"])
        writer.writerow(["encoder", "mean_cosine_correct_vs_own_minimal_hard",
                          "mean_cosine_correct_vs_handwritten_wrong_hard"])
        for r in direct_probe_rows:
            writer.writerow([r["encoder"], r["mean_cosine_correct_vs_own_minimal_hard"],
                              r["mean_cosine_correct_vs_handwritten_wrong_hard"]])
        writer.writerow([])
        writer.writerow(["# Section 3: mean string length by variant x label"])
        writer.writerow(["variant", "label", "n", "mean_string_length"])
        for r in length_rows:
            writer.writerow([r["variant"], r["label"], r["n"], r["mean_string_length"]])
    log.info("Wrote %s", path)


def print_authorship_table(authorship_rows, direct_probe_rows, length_rows):
    print("\n" + "=" * 110)
    print("PART 4: AUTHORSHIP CHECK")
    print("=" * 110)
    print("\n-- AUC: handwritten (correct vs wrong_hard) vs minimal-edit (correct vs minimal_hard) --")
    hdr = f"{'encoder':46s} {'gold_mode':19s} {'gold_len':13s} {'var':9s} {'auc_hw':>8s} {'auc_min':>8s} {'delta':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in authorship_rows:
        def f3(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{r['encoder']:46s} {r['gold_mode']:19s} {r['gold_length']:13s} {r['variant']:9s} "
              f"{f3(r['auc_handwritten']):>8s} {f3(r['auc_minimal']):>8s} {f3(r['delta']):>8s}")

    print("\n-- direct probe: correct vs its own edit (no gold reference involved) --")
    hdr2 = f"{'encoder':46s} {'mean_sim(correct, own minimal_hard)':>36s} {'mean_sim(correct, handwritten wrong_hard)':>42s}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in direct_probe_rows:
        def f3(x):
            return f"{x:.3f}" if isinstance(x, float) else "  -  "
        print(f"{r['encoder']:46s} {f3(r['mean_cosine_correct_vs_own_minimal_hard']):>36s} "
              f"{f3(r['mean_cosine_correct_vs_handwritten_wrong_hard']):>42s}")

    print("\n-- mean answer string length (characters), by variant x label --")
    hdr3 = f"{'variant':10s} {'label':16s} {'n':>4s} {'mean_length':>12s}"
    print(hdr3)
    print("-" * len(hdr3))
    for r in length_rows:
        ml = f"{r['mean_string_length']:.1f}" if r["mean_string_length"] is not None else "  -  "
        print(f"{r['variant']:10s} {r['label']:16s} {r['n']:>4d} {ml:>12s}")


def make_auc_comparison_chart(summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gold_mode, gold_length = "english_gold", "full_sentence"
    rows = [r for r in summary_rows if r["gold_mode"] == gold_mode and r["gold_length"] == gold_length]

    encoders = []
    seen = set()
    for r in rows:
        if r["encoder"] not in seen:
            encoders.append(r["encoder"])
            seen.add(r["encoder"])
    short_names = [e.split("/")[-1] for e in encoders]

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(6 * len(VARIANTS), 6), sharey=True)
    if len(VARIANTS) == 1:
        axes = [axes]

    x = np.arange(len(encoders))
    width = 0.35

    for ax, v in zip(axes, VARIANTS):
        hw_vals, min_vals = [], []
        for enc_name in encoders:
            match = [r for r in rows if r["encoder"] == enc_name and r["variant"] == v]
            if match:
                hw_vals.append(match[0]["roc_auc_wrong_hard"] or 0)
                min_vals.append(match[0]["roc_auc_minimal_hard"] or 0)
            else:
                hw_vals.append(0)
                min_vals.append(0)
        ax.bar(x - width / 2, hw_vals, width, label="hand-authored (wrong_hard)", color="#4C72B0")
        ax.bar(x + width / 2, min_vals, width, label="minimal-edit (minimal_hard)", color="#C44E52")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"variant = {v}")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Encoder")

    axes[0].set_ylabel("ROC AUC (correct vs negative)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"AUC: hand-authored vs minimal-edit negatives\n(gold_mode={gold_mode}, gold_length={gold_length})")
    fig.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "auc_handwritten_vs_minimal.png")
    fig.savefig(chart_path, dpi=150)
    log.info("Wrote %s", chart_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="dataset_path", default=None,
                         help=f"Path to the answers JSON (default: {DATASET_PATH})")
    args = parser.parse_args()
    main(args.dataset_path)
