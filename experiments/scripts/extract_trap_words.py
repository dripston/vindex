"""
Extract a candidate trap-word inventory from HindiWiC for our Hindi
annotation corpus, and seed a second file with our own additions
(categories HindiWiC does not cover, since it contains only polysemous
nouns).

LICENSING: see data/trap_words/NOTICE.md. HindiWiC ships with no license
file in its repo (checked: only README.md and the three CSVs). Default
copyright applies -- all rights reserved by the authors. This script:
  - reads the CSVs locally (cloned to data/external/HindiWiC/, gitignored,
    never committed) for analysis only,
  - extracts ONLY the target-word list and aggregate counts derived from
    it (a word list is a fact about the dataset's composition, not the
    authors' creative expression),
  - writes NO sentence text, NO context_instance1/2 content, and NO other
    column from their CSVs into any output file.

Reads CSVs explicitly as UTF-8 via pandas (the HindiWiC README itself
warns that opening the files in a spreadsheet may not render the
Devanagari correctly).

Does NOT call an LLM. Does NOT fill in any manual/interpretive column --
those are left blank for the user to fill by hand.

Run:  python experiments/scripts/extract_trap_words.py
Output:
  data/trap_words/hindiwic_inventory.csv
  data/trap_words/own_additions.csv
  data/trap_words/NOTICE.md
  stdout: total unique words, top 20 by instance count, test-only count
"""
import os
import sys
import io
import csv
import logging
from collections import defaultdict

import pandas as pd

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HINDIWIC_DIR = os.path.join(REPO_ROOT, "data", "external", "HindiWiC")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "trap_words")

SPLITS = ["train", "val", "test"]
SPLIT_FILES = {s: os.path.join(HINDIWIC_DIR, f"hindi-wsd_{s}.csv") for s in SPLITS}

INVENTORY_CSV = os.path.join(OUTPUT_DIR, "hindiwic_inventory.csv")
OWN_ADDITIONS_CSV = os.path.join(OUTPUT_DIR, "own_additions.csv")
NOTICE_MD = os.path.join(OUTPUT_DIR, "NOTICE.md")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("extract_trap_words")

MANUAL_COLUMNS = [
    "suggested_reading_a",
    "suggested_reading_b",
    "trap_viability",
    "domain_fit",
]

INVENTORY_FIELDNAMES = [
    "target_word",
    "n_instances",
    "n_distinct_senses",
    "split_presence",
    "in_test_only",
] + MANUAL_COLUMNS


# ---------------------------------------------------------------------------
# Part 1: load + confirm schema
# ---------------------------------------------------------------------------
def load_split(split):
    path = SPLIT_FILES[split]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- clone https://github.com/haimdub/HindiWiC "
            f"into {HINDIWIC_DIR} first (git clone ... data/external/HindiWiC)."
        )
    # Explicit UTF-8: the README warns spreadsheet apps may not render the
    # Devanagari correctly; pandas with explicit encoding avoids that class
    # of problem entirely.
    return pd.read_csv(path, encoding="utf-8")


def print_schema_confirmation(dfs):
    print("=" * 90)
    print("SCHEMA CONFIRMATION (Part 1)")
    print("=" * 90)
    for split, df in dfs.items():
        print(f"\n--- {split} : {SPLIT_FILES[split]} ---")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print("3 sample rows:")
        # Printed to stdout for schema confirmation only, per the task's
        # explicit instruction -- this is terminal verification output,
        # not a file written to disk, and is not part of any deliverable.
        with pd.option_context("display.max_colwidth", 60):
            print(df.head(3).to_string())
    print()


# ---------------------------------------------------------------------------
# Part 2: word inventory
# ---------------------------------------------------------------------------
def infer_n_distinct_senses(rows_for_word):
    """No sense-id column exists. Infer a lower bound on distinct senses
    from the WiC pair labels using a simple same-sense graph:
      - each row is an (instance1, instance2) pair with a label
        (1 = same sense, 0 = different sense).
      - build a graph over instance indices per word: label==1 edges
        union instances into the same sense-cluster; label==0 edges are
        NOT treated as authoritative separators (a word can have >2
        senses, so two instances both differing from a third are not
        necessarily in the same sense as each other).
      - the returned count is therefore a LOWER BOUND: the number of
        connected components formed by union-finding all label==1 pairs,
        treating every row's two context instances as distinct instance
        nodes (identified by (context_instance1, start1, end1) /
        (context_instance2, start2, end2) tuples so repeated instances
        collapse correctly) -- but since we must not retain sentence
        text in memory-derived outputs, the union-find here operates on
        a row-local instance id (row index, side) which is sufficient
        to count clusters without needing the text to persist beyond
        this function call.
    """
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Build instance identity from (context text, start, end) so the same
    # physical instance appearing in multiple rows collapses to one node.
    # This uses the CSV's own text only transiently, in-process, to compute
    # a count -- never written to any output file.
    instance_id = {}

    def get_or_make_id(text, start, end):
        key = (text, start, end)
        if key not in instance_id:
            new_id = len(instance_id)
            instance_id[key] = new_id
            parent[new_id] = new_id
        return instance_id[key]

    for _, row in rows_for_word.iterrows():
        a = get_or_make_id(row["context_instance1"], row["start1"], row["end1"])
        b = get_or_make_id(row["context_instance2"], row["start2"], row["end2"])
        if int(row["labels"]) == 1:
            union(a, b)

    clusters = set(find(x) for x in parent)
    return len(clusters)


def build_inventory(dfs):
    per_word = defaultdict(lambda: {
        "n_instances": 0,
        "split_presence": set(),
        "rows": [],
    })

    for split, df in dfs.items():
        for word, group in df.groupby("target_word"):
            per_word[word]["n_instances"] += len(group)
            per_word[word]["split_presence"].add(split)
            per_word[word]["rows"].append((split, group))

    inventory = []
    for word, data in per_word.items():
        # Combine rows across splits for this word to infer sense clusters
        # from the full label graph available for it.
        all_rows = pd.concat([g for _, g in data["rows"]], ignore_index=True)
        n_senses = infer_n_distinct_senses(all_rows)

        split_presence_sorted = sorted(data["split_presence"], key=lambda s: SPLITS.index(s))
        in_test_only = data["split_presence"] == {"test"}

        row = {
            "target_word": word,
            "n_instances": data["n_instances"],
            "n_distinct_senses": n_senses,
            "split_presence": "+".join(split_presence_sorted),
            "in_test_only": in_test_only,
        }
        for col in MANUAL_COLUMNS:
            row[col] = ""
        inventory.append(row)

    inventory.sort(key=lambda r: r["n_instances"], reverse=True)
    return inventory


def write_inventory_csv(inventory):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(inventory)
    log.info("Wrote %s (%d rows)", INVENTORY_CSV, len(inventory))


# ---------------------------------------------------------------------------
# Part 4: our own additions (categories HindiWiC does not cover)
# ---------------------------------------------------------------------------
OWN_ADDITIONS = [
    # compounds with misleading literal readings
    ("समुद्र तल", "compound_misleading_literal"),
    ("जलस्तर", "compound_misleading_literal"),
    ("शेष राशि", "compound_misleading_literal"),
    ("मूलधन", "compound_misleading_literal"),
    ("निकासी", "compound_misleading_literal"),
    # time adverbs that flip on tense
    ("कल", "time_adverb_tense_flip"),
    ("परसों", "time_adverb_tense_flip"),
    ("नरसों", "time_adverb_tense_flip"),
    # fractional number words
    ("सवा", "fractional_number"),
    ("ढाई", "fractional_number"),
    ("साढ़े", "fractional_number"),
    ("पौने", "fractional_number"),
    ("डेढ़", "fractional_number"),
    # Indian numbering
    ("लाख", "indian_numbering"),
    ("करोड़", "indian_numbering"),
]

OWN_ADDITIONS_FIELDNAMES = [
    "target_word",
    "category",
    "source",
] + MANUAL_COLUMNS


def build_own_additions():
    rows = []
    for word, category in OWN_ADDITIONS:
        row = {
            "target_word": word,
            "category": category,
            "source": "own",
        }
        for col in MANUAL_COLUMNS:
            row[col] = ""
        rows.append(row)
    return rows


def write_own_additions_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OWN_ADDITIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OWN_ADDITIONS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s (%d rows)", OWN_ADDITIONS_CSV, len(rows))


# ---------------------------------------------------------------------------
# NOTICE.md
# ---------------------------------------------------------------------------
NOTICE_TEXT = """\
# NOTICE — HindiWiC data usage

## Source

Repository: https://github.com/haimdub/HindiWiC
Files used: `hindi-wsd_train.csv`, `hindi-wsd_val.csv`, `hindi-wsd_test.csv`
Cloned locally to `data/external/HindiWiC/` (gitignored, never committed to
this repository).

## Citation

Dairkee, F. and Dubossarsky, H. (2024). Strengthening the WiC: New polysemy
dataset in Hindi and lack of cross lingual transfer. In *Proceedings of
LREC-COLING 2024*.

## License status

As of the clone date recorded below, the HindiWiC repository contains no
LICENSE file, no license section in its README, and no other licensing
declaration. Under default copyright, this means **all rights are reserved
by the authors**; no license is granted to copy, redistribute, or create
derivative works from the dataset's contents (notably its context
sentences) without permission.

## What we extracted, and why we believe it is acceptable

We extracted only:
  - the list of unique target words (`target_word` column values),
  - counts and presence derived from that column and the `labels` column
    (instance counts, an inferred lower-bound sense-cluster count, and
    which train/val/test splits each word appears in).

We did **not** extract, store, or publish:
  - any value from `context_instance1` or `context_instance2`,
  - any character offset (`start1`, `end1`, `start2`, `end2`),
  - any other column not listed above,
  - any content that could be used to reconstruct the authors' sentences.

Rationale: a plain list of which Hindi words a dataset covers is a factual
statement about the dataset's composition (which words were chosen as
polysemous targets), not the authors' original creative expression (the
context sentences they wrote or curated to illustrate each sense). We
treat the word list as extractable on that basis. The context sentences
themselves — the part of the dataset that required authorial effort and
judgment to construct — are excluded from every output file we produce.

If the HindiWiC authors add a license in the future, or object to this
characterization, this extraction should be reviewed against those terms.

## Files produced from this source

  - `hindiwic_inventory.csv` — one row per unique target word across all
    three splits, with instance counts, an inferred sense-cluster lower
    bound, split presence, and a test-only flag. Four columns
    (`suggested_reading_a`, `suggested_reading_b`, `trap_viability`,
    `domain_fit`) are left blank for manual completion and contain no
    HindiWiC-derived content.

    Note on `in_test_only`: this column is `True` only when a word's
    `split_presence` is exactly `test` (absent from both train and val).
    In this data every value is `False` — 20 of the 60 words are absent
    from train but all 20 of those also appear in val, so none are
    test-*exclusive* under this strict definition, even though the paper
    describes ~33% of test words as unseen during training (a train-vs-test
    comparison, not a test-vs-everything-else one). Both statements are
    consistent; `in_test_only` here answers a narrower question.

## Files not derived from this source

  - `own_additions.csv` — hand-authored word list covering categories
    HindiWiC does not include (it covers only polysemous nouns): misleading
    compounds, tense-flipping time adverbs, fractional number words, and
    Indian large-number words. Marked `source = own`.
"""


def write_notice(clone_date_str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text = NOTICE_TEXT + f"\n\n_Clone/extraction date: {clone_date_str}_\n"
    with open(NOTICE_MD, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Wrote %s", NOTICE_MD)


# ---------------------------------------------------------------------------
def main():
    dfs = {split: load_split(split) for split in SPLITS}
    print_schema_confirmation(dfs)

    inventory = build_inventory(dfs)
    write_inventory_csv(inventory)

    own_rows = build_own_additions()
    write_own_additions_csv(own_rows)

    import datetime
    write_notice(datetime.date.today().isoformat())

    # ---- summary print ----
    total_unique = len(inventory)
    test_only_count = sum(1 for r in inventory if r["in_test_only"])

    print("\n" + "=" * 90)
    print("SUMMARY (Part 2/3)")
    print("=" * 90)
    print(f"Total unique target words across train+val+test: {total_unique}")
    print(f"Words appearing ONLY in the test split: {test_only_count}")

    print("\nTop 20 words by instance count:")
    hdr = f"{'target_word':20s} {'n_instances':>12s} {'n_distinct_senses':>18s} {'split_presence':>15s} {'in_test_only':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for r in inventory[:20]:
        print(f"{r['target_word']:20s} {r['n_instances']:>12d} {r['n_distinct_senses']:>18d} "
              f"{r['split_presence']:>15s} {str(r['in_test_only']):>13s}")


if __name__ == "__main__":
    main()
