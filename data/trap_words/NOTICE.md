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


_Clone/extraction date: 2026-09-13_
