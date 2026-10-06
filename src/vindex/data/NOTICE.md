# Bundled data

## `trap_words/hindiwic_inventory.csv`

A list of polysemous Hindi target words, derived from the HindiWiC dataset:

> Dairkee, F. and Dubossarsky, H. (2024). *Strengthening the WiC: New polysemy
> dataset in Hindi and lack of cross lingual transfer.* LREC-COLING 2024.
> https://github.com/haimdub/HindiWiC

Only the unique `target_word` values and counts derived from the `target_word`
and `labels` columns are included. No context sentences, character offsets, or
other HindiWiC content are redistributed. The HindiWiC repository declares no
license; if its authors add one or object to this use, this file will be
reviewed against their terms. The `reading_a` / `reading_b` columns are
original work.

## `trap_words/own_additions.csv`

Original, hand-authored entries (misleading compounds, tense-flipping time
adverbs, fractional number words, Indian large-number words). MIT licensed
with the rest of vindex.

## `benchmarks/`

Labelled cases used to calibrate `calibrated_similarity` and to measure
`indic_judge` agreement with human graders. Original work, MIT licensed.
One of the two human graders in `judge_benchmark.json` is the project author;
report agreement against both graders, not grader 1 alone.
