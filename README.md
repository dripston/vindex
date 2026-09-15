# vindex

Evaluation metrics for Indic and code-mixed LLM output.

## Install

```bash
pip install vindex
```

## Example

```python
from vindex import script_adherence

result = script_adherence(
    prompt="Mumbai kahan hai?",
    response="Mumbai is the capital of Maharashtra.",
)

print(result.label)   # "language_mismatch"
print(result.passed)  # False
print(result.reason)  # "prompt is Romanized Hindi; response is Roman-script English."
```

`script_adherence(prompt, response)` checks whether a response came back
in the script and language the prompt used -- no reference answer
needed. Labels: `matched`, `mixed`, `script_mismatch`,
`language_mismatch`, `empty`.

```bash
pip install vindex[similarity]
```

```python
from vindex import calibrated_similarity

result = calibrated_similarity(
    gold="The capital of Maharashtra is Mumbai.",
    response="Mumbai is the capital of Maharashtra, the most populous state in India.",
    language="en",
)

print(result.label)   # "similar"
print(result.passed)  # True
print(result.score)   # cosine similarity, e.g. 0.95
```

`calibrated_similarity(gold, response, language, encoder_name=...)`
checks semantic closeness to a gold reference, using a threshold
calibrated per (encoder, language) instead of an uncalibrated 0.5 --
see the table below and the Limitations section for what "calibrated"
means here and its own limits. Needs a gold reference, unlike
`script_adherence`. Requires the `similarity` extra (sentence-
transformers, transformers, torch -- not installed by plain
`pip install vindex`).

## The finding this package is built around

Same model, same 30 questions, one system-prompt change. The first
prompt ("reply in the same language and script the user used") is
ambiguous enough that a Romanized-Hindi ("Hinglish") prompt gets
answered in Devanagari 4 times out of 5. A strict, script-forbidding
prompt fixes it completely.

| variant  | rate, original prompt | rate, strict prompt |
|----------|-----------------------:|---------------------:|
| en       | 1.000                  | 1.000                 |
| hi       | 0.900                  | 0.900                 |
| hinglish | 0.200                  | 1.000                 |

Reproduced by `vindex.script_adherence` against the two source datasets
in `experiments/scripts/validate_vindex_port.py` -- run it yourself:

```bash
python experiments/scripts/validate_vindex_port.py
```

## The argument for calibrated_similarity

At the naive default of 0.5 cosine similarity, 11 of 15 (encoder,
language) combinations in this package's own calibration data score at
or below 0.5 accuracy for correct-vs-wrong discrimination -- a coin
flip does as well. Calibrated per (encoder, language), 10 of 15 clear
0.6, and the best cell reaches 0.90.

| encoder                          | language | accuracy @ 0.5 | calibrated | accuracy @ calibrated |
|-----------------------------------|----------|----------------:|-----------:|------------------------:|
| multilingual-e5-base               | en       | 0.500           | 0.878      | **0.900**               |
| paraphrase-multilingual-mpnet-v2  | en       | 0.500           | 0.818      | 0.850                   |
| paraphrase-multilingual-mpnet-v2  | hinglish | 0.700           | 0.412      | 0.800                   |
| all-MiniLM-L6-v2                  | hinglish | 0.600           | 0.351      | 0.750                   |
| LaBSE                             | hinglish | 0.650           | 0.423      | 0.750                   |
| multilingual-e5-base               | hinglish | 0.500           | 0.845      | 0.750                   |
| paraphrase-multilingual-mpnet-v2  | hi       | 0.474           | 0.871      | 0.737                   |
| all-MiniLM-L6-v2                  | en       | 0.500           | 0.863      | 0.650                   |
| all-MiniLM-L6-v2                  | hi       | 0.526           | 0.072      | 0.632                   |
| multilingual-e5-base               | hi       | 0.474           | 0.880      | 0.632                   |
| muril-base-cased                  | hinglish | 0.500           | 0.991      | 0.600                   |
| LaBSE                             | en       | 0.500           | 0.553      | 0.500                   |
| muril-base-cased                  | en       | 0.500           | 0.996      | 0.500                   |
| LaBSE                             | hi       | 0.474           | 0.867      | 0.526                   |
| muril-base-cased                  | hi       | 0.474           | 0.995      | 0.526                   |

Full methodology, the "13 of 15" vs "11 of 15" note, and how to
reproduce this table: `experiments/README.md`'s Milestone 3 section.

## Limitations

- **9 scripts recognized, nothing else.** Devanagari, Kannada, Tamil,
  Telugu, Bengali, Gujarati, Malayalam, Odia, Gurmukhi, plus Latin.
  Anything else (Cyrillic, CJK, emoji, digits, punctuation-only text)
  has no script bucket of its own and falls through to `mixed` -- this
  is a real gap, not a rare edge case, if your data has other scripts in
  it.
- **`language_mismatch` detection is a v0 heuristic.** It checks for 14
  hand-picked Hindi function words (`hai`, `hain`, `kya`, `nahi`, ...) in
  Romanized text. No transliteration-variant coverage, no verb
  conjugations, no other Romanized Indic languages, not ML-based. One
  matching word is treated as a signal, not proof.
- **Devanagari's danda (।) is shared punctuation.** It lives in the
  Devanagari Unicode block but is reused as a sentence-ending mark in
  Bengali, Odia, Gurmukhi, and others, so a couple of stray
  `devanagari_chars` can show up in a purely non-Devanagari sentence.
  Documented in `vindex/script.py`; doesn't change classification output
  in practice, since real sentences have far more dominant-script
  characters than stray punctuation.
- **No reference-based correctness check.** `script_adherence` verifies
  script/language, not whether the answer is factually right.
  `script_normalized_match` (transliteration-aware answer comparison) is
  on the roadmap, not shipped yet. The pieces it will be built from
  (`vindex.normalize`, `vindex.match`, `vindex.transliterate`) exist
  internally but are not yet public API -- see the two limitations below
  for what they can and cannot do.
- **Transliteration is many-to-many; this is not fully solvable.**
  "tune" can mean the loanword "tune" (ट्यून) or the pronoun+postposition
  "tune" (तूने, "you [did]") -- genuinely different words that share a
  Roman spelling. This isn't a gap unique to this package: a Jio
  engineer working on their own in-house transliteration layer confirmed
  directly that it doesn't fully resolve this class of ambiguity either.
  `vindex.transliterate` picks one deterministic rendering and does not
  attempt disambiguation by context.
- **Where this beats an LLM-based normalizer, for a specific, narrow
  reason.** Sarvam's published work solves the loanword problem ("वह
  doctor" vs "वह डॉक्टर") with an LLM call per case. `vindex.loanwords`
  solves the same class of case with a small, hand-picked lookup table
  (10 words) consulted before phonetic transliteration runs -- so a
  known loanword gets its real spelling instead of a letter-by-letter
  guess, deterministically, for free, reproducibly. This is a genuine
  advantage for exactly that fixed vocabulary, not a general claim that
  a lookup table beats LLM judging -- a loanword outside the table falls
  straight through to phonetic transliteration, unfixed.
- **`calibrated_similarity`'s table is calibrated from 10 cases per
  cell.** Small-sample, from one dataset, one point in time -- a
  starting point, not ground truth. Use `vindex.calibrate()` to fit a
  threshold on your own labelled data. Only 3 languages (en, hi,
  hinglish) and 5 encoders are covered; any other combination falls
  back to an uncalibrated 0.5 threshold and says so plainly in the
  result's `reason` and `detail["calibrated"]`.
- **`calibrated_similarity` measures closeness, not correctness.** Two
  answers can be topically similar and still disagree on the actual
  fact -- this metric will not catch that. It also requires a gold
  reference, unlike `script_adherence`.
- **MuRIL cannot be calibrated into working.** It scores ~0.99 on
  nearly everything regardless of correctness (std ~0.002), so no
  threshold separates its correct answers from its wrong ones.
  Passing it to `calibrated_similarity` raises loudly by default; see
  `vindex.calibration.MURIL_WARNING` for the HindiWiC citation this is
  based on. It is the encoder an Indian-language project reaches for
  first, and the one that fails hardest.
