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

```bash
pip install vindex[judge]
export GROQ_API_KEY=...
```

```python
from vindex import indic_judge

result = indic_judge(
    question="समुद्र तल पर पानी किस तापमान पर उबलता है?",
    answer="समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।",
)

print(result.label)                        # "matched"
print(result.passed)                       # True
print(result.detail["judge_reasoning"])    # judge's own Hindi reasoning
```

`indic_judge(question, answer, gold=None, judge=None, answering_model_id=None)`
is an LLM judge -- the only non-deterministic metric in this package,
because semantic correctness with no reference answer cannot be done
any other way. Built for Indic, not an English judge pointed at Hindi:
the rubric is written in Hindi (this project's own data found an
English-reasoning judge silently mistranslating समुद्र तल as "sea
floor" instead of "sea level" mid-thought, and scoring a correct
answer 0.0 -- see `vindex/judge_rubric.py`), it states explicitly that
Romanized Hindi is not an error, and it is reference-free by default
because a gold reference measurably masks comprehension drift in this
project's own data. Pass `gold=` for reference-based mode, which runs
align-then-judge first (Sarvam's shape): an exact match with `gold`
skips the LLM call entirely.

Conservative by design: `passed` is only `True` at high judge-reported
confidence and a high score -- an ambiguous case is `label="flagged"`,
not a silent pass. Requires the `judge` extra (the `groq` SDK) and a
`GROQ_API_KEY`. See the Limitations section for what this metric does
not (yet) do.

```python
from vindex import check_trace

# Does a judge's own reasoning trace correctly read the source, or did
# it silently misread a known ambiguous term? Works on ANY judge's
# trace, not only indic_judge's -- pass any reasoning text.
result = check_trace(
    source="समुद्र तल पर पानी किस तापमान पर उबलता है?",
    trace="The question asks for the boiling point at the sea floor...",
)

print(result.label)   # "misread_detected"
print(result.passed)  # False
```

`check_trace(source, trace, trap_words=None)` is a deterministic,
dictionary-based check -- no LLM call, free, instant, reproducible.
The error it catches happens *inside* a judge's reasoning before any
output exists, so nothing else in this package (or in a
transliteration layer, or WER) can catch it. The honest claim: this
detects mistranslation of the N known ambiguous terms in
`vindex.trap_words.load_trap_words()`, not mistranslation in general
-- N is currently 70 (see the Limitations section for exactly what
that does and doesn't cover). Pass your own `trap_words=[...]` to use
a different or larger dictionary.
`check_trace_llm_fallback(source, trace, judge, trap_words=None)` is
an opt-in second mode (Sarvam's align-then-judge shape) for
mistranslation categories a fixed dictionary structurally cannot catch
-- a real LLM call, only on the segments that don't align, defaulting
to flagging when the judge's own answer is ambiguous.

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

## Judge selection guidance

Judge model choice is not a solved default -- see
`vindex.judge.JUDGE_SELECTION_GUIDANCE` for the full text. Short
version: `openai/gpt-oss-120b` (via Groq, the shipped default) is
recommended specifically because it's the one model this project has
direct evidence for (`experiments/FINDINGS.md`'s Phase 0 rubric-quality
runs), not because it's assumed to generalize. This project's own
calibration data and the HindiWiC finding both say Indic-language
competence is not a simple function of overall model capability --
don't assume a smaller/cheaper model "should be fine" for Indic judging
without checking against your own data. And never use the same model
(or family) as both judge and the model being judged.

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
- **`script_normalized_match` (a deterministic, transliteration-aware
  answer comparison) is still not shipped as public API.** Its pieces
  (`vindex.normalize`, `vindex.match`, `vindex.transliterate`) exist
  internally -- see the two limitations below for what they can and
  cannot do. `indic_judge` (below) covers factual-correctness checking
  in the meantime, but as an LLM judge, not a deterministic one.
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
- **`indic_judge` vs an English-rubric baseline is still open.** The
  Milestone 6.3 study below validates `indic_judge`'s verdict against
  human graders in isolation (90.3% agreement); it does not yet compare
  that number against an English-rubric judge on the same 62 traces, so
  the specific claim "the Hindi rubric agrees with humans more than an
  English one would" is not yet measured, only motivated by the Phase 0
  qualitative finding (see `experiments/FINDINGS.md`).
- **`indic_judge` is non-deterministic in the sense that matters: it
  calls a hosted LLM.** Temperature is fixed at 0 and the judge model
  is version-pinned and recorded in every result's
  `detail["judge_model_id"]`, but this project's own data showed the
  same model rewording output roughly a third of the time even at
  temperature 0 -- a "0.87" from one run and a "0.85" from a rerun on
  the same input are both plausible, and neither is a bug. Compare
  scores from the same `judge_model_id`, not across a model update.
- **Never judge a model with itself, or a model from the same family.**
  `indic_judge` does a partial, name-based same-family check when you
  pass `answering_model_id` and warns in `detail`, but it cannot
  detect this in general (e.g. a fine-tune with an unrelated-looking
  name). Self-enhancement bias is documented, not theoretical --
  verify your own judge/answering-model pairing.
- **The align-then-judge cost optimization (Milestone 5.4) only
  applies in reference-based mode.** Reference-free mode (the default)
  has nothing to diff against by definition, so every reference-free
  call is a real LLM call -- there is no free shortcut for the
  recommended mode. This is inherent to reference-free judging, not a
  missed optimization.
- **`check_trace`'s dictionary has 70 entries, hand-reviewed, not
  exhaustive.** `data/trap_words/hindiwic_inventory.csv` (60
  HindiWiC-sourced polysemous Hindi nouns) and
  `data/trap_words/own_additions.csv` (15 hand-authored terms:
  misleading compounds, tense-flip time adverbs, fractional numbers,
  Indian large-number words) had their `reading_a`/`reading_b` filled
  in by hand -- drafted with help from a Hindi-fluent LLM (Sarvam) for
  speed, then reviewed and corrected before committing, not machine-
  generated without review (see `vindex/trap_words.py`'s module
  docstring; this is a judgment call this library does not, and
  should not, automate). 5 of the 60 HindiWiC words (तेल, धन, डब्बा,
  संबंध, थान) were deliberately left blank -- no realistically
  confusable wrong reading exists for them, so they're excluded rather
  than forced into a weak pair. Grows from user reports going forward,
  per Milestone 6.2.
- **`check_trace` only catches mistranslation of a term already in the
  dictionary.** By design (Milestone 6.2's honest claim: "detects
  mistranslation of N known ambiguous terms," never "detects
  mistranslation"). A term not yet in the dictionary, or an ambiguity
  that isn't a fixed word-pair at all (e.g. a dropped negation, or an
  idiom translated literally), needs `check_trace_llm_fallback`
  instead -- and even that is scoped to the segments that don't align,
  not a general mistranslation detector.
- **`check_trace`/`indic_judge` human-agreement study (Milestone 6.3):
  62 traces (31 trap words x correct/wrong answer), graded
  independently by two fluent Hindi speakers under a bias protocol
  (`docs/annotation/BIAS_PROTOCOL.md`: ground truth committed before
  grading, blind sheet, frozen rubric, 30% holdout, no cross-visibility
  until both submitted). Result: **90.3% agreement (56/62) between
  human graders and `indic_judge`'s own verdict**, 100% inter-annotator
  agreement between the two humans, no meaningful gap between the
  tuning set (88.4%) and the untouched 30% holdout (94.7%) -- no sign
  of grading drift. Disclosure required by the protocol: one of the
  two graders is the project owner, who also built the metric being
  evaluated; the second grader is an independent fluent Hindi speaker
  with no stake in the result. Do not cite the 90.3% figure without
  this disclosure next to it. All disagreement was on the same small
  set of traces the judge itself flags as conservative-by-design (a
  correct answer marked "flagged" over an edge-case nuance, not a
  missed mistranslation) -- see `docs/annotation/BIAS_PROTOCOL.md` for
  the full breakdown and raw numbers.
