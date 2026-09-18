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
    response="Mumbai Maharashtra mein hai.",
)

print(result.label)   # "matched"
print(result.passed)  # True
```

`script_adherence(prompt, response, strict_language_check=False)`
checks whether a response came back in the script and language the
prompt used -- no reference answer needed. Labels: `matched`, `mixed`,
`script_mismatch`, `language_mismatch` (only reachable with
`strict_language_check=True`, see the Limitations section for why it's
opt-in), `no_script_signal`, `empty`.

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

`calibrated_similarity(gold, response, language, encoder_name=..., min_auc=0.7)`
checks semantic closeness to a gold reference, using a threshold
calibrated per (encoder, language) instead of an uncalibrated 0.5 --
see the table below and the Limitations section for what "calibrated"
means here and its own limits. Needs a gold reference, unlike
`script_adherence`. Requires the `similarity` extra (sentence-
transformers, transformers, torch -- not installed by plain
`pip install vindex`).

**SAFE DEFAULT (v0.4.0):** only 3 of the 15 shipped (encoder,
language) cells have real, threshold-independent AUC >= 0.75 (see the
table below); several others -- most sharply multilingual-e5-base/hi
at real AUC exactly 0.500, chance -- look fine on in-sample accuracy
while having no real discrimination power. By default,
`calibrated_similarity` now returns `label="low_discrimination",
passed=False` for any cell whose real AUC is below `min_auc` (default
0.7), instead of a clean `similar`/`dissimilar` verdict that hides the
fact that particular cell can't actually tell correct from wrong.
Pass `min_auc=0.0` to disable this and trust the calibrated threshold
alone -- only do this after checking the AUC table below for the
specific cell you're using.

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
skips the LLM call entirely -- including the API key requirement
(fixed in v0.4.0: this used to construct and validate a judge before
checking for an exact match, so a caller with no `GROQ_API_KEY` got a
`ValueError` even when the answer matched `gold` exactly). The match
is exact at the word level after whitespace normalization, but
case-sensitive -- `"Same Answer"` vs `"same answer"` does not qualify
for the free short-circuit and falls through to a real judge call.

Conservative by design: `passed` is only `True` at high judge-reported
confidence AND a normalized score >= 0.8 -- since the rubric is 1-5,
that means only a raw score of 5 passes; a raw 4 at high confidence is
still `label="flagged"`, not a pass. Requires the `judge` extra (the
`groq` SDK) and a `GROQ_API_KEY`. See the Limitations section for what
this metric does not (yet) do.

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
in `experiments/scripts/validate_vindex_port.py` -- run it yourself
from a clone of the repo (not from a `pip install`'d copy: `experiments/`
is deliberately not shipped in the package, see `experiments/README.md`):

```bash
git clone https://github.com/dripston/vindex
cd vindex
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

## The argument for calibrated_similarity -- and where that argument runs out

At the naive default of 0.5 cosine similarity, 11 of 15 (encoder,
language) combinations in this package's own calibration data score at
or below 0.5 accuracy for correct-vs-wrong discrimination -- a coin
flip does as well. Calibrated per (encoder, language), 10 of 15 clear
0.6, and the best cell reaches 0.90.

**But "calibrated accuracy" is the accuracy of a threshold chosen, by
argmax sweep, on the SAME 10-19 points it's then scored on, with no
holdout.** That number can only go up or stay flat relative to 0.5 --
it is not independent evidence the encoder can actually discriminate.
The column that IS threshold-independent is ROC AUC (correct vs
wrong-hard, from `experiments/results_clean/discrimination_summary.csv`'s
`roc_auc_hard` column, `english_gold`/`full_sentence` slice -- the same
slice `experiments/README.md`'s Milestone 3 section reports).

Generated by `python experiments/scripts/generate_auc_table.py`, from
the CSV, not hand-transcribed -- after the first version of this table
was a hand-transcription of the wrong column (see CHANGELOG's v0.2.2
entry: every "AUC" value was actually a copy of the threshold column,
which happened to look plausible enough that neither the wrong ranking
it produced, nor the wrong conclusions built on it below, were caught
before publishing). Re-run the script before editing this table by hand
again.

| encoder                          | language | AUC (threshold-free) | accuracy @ 0.5 | calibrated threshold | accuracy @ calibrated |
|-----------------------------------|----------|----------------------:|----------------:|-----------:|------------------------:|
| paraphrase-multilingual-mpnet-v2   | en       | **0.930**             | 0.500           | 0.818      | 0.850                   |
| multilingual-e5-base               | en       | **0.860**             | 0.500           | 0.878      | 0.900                   |
| all-MiniLM-L6-v2                   | hinglish | **0.750**             | 0.600           | 0.351      | 0.750                   |
| LaBSE                              | hinglish | 0.730                 | 0.650           | 0.423      | 0.750                   |
| paraphrase-multilingual-mpnet-v2   | hinglish | 0.730                 | 0.700           | 0.412      | 0.800                   |
| paraphrase-multilingual-mpnet-v2   | hi       | 0.722                 | 0.474           | 0.871      | 0.737                   |
| multilingual-e5-base               | hinglish | 0.680                 | 0.500           | 0.845      | 0.750                   |
| all-MiniLM-L6-v2                   | en       | 0.600                 | 0.500           | 0.863      | 0.650                   |
| all-MiniLM-L6-v2                   | hi       | 0.589                 | 0.526           | 0.072      | 0.632                   |
| multilingual-e5-base               | hi       | 0.500 ⚠️               | 0.474           | 0.880      | 0.632                   |
| muril-base-cased                   | hinglish | 0.440 ⚠️               | 0.500           | 0.991      | 0.600                   |
| LaBSE                              | hi       | 0.167 ⚠️               | 0.474           | 0.867      | 0.526                   |
| muril-base-cased                   | hi       | 0.111 ⚠️               | 0.474           | 0.995      | 0.526                   |
| LaBSE                              | en       | 0.070 ⚠️               | 0.500           | 0.553      | 0.500                   |
| muril-base-cased                   | en       | 0.000 ⚠️               | 0.500           | 0.996      | 0.500                   |

⚠️ = AUC at or below chance, or inverted (worse than a coin flip) --
even where "accuracy @ calibrated" looks fine or good. Bold AUC = the
3 cells with real, strong discrimination (>=0.75).

**Read the ⚠️ rows carefully, they are not what the accuracy column
implies:**
- **multilingual-e5-base/hi, AUC 0.500**: exactly chance. This is the
  single highest "accuracy @ calibrated" cell in the *previous* version
  of this table under the wrong column, and it has no discrimination
  power at all -- the calibrated threshold routes around a coin flip,
  it doesn't reflect real signal. Don't reach for e5-base on Hindi
  expecting the English-language result to carry over.
- **muril-base-cased, all three languages, AUC 0.000-0.440**: MuRIL's
  raw scores are ~0.99 on nearly everything regardless of correctness
  (see the MuRIL warning below), and here that shows up as AUC at or
  near 0 -- on `en` and `hi` specifically, MuRIL ranks *wrong* answers
  above correct ones more often than not. `calibrated_similarity`
  raises loudly by default for MuRIL specifically because of this.
- **LaBSE/en, AUC 0.070**: strongly inverted, not "barely above
  chance" -- wrong answers score higher than correct ones on this
  sample far more often than the reverse. The calibrated threshold
  (0.553) exists only because argmax will always find *some* cutpoint;
  it does not mean the encoder discriminates.
- **LaBSE/hi, AUC 0.167**: same inversion pattern as LaBSE/en, less
  extreme. Do not read "accuracy @ calibrated = 0.526" as usable Hindi
  signal from this encoder.

**The honest summary**: of these 15 cells, 9 have AUC above 0.5, and
of those, only 3 are strong (>=0.75: mpnet-v2/en at 0.930,
e5-base/en at 0.860, all-MiniLM-L6-v2/hinglish at 0.750). 6 cells have
AUC at or below chance -- including e5-base/hi, this package's
best-looking calibrated-accuracy cell for Hindi under the old (wrong)
column. Mean AUC across all 15 cells is 0.525, barely above a coin
flip. A threshold fitted on 10-19 in-sample points and then evaluated
on the same points will always look better than 0.5 by construction --
it is not, by itself, evidence the encoder works. Use `calibrate()` on
your own held-out data, and check the AUC (or your own discrimination
metric) before trusting any cell, shipped or self-calibrated.

Full methodology, the "13 of 15" vs "11 of 15" note, and how to
reproduce this table: `experiments/README.md`'s Milestone 3 section.

## Limitations

- **9 scripts recognized, nothing else.** Devanagari, Kannada, Tamil,
  Telugu, Bengali, Gujarati, Malayalam, Odia, Gurmukhi, plus Latin.
  Anything else (Cyrillic, CJK, emoji, ASCII digits, punctuation-only
  text) has no script bucket of its own and falls through to
  `classify()`'s `"mixed"` label -- this is a real gap, not a rare edge
  case, if your data has other scripts in it. `script_adherence` used
  to then also count that fallback as a PASS for romanized/code-mixed
  prompts (an emoji-only or CJK-only response scored `passed=True`) --
  fixed: a response with no alphabetic content in any recognized script
  now scores `label="no_script_signal", passed=False` instead, distinct
  from genuine code-mixing (which always has real Latin or Indic
  letters). The underlying script-recognition gap above is unchanged;
  only the silent pass on top of it is fixed. **Indic digits are
  different from ASCII digits here, and were a separate real bug**
  (found by an independent outside review, fixed after the
  no_script_signal label shipped): each Indic script's own decimal
  digits live inside that script's own Unicode block (e.g. Devanagari
  ०-९ are in the same block as the letters), so a response that is
  purely Devanagari digits ("१४०००००००००") was NOT caught by the
  no_script_signal fix above -- it has real characters in the
  Devanagari block, so `classify()` confidently called it
  `"devanagari"`, the same verdict as a real Devanagari sentence. Fixed
  by excluding decimal-digit (Unicode category Nd) and danda
  characters when deciding whether a script has real letter content;
  a real sentence that happens to contain Indic digits is unaffected.
- **`language_mismatch` detection is a v0 heuristic, and OFF BY
  DEFAULT since v0.4.0** (`strict_language_check=False`). It checks
  for 14 hand-picked Hindi function words (`hai`, `hain`, `kya`,
  `nahi`, ...) in Romanized text. No transliteration-variant coverage,
  no verb conjugations, no other Romanized Indic languages, not
  ML-based. One matching word is treated as a signal, not proof -- and
  one incidental match in the PROMPT is enough to flip the whole
  prompt's bucket and hard-fail a genuinely correct English response:
  `"Who directed Se7en?"` contains `"se"` only because the digit
  splits the word, and `"What is the ka in Egyptian belief?"` contains
  `"ka"` as an ordinary English word -- both wrongly trigger
  `language_mismatch` if `strict_language_check=True`. A stricter
  "require 2+ matches" rule was tried and reverted: it also breaks
  short, genuine Hinglish questions, including this README's own
  `"Mumbai kahan hai?"` example above, which has exactly one function
  word. Because the false-positive rate can't be fixed by a threshold,
  it's an opt-in strict mode rather than the default -- pass
  `strict_language_check=True` only after verifying this heuristic
  doesn't false-positive on your own data.
- **Devanagari's danda (।) is shared punctuation.** It lives in the
  Devanagari Unicode block but is reused as a sentence-ending mark in
  Bengali, Odia, Gurmukhi, and others, so a couple of stray
  `devanagari_chars` can show up in a purely non-Devanagari sentence.
  Documented in `vindex/script.py`; doesn't change classification output
  in practice, since real sentences have far more dominant-script
  characters than stray punctuation. It DID change output for a
  degenerate response that is purely a lone danda and nothing else
  (e.g. a truncated generation) -- `classify("।")` confidently returns
  `"devanagari"` off one punctuation mark. Fixed in `script_adherence`
  (`script.is_only_danda_punctuation`): such a response now scores
  `no_script_signal`, not a confident script match. `classify()` itself
  is unchanged, since this is scoped to the one degenerate case, not a
  change to real-sentence classification.
- **`script_normalized_match` (a deterministic, transliteration-aware
  answer comparison) is still not shipped as public API.** Its pieces
  (`vindex.normalize`, `vindex.match`, `vindex.transliterate`) exist
  internally -- see the two limitations below for what they can and
  cannot do. `indic_judge` (below) covers factual-correctness checking
  in the meantime, but as an LLM judge, not a deterministic one.
  `vindex.normalize`'s punctuation-stripping used to also strip a
  numeric sign and decimal point as generic punctuation, so
  `exact_match_score("-5", "5")` and `exact_match_score("100.5",
  "1005")` both returned a false 1.0 (a sign flip and a 10x magnitude
  error reported as exact matches) -- fixed: a `-` directly before a
  digit and a `.`/`,` directly between two digits are now preserved.
  Fixed before this comparator ever shipped publicly, but flagged here
  since the underlying primitives are already used by this project's
  own tests.
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
  result's `reason` and `detail["calibrated"]`. More specifically: of
  the 15 shipped cells, 9 have AUC above chance (measured by
  threshold-independent ROC AUC), and only 3 of those are strong
  (>=0.75). 6 cells -- including e5-base/hi, whose calibrated accuracy
  (0.632) looks like one of the better Hindi results -- have AUC at or
  below chance (mean AUC across all 15 is 0.525). A calibrated-per-cell
  accuracy number is not, by itself, evidence a cell has real signal;
  see the AUC column in "The argument for calibrated_similarity" above
  before trusting any shipped threshold. **Since v0.4.0, this is
  enforced, not just documented**: `min_auc=0.7` is the default, so a
  cell below that (12 of 15, including e5-base/hi) returns
  `label="low_discrimination", passed=False` instead of a clean
  similar/dissimilar verdict. Pass `min_auc=0.0` to opt back into
  trusting the calibrated threshold alone.
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
- **`indic_judge` vs an English-rubric baseline (Milestone 5.8): the
  English rubric agreed with humans slightly MORE, not less, on this
  62-trace set.** 93.5% (58/62) for an English-rubric baseline vs
  90.3% (56/62) for `indic_judge`'s Hindi rubric, same traces, same
  human grades. This is the opposite of the motivating hypothesis
  (Phase 0's समुद्र तल mistranslation finding), and it is reported
  as-is rather than adjusted. The 4 disagreements are not the judge
  misreading Hindi -- inspecting them shows `indic_judge` being
  *more conservative about completeness*: 3 of 4 flag a substantively
  correct answer for omitting a secondary detail (e.g. not mentioning
  postage alongside the envelope and registration fee), which the
  English-rubric run and the human graders both accepted as correct.
  Only 1 of 4 goes the other way. Small sample (62 traces, 4
  disagreements) -- not strong evidence the Hindi rubric is worse in
  general, but it is real evidence that "Hindi rubric > English
  rubric" was an assumption, not yet a demonstrated result, on this
  data. See `experiments/README.md` for the full breakdown and the
  raw per-trace reasoning.
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
  detect this in general. This check is narrower than "a fine-tune
  with a deliberately unrelated name evades it" might suggest: it only
  strips a trailing size token (`-120b`), so a provider prefix
  (`openai/gpt-oss-120b` vs `groq/gpt-oss-120b`) or an ordinary suffix
  (`-instruct`, `-turbo`) already evades it too -- see
  `vindex.judge._family`'s docstring for exactly what it does and
  doesn't catch. Self-enhancement bias is documented, not theoretical
  -- verify your own judge/answering-model pairing; don't rely on this
  check to catch it for you.
- **The align-then-judge cost optimization (Milestone 5.4) only
  applies in reference-based mode.** Reference-free mode (the default)
  has nothing to diff against by definition, so every reference-free
  call is a real LLM call -- there is no free shortcut for the
  recommended mode. This is inherent to reference-free judging, not a
  missed optimization.
- **`check_trace`'s dictionary has 69 entries, hand-reviewed, not
  exhaustive.** `data/trap_words/hindiwic_inventory.csv` (59
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
  than forced into a weak pair. A 6th, उत्तर (north/answer), was
  removed after being filled in (found by an independent outside
  review): this library evaluates question-answering, and उत्तर
  meaning "answer" is correct in that context far more often than
  "north" -- the reading_a/reading_b assignment was backwards for the
  library's own primary use case, and no single assignment is right
  for both senses, so it was removed rather than shipped either way.
  Grows from user reports going forward, per Milestone 6.2.
  **A further, separate limitation, also found by that review: every
  gloss in the dictionary is ASCII English.** `judge_rubric.py`
  instructs the judge to reason in Hindi and not translate to English
  while thinking -- so a Hindi-language reasoning trace can never
  trigger `check_trace` at all, regardless of whether it misread
  anything, since there is no English gloss to match against. This
  means `check_trace`'s effective N is 0 on the trace language
  `indic_judge`'s own recommended (Hindi-rubric) configuration
  produces, even though `detail["dictionary_size"]` reports 69.
  `check_trace`/`indic_judge` do not currently compose safely for this
  reason; treat any `check_trace` result on a Hindi-language trace as
  uninformative, not as confirmation nothing was misread.
- **`check_trace` only catches mistranslation of a term already in the
  dictionary.** By design (Milestone 6.2's honest claim: "detects
  mistranslation of N known ambiguous terms," never "detects
  mistranslation"). A term not yet in the dictionary, or an ambiguity
  that isn't a fixed word-pair at all (e.g. a dropped negation, or an
  idiom translated literally), needs `check_trace_llm_fallback`
  instead -- and even that is scoped to the segments that don't align,
  not a general mistranslation detector.
- **`check_trace`'s "mentions both readings" check is a literal
  substring match, not a paraphrase check -- it can be wrong in both
  directions.** A trace that correctly reasons through an ambiguity in
  different words than the dictionary's exact `reading_a` gloss (e.g.
  "it's about the surface" instead of the literal string "sea level")
  is flagged as a misread anyway -- a false positive on genuinely
  correct reasoning. A trace that uses a synonym for the wrong reading
  not in the dictionary (e.g. "ocean bottom" instead of "sea floor") is
  invisible to the check -- a false negative. Both are inherent to
  matching without any NLP, not something the current mechanism claims
  to solve; see `judge_trace_check.py`'s `_trace_says_wrong_reading`
  docstring and its two pinned regression tests.
- **`indic_judge` used to silently clamp an out-of-range judge score
  instead of treating it as a malformed response.** A score of e.g.
  100 (a judge misreading the "1 to 5" rubric, or a corrupted/
  adversarial response) was clamped into range and normalized to the
  maximum, producing `score=1.0` and a potential `passed=True` -- the
  package's most confident possible result, from a response that never
  actually followed the scoring contract. Fixed: any score outside 1-5
  now raises and surfaces as `label="judge_error"`, the same as every
  other malformed judge response.
- **`indic_judge` human-agreement study (Milestone 6.3): 62 traces (31
  trap words x correct/wrong answer), graded independently by two
  fluent Hindi speakers under a bias protocol**
  (`docs/annotation/BIAS_PROTOCOL.md`: ground truth committed before
  grading, blind sheet, frozen rubric, 30% holdout, no cross-visibility
  until both submitted). **This is a study of `indic_judge`'s verdict
  agreement with humans, NOT a precision/recall study of `check_trace`**
  -- an earlier version of this bullet was headed "`check_trace`/
  `indic_judge`", which was wrong (found by an independent outside
  review): `BIAS_PROTOCOL.md`'s own comparison is explicitly "each
  grader's verdict vs. `indic_judge`'s own [verdict]"; `check_trace`
  was never part of this comparison, despite using the same 62 trap-
  word-derived traces as source material. No precision/recall study of
  `check_trace` itself has been run -- see `judge_trace_check.py`'s
  module docstring's MILESTONE 6.3 section, which is still accurate:
  that work has not happened. Result of the actual `indic_judge` study:
  **90.3% agreement (56/62) between human graders and `indic_judge`'s
  own verdict**, 100% inter-annotator agreement between the two
  humans, no meaningful gap between the tuning set (88.4%) and the
  untouched 30% holdout (94.7%) -- no sign of grading drift. Disclosure
  required by the protocol: one of the two graders is the project
  owner, who also built the metric being evaluated; the second grader
  is an independent fluent Hindi speaker with no stake in the result.
  Do not cite the 90.3% figure without this disclosure next to it, and
  do not cite it as evidence about `check_trace`. All disagreement was
  on the same small set of traces the judge itself flags as
  conservative-by-design (a correct answer marked "flagged" over an
  edge-case nuance, not a missed mistranslation) -- see
  `docs/annotation/BIAS_PROTOCOL.md` for the full breakdown and raw
  numbers.
