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
  on the roadmap, not shipped yet.
