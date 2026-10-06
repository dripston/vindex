<div align="center">

<img src="site/assets/logo.svg" width="72" alt="vindex logo">

# vindex

**Evals for Indian-language LLM apps.**

Catch the failures English-only eval tools miss in Hindi, Hinglish and eight other Indian scripts.

[![PyPI](https://img.shields.io/pypi/v/vindex?color=e8890c)](https://pypi.org/project/vindex/)
[![CI](https://github.com/dripston/vindex/actions/workflows/ci.yml/badge.svg)](https://github.com/dripston/vindex/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/vindex)](https://pypi.org/project/vindex/)
[![License: MIT](https://img.shields.io/badge/license-MIT-1b1511.svg)](LICENSE)

[Website](https://dripston.github.io/vindex/) · [Docs](https://dripston.github.io/vindex/docs.html) · [Playground](https://dripston.github.io/vindex/#playground) · [Changelog](CHANGELOG.md)

</div>

---

```console
$ pip install vindex
$ vindex run evals/support_bot.jsonl

  vindex 0.6.0  support_bot.jsonl · 240 cases

  ✗ script_adherence       █████████████████░░░  86.7%  208/240 passed
  ✓ check_trace            ████████████████████ 100.0%  64/64 passed

  Failures

  ● refund-017 · script_adherence · script_mismatch
    prompt   Mera refund kab tak aayega?
    response आपका रिफंड 5-7 कार्यदिवसों में आ जाएगा।
    reason   prompt is code-mixed; response in devanagari, not Roman script.
```

## Why

Your users type *"Mumbai kahan hai?"* Your model replies in fluent Devanagari. Your reviewer reads Hindi, so it looks fine. The user, who typed in Roman script because that's what they read, gets an answer they can't use.

Told to *"reply in the user's language"*, a popular open-weight model did this to **4 of every 5** Hinglish questions. A one-line prompt change fixed it completely, but nobody would have known without a check that runs on every response.

vindex is that check, plus three more for the other ways Indic output goes wrong.

## The checks

| Check | Catches | Cost |
|---|---|---|
| [`script_adherence`](https://dripston.github.io/vindex/docs.html#script-adherence) | Replies in the wrong script: Devanagari to a Hinglish user, English to a Tamil one, emoji-only non-answers. | Free · no LLM · no reference |
| [`check_trace`](https://dripston.github.io/vindex/docs.html#check-trace) | An LLM judge misreading ambiguous Hindi: समुद्र तल as "sea floor", कल as "yesterday". | Free · deterministic |
| [`calibrated_similarity`](https://dripston.github.io/vindex/docs.html#similarity) | Wrong answers, using per-encoder, per-language thresholds. Refuses to score where the encoder can't discriminate. | Local encoder · needs gold |
| [`indic_judge`](https://dripston.github.io/vindex/docs.html#judge) | Incorrect answers, judged with a Hindi rubric that knows Romanized Hindi isn't an error. | LLM · reference-free |

Every check returns the same `MetricResult`: `score`, `passed`, a stable `label`, a readable `reason`, and the raw `detail`.

## Quickstart

```python
from vindex import script_adherence, check_trace

script_adherence("Mumbai kahan hai?", "मुंबई महाराष्ट्र में है।").label
# 'script_mismatch'

check_trace(
    source="समुद्र तल पर पानी किस तापमान पर उबलता है?",
    trace="The question asks for the boiling point at the sea floor...",
).label
# 'misread_detected'
```

Add an LLM judge from any provider:

```bash
pip install "vindex[judge]"          # Groq (default)
pip install openai                   # or anthropic, or litellm
```

```python
from vindex import indic_judge
from vindex.judge_model import OpenAIJudge

result = indic_judge(question, answer, judge=OpenAIJudge("gpt-4o"))
result.passed, result.detail["judge_model_id"]
```

## CLI

```bash
vindex run cases.jsonl                                 # script_adherence (+ check_trace on rows with a trace)
vindex run cases.jsonl -m script,judge --judge anthropic
vindex run cases.csv -m similarity -l hi --fail-under 0.9
vindex run cases.jsonl -f json -o report.json
vindex check "Mumbai kahan hai?" "मुंबई महाराष्ट्र में है।"
vindex detect "Kal office aa raha hai kya?"
```

Reads JSONL, JSON or CSV. Column names from DeepEval, promptfoo and Ragas exports (`input`, `actual_output`, `expected_output`, …) work as-is. See the [dataset format](https://dripston.github.io/vindex/docs.html#datasets).

### In CI

```yaml
- run: pip install vindex
- run: vindex run evals/cases.jsonl --fail-under 0.95
```

Exits 1 when a pass rate drops below the bar, and writes a markdown report to the GitHub Actions job summary.

## Integrations

Drop-in adapters for [DeepEval, promptfoo and Ragas](examples/integrations/). Or call vindex from pytest:

```python
from vindex import script_adherence

def test_bot_replies_in_users_script(bot):
    prompt = "Mera refund kab aayega?"
    result = script_adherence(prompt, bot(prompt))
    assert result.passed, result.reason
```

## Supported scripts

Devanagari · Bengali · Gurmukhi · Gujarati · Odia · Tamil · Telugu · Kannada · Malayalam · Latin (incl. Hinglish)

Similarity thresholds ship for English, Hindi and Hinglish; [calibrate on your own data](https://dripston.github.io/vindex/docs.html#calibrate) for anything else. The judge rubric is written for Hindi. Need another language? [Open an issue](https://github.com/dripston/vindex/issues).

## Development

```bash
git clone https://github.com/dripston/vindex && cd vindex
pip install -e ".[dev,similarity,judge]"
ruff check . && mypy src && pytest
```

The website lives in [`site/`](site/) and is deployed to GitHub Pages on every push to `master`.

## License

MIT. Bundled data attributions are in [`src/vindex/data/NOTICE.md`](src/vindex/data/NOTICE.md).
