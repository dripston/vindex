# vindex integrations (Milestone 4.1/4.2)

Three ~50-line adapters, one per eval tool, each written in that
tool's own idiom -- not vindex's. All three wrap
`vindex.script_adherence`: no reference answer needed, deterministic,
no LLM call, so it drops into an existing suite without adding cost or
latency.

| file | tool | verified how |
|------|------|----------------|
| `deepeval_vindex.py` | [DeepEval](https://github.com/confident-ai/deepeval) | Ran for real: `python deepeval_vindex.py` (smoke test) and through DeepEval's own `evaluate()` runner end to end. |
| `promptfoo_vindex.py` + `promptfoo_vindex.yaml` | [promptfoo](https://www.promptfoo.dev/) | Ran for real: `promptfoo eval -c promptfoo_vindex.yaml` against the actual installed CLI (0.122.2), one passing case and one failing case, both confirmed correct. `promptfoo_fixture_provider.py` is a tiny deterministic stand-in provider for the example only -- not part of the integration. |
| `ragas_vindex.py` | [Ragas](https://github.com/vibrantlabsai/ragas) | Checked directly against ragas 0.4.3's real downloaded source (`SingleTurnMetric`, `SingleTurnSample`, and its own built-in `ExactMatch` metric as the reference pattern) -- **not run end to end**. `pip install ragas` fails in this environment: one of its dependencies, `scikit-network`, needs a C++ compiler this machine doesn't have, and newer scikit-network releases don't ship a prebuilt Windows wheel. Stated here plainly rather than silently presented as tested. |

## Quick start

```bash
pip install vindex
```

**DeepEval:**
```bash
pip install deepeval
python docs/integrations/deepeval_vindex.py
```

**promptfoo:**
```bash
npm install -g promptfoo
cd docs/integrations
promptfoo eval -c promptfoo_vindex.yaml
```

**Ragas:**
```bash
pip install ragas   # see the verification note above before relying on this
python docs/integrations/ragas_vindex.py
```

## Why script_adherence specifically

It needs no gold reference, no LLM judge call, and returns in
microseconds -- the cheapest possible thing to add to an existing eval
suite, and the metric behind this package's headline finding: the same
model, the same 30 questions, answered in Devanagari 4 times out of 5
under an ambiguous system prompt, fixed completely by a stricter one.
See the root `README.md` for that table.

`calibrated_similarity` (Milestone 3) is not wrapped here -- it needs a
gold reference and a loaded encoder, a different shape than the other
two tools' simpler assertion patterns. The same wrapping pattern in
each file above applies directly if you need it: call
`vindex.calibrated_similarity(gold, response, language)` instead of
`vindex.script_adherence(prompt, response)` inside `measure`/
`_single_turn_ascore`/`get_assert`.
