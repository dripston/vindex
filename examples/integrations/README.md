# Integrations

Drop-in adapters that run `vindex.script_adherence` inside the eval tool you
already use. No reference answer, no LLM call, microseconds per case.

| File | Tool |
|------|------|
| `deepeval_vindex.py` | [DeepEval](https://github.com/confident-ai/deepeval) — `ScriptAdherenceMetric` for `evaluate()` / `assert_test()` |
| `promptfoo_vindex.py`, `promptfoo_vindex.yaml` | [promptfoo](https://www.promptfoo.dev/) — Python assertion |
| `ragas_vindex.py` | [Ragas](https://github.com/vibrantlabsai/ragas) — `SingleTurnMetric` |

## DeepEval

```bash
pip install vindex deepeval
python examples/integrations/deepeval_vindex.py
```

## promptfoo

```bash
pip install vindex && npm install -g promptfoo
cd examples/integrations
promptfoo eval -c promptfoo_vindex.yaml
```

`promptfoo_fixture_provider.py` is a deterministic stand-in provider used only
by the example config.

## Ragas

```bash
pip install vindex ragas
python examples/integrations/ragas_vindex.py
```

## Other metrics

Each adapter calls `vindex.script_adherence(prompt, response)`. To wrap a
different check, call `vindex.calibrated_similarity(gold, response, language)`
or `vindex.indic_judge(question, answer)` in the same place.
