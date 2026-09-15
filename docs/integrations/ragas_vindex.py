"""
vindex as a Ragas custom metric (Milestone 4.1/4.2).

Written in Ragas's own idiom for a deterministic (non-LLM) metric:
subclass SingleTurnMetric, declare _required_columns, implement
_single_turn_ascore and _ascore -- the exact shape Ragas's own built-in
ExactMatch/StringPresence metrics use (ragas/metrics/_string.py in the
ragas package itself). Matches script_adherence's two-argument
(prompt, response) shape onto Ragas's SingleTurnSample.user_input /
SingleTurnSample.response fields.

VERIFICATION NOTE: this file's shape was checked directly against
ragas 0.4.3's real source (ragas/metrics/base.py's SingleTurnMetric,
ragas/metrics/_string.py's ExactMatch, ragas/dataset_schema.py's
SingleTurnSample) downloaded via `pip download ragas --no-deps`. It
was NOT run end-to-end against an installed ragas: `pip install ragas`
fails in this environment because one of its transitive dependencies
(scikit-network) needs a C++ compiler (Microsoft Visual C++ Build
Tools) not present here. Stated plainly rather than silently presented
as tested like the other two adapters in this directory.

Install:
    pip install vindex ragas
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, SingleTurnMetric
from ragas.run_config import RunConfig

from vindex import script_adherence


@dataclass
class ScriptAdherence(SingleTurnMetric):
    """Ragas metric: did the response come back in the script and
    language the prompt used? No LLM call -- deterministic, from
    vindex.script_adherence.

    Like Ragas's own ExactMatch/StringPresence, this returns only the
    float score (0.0 or 1.0) through Ragas's scoring path -- Ragas's
    non-LLM metric interface does not carry a reason/label alongside
    the score. Call vindex.script_adherence directly if you need the
    label/reason (e.g. for debugging a failing case), the same way you
    would inspect script_adherence's own MetricResult outside Ragas.
    """

    name: str = "script_adherence"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )

    def init(self, run_config: RunConfig) -> None:
        pass

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: t.Any) -> float:
        result = script_adherence(sample.user_input, sample.response)
        return result.score

    async def _ascore(self, row: t.Dict[str, t.Any], callbacks: t.Any) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


if __name__ == "__main__":
    # Idiomatic Ragas usage: build a SingleTurnSample, score it
    # directly, or run it through Ragas's evaluate() over a Dataset the
    # same way any other Ragas metric works.
    import asyncio

    sample = SingleTurnSample(
        user_input="Mumbai kahan hai?",
        response="Mumbai is the capital of Maharashtra.",
    )

    metric = ScriptAdherence()
    score = asyncio.run(metric._single_turn_ascore(sample, callbacks=None))
    print(f"score: {score}")

    # Wire into ragas.evaluate() over a full dataset:
    #
    #   from ragas import evaluate, EvaluationDataset
    #   dataset = EvaluationDataset.from_list([
    #       {"user_input": "...", "response": "..."},
    #       ...
    #   ])
    #   results = evaluate(dataset, metrics=[ScriptAdherence()])
