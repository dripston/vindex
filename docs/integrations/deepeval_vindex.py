"""
vindex as a DeepEval custom metric (Milestone 4.1/4.2).

Wraps vindex.script_adherence -- no reference answer needed, checks
whether a response came back in the script/language the prompt used.
Written in DeepEval's own idiom (subclass BaseMetric, implement
measure/a_measure, set score/reason/success) so it drops into an
existing DeepEval test suite unchanged.

Install:
    pip install vindex deepeval

Run this file directly for a smoke test, or import ScriptAdherenceMetric
into your own DeepEval test suite (see the __main__ block below for the
idiomatic usage: assert_test / evaluate).
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from vindex import script_adherence


class ScriptAdherenceMetric(BaseMetric):
    """DeepEval metric: did the response come back in the script and
    language the prompt used? No LLM call, no reference answer --
    deterministic, from vindex.script_adherence.

    threshold is unused in the pass/fail sense (script_adherence's own
    `passed` decides success) but kept for DeepEval's is_successful()
    plumbing and to satisfy BaseMetric's interface.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.success = None

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        result = script_adherence(test_case.input, test_case.actual_output)
        self.score = result.score
        self.reason = f"[{result.label}] {result.reason}"
        self.success = result.passed
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        # script_adherence is synchronous and cheap (no LLM call), so
        # async mode just calls the sync path directly.
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Script Adherence (vindex)"


if __name__ == "__main__":
    # Idiomatic DeepEval usage: build an LLMTestCase, run the metric,
    # inspect score/reason/success the same way any other DeepEval
    # metric works.
    test_case = LLMTestCase(
        input="Mumbai kahan hai?",
        actual_output="Mumbai is the capital of Maharashtra.",
    )

    metric = ScriptAdherenceMetric()
    metric.measure(test_case)

    print(f"score:   {metric.score}")
    print(f"reason:  {metric.reason}")
    print(f"success: {metric.is_successful()}")

    # Wire into DeepEval's own test runner:
    #
    #   from deepeval import assert_test
    #   def test_hinglish_reply():
    #       assert_test(test_case, [ScriptAdherenceMetric()])
    #
    # or batch-evaluate a dataset:
    #
    #   from deepeval import evaluate
    #   evaluate([test_case, ...], [ScriptAdherenceMetric()])
