"""
Tests for the DeepEval integration in docs/integrations/deepeval_vindex.py
(Milestone 4.1/4.2). Skipped if deepeval isn't installed.

The promptfoo integration (docs/integrations/promptfoo_vindex.py) is a
promptfoo-CLI Python-assertion entrypoint, not a Python library --
verified by actually running `promptfoo eval` against
promptfoo_vindex.yaml (see docs/integrations/README.md), not covered
here since it needs the Node-based promptfoo CLI, not a pip package.

The ragas integration (docs/integrations/ragas_vindex.py) is checked
against ragas's real source but not run end to end in this environment
-- see that file's own docstring and docs/integrations/README.md for
why (a transitive dependency needs a C++ compiler not present here) --
so it has no test here either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("deepeval")

_INTEGRATIONS_DIR = Path(__file__).parent.parent / "docs" / "integrations"
sys.path.insert(0, str(_INTEGRATIONS_DIR))

from deepeval.test_case import LLMTestCase  # noqa: E402
from deepeval_vindex import ScriptAdherenceMetric  # noqa: E402


def test_deepeval_adapter_matched_case() -> None:
    tc = LLMTestCase(
        input="Mumbai kahan hai?",
        actual_output="Mumbai Maharashtra ki rajdhani hai.",
    )
    metric = ScriptAdherenceMetric()
    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert "matched" in metric.reason


def test_deepeval_adapter_language_mismatch_off_by_default() -> None:
    # DEFAULT CHANGED (found by repeated independent outside review):
    # script_adherence's strict_language_check now defaults to False,
    # so this case (a Roman-script response to a Hinglish prompt)
    # passes on the script check alone, without also gating on the
    # 14-word Hinglish-detection heuristic's proven false-positive rate.
    tc = LLMTestCase(
        input="Mumbai kahan hai?",
        actual_output="Mumbai is the capital of Maharashtra.",
    )
    metric = ScriptAdherenceMetric()
    metric.measure(tc)

    assert metric.score == 1.0
    assert metric.is_successful() is True
    assert "matched" in metric.reason


def test_deepeval_adapter_language_mismatch_case_when_strict() -> None:
    tc = LLMTestCase(
        input="Mumbai kahan hai?",
        actual_output="Mumbai is the capital of Maharashtra.",
    )
    metric = ScriptAdherenceMetric(strict_language_check=True)
    metric.measure(tc)

    assert metric.score == 0.0
    assert metric.is_successful() is False
    assert "language_mismatch" in metric.reason


def test_deepeval_adapter_name_is_descriptive() -> None:
    metric = ScriptAdherenceMetric()
    assert metric.__name__ == "Script Adherence (vindex)"


@pytest.mark.asyncio
async def test_deepeval_adapter_async_matches_sync() -> None:
    tc = LLMTestCase(
        input="Mumbai kahan hai?",
        actual_output="Mumbai Maharashtra ki rajdhani hai.",
    )
    metric = ScriptAdherenceMetric()
    score = await metric.a_measure(tc)
    assert score == 1.0
