"""
JudgeModel interface and Groq-backed implementation (Milestone 5.6).

indic_judge (Milestone 5) needs an LLM call -- semantic correctness
with no reference answer cannot be done deterministically (see
judge.py's module docstring). This module defines a small,
provider-agnostic interface so the judge model can be swapped (5.7:
never judge a model with itself, judge competence on Indic is not
uniform -- these only matter if judges are actually swappable) and
ships one concrete implementation, Groq, matching every other script
in this repo (experiments/run_experiment.py, regenerate_dataset.py).

DETERMINISM DISCIPLINE (Milestone 5.6): temperature is fixed at 0.0
and is not a constructor parameter -- there is no legitimate reason
for a judge to be non-deterministic. Every JudgeModel implementation
must report `model_id` as a concrete, versioned string (e.g.
"openai/gpt-oss-120b" via Groq, not "latest" or an unpinned alias) and
every judge call result records it (see judge.py's MetricResult
detail payload) -- this repo's own data showed gpt-oss-20b reworded
output roughly a third of the time even at temperature 0 (see
experiments/README.md and data/NOTICE.md), so the model identity must
be recorded per result, not just per run: a silently-updated judge
model makes every historical score incomparable, and Sarvam's
published work names this same risk.

DEPENDENCIES: groq is NOT a core vindex dependency -- install via
`pip install vindex[judge]`. This module lazy-imports it so plain
`import vindex` never requires it, same pattern as vindex.encoder for
the similarity extra.
"""

from __future__ import annotations

import os
from typing import Protocol


class JudgeModel(Protocol):
    """Minimal interface indic_judge needs from an LLM judge backend.

    model_id must be a concrete, version-pinned string -- see module
    docstring's Determinism Discipline section. call() must be
    deterministic (temperature 0 or equivalent) and return the raw
    text response; indic_judge handles JSON parsing itself so this
    interface stays provider-agnostic.
    """

    @property
    def model_id(self) -> str: ...

    def call(self, prompt: str) -> str: ...


class GroqJudge:
    """Groq-backed JudgeModel. Default model_id matches the model used
    in this project's own Phase 0 judge-quality experiments
    (experiments/run_experiment.py's JUDGE_MODEL) -- openai/gpt-oss-120b,
    not the smaller openai/gpt-oss-20b used for answer generation (see
    judge.py's self-enhancement-bias guidance: the judge model should
    differ from, and ideally be stronger than, the model being judged).

    Requires GROQ_API_KEY in the environment (matches every other
    script in this repo's convention -- see experiments/run_experiment.py).
    """

    DEFAULT_MODEL_ID = "openai/gpt-oss-120b"

    def __init__(self, model_id: str = "", api_key: str = "") -> None:
        self._model_id = model_id or self.DEFAULT_MODEL_ID
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "GroqJudge needs an API key: pass api_key= or set the "
                "GROQ_API_KEY environment variable."
            )
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self) -> object:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
        return self._client

    def call(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        content: str = response.choices[0].message.content
        return content
