"""Judge model interface and provider implementations for indic_judge.

:class:`JudgeModel` is a small, provider-agnostic protocol so the judge
behind :func:`vindex.indic_judge` can be swapped. Shipped
implementations:

- :class:`GroqJudge` (default; ``pip install vindex[judge]``,
  ``GROQ_API_KEY``)
- :class:`OpenAIJudge` (``pip install openai``, ``OPENAI_API_KEY``)
- :class:`AnthropicJudge` (``pip install anthropic``,
  ``ANTHROPIC_API_KEY``)
- :class:`LiteLLMJudge` (``pip install litellm``), which covers Gemini,
  Azure OpenAI, Bedrock, Ollama, OpenRouter, and roughly a hundred other
  providers through LiteLLM's provider-prefixed model strings.

Each class imports its SDK lazily, so ``import vindex`` never requires
any of them. All implementations call the model at temperature 0, which
is not configurable, and expose a concrete ``model_id`` that is recorded
in every result's ``detail``. Even at temperature 0 some models reword
their output between calls, so pin a specific model version and compare
scores only across results with the same ``judge_model_id``.
"""

from __future__ import annotations

import os
from typing import Protocol


class JudgeModel(Protocol):
    """Minimal interface indic_judge needs from an LLM judge backend.

    ``model_id`` must be a concrete, version-pinned identifier (not
    ``"latest"`` or another floating alias). ``call()`` must be
    deterministic (temperature 0 or equivalent) and return the raw text
    response; indic_judge parses the JSON itself.
    """

    @property
    def model_id(self) -> str: ...

    def call(self, prompt: str) -> str: ...


class GroqJudge:
    """Groq-backed judge model (the default).

    The default model is ``openai/gpt-oss-120b``, the model the rubric
    was developed and validated against. Use a judge that differs from,
    and ideally is stronger than, the model being judged.

    Args:
        model_id: Groq model id. Defaults to :attr:`DEFAULT_MODEL_ID`.
        api_key: API key. Defaults to ``$GROQ_API_KEY``.

    Raises:
        ValueError: If no API key is available.
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


class OpenAIJudge:
    """OpenAI-backed judge model.

    Requires the ``openai`` SDK (``pip install openai``).

    Args:
        model_id: OpenAI model id. Defaults to :attr:`DEFAULT_MODEL_ID`.
        api_key: API key. Defaults to ``$OPENAI_API_KEY``.

    Raises:
        ValueError: If no API key is available.
    """

    DEFAULT_MODEL_ID = "gpt-4o"

    def __init__(self, model_id: str = "", api_key: str = "") -> None:
        self._model_id = model_id or self.DEFAULT_MODEL_ID
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAIJudge needs an API key: pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self) -> object:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
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


class AnthropicJudge:
    """Anthropic-backed judge model.

    Requires the ``anthropic`` SDK (``pip install anthropic``). Anthropic
    has no JSON response mode; the rubric asks for JSON and indic_judge
    extracts the first valid JSON object from the response.

    Args:
        model_id: Anthropic model id. Defaults to :attr:`DEFAULT_MODEL_ID`.
        api_key: API key. Defaults to ``$ANTHROPIC_API_KEY``.

    Raises:
        ValueError: If no API key is available.
    """

    DEFAULT_MODEL_ID = "claude-sonnet-4-5"

    def __init__(self, model_id: str = "", api_key: str = "") -> None:
        self._model_id = model_id or self.DEFAULT_MODEL_ID
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "AnthropicJudge needs an API key: pass api_key= or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        self._client: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_client(self) -> object:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def call(self, prompt: str) -> str:
        client = self._get_client()
        response = client.messages.create(  # type: ignore[attr-defined]
            model=self._model_id,
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content: str = response.content[0].text
        return content


class LiteLLMJudge:
    """LiteLLM-backed judge model covering roughly a hundred providers.

    Requires ``litellm`` (``pip install litellm``). Select the provider
    with LiteLLM's model-string prefix, for example::

        LiteLLMJudge("gemini/gemini-1.5-pro")
        LiteLLMJudge("ollama/llama3")
        LiteLLMJudge("azure/my-deployment-name")

    Credentials are read by LiteLLM from each provider's own environment
    variables (e.g. ``GEMINI_API_KEY``); see
    https://docs.litellm.ai/docs/providers.

    Args:
        model_id: Provider-prefixed LiteLLM model string. Required; there
            is no default.

    Raises:
        ValueError: If ``model_id`` is empty.
    """

    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ValueError(
                "LiteLLMJudge needs an explicit model_id (e.g. "
                "'gemini/gemini-1.5-pro') -- unlike GroqJudge/OpenAIJudge/"
                "AnthropicJudge, there is no single sensible default "
                "across a hundred different providers."
            )
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def call(self, prompt: str) -> str:
        import litellm

        response = litellm.completion(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        content: str = response.choices[0].message.content
        return content
