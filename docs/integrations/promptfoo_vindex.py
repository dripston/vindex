"""
vindex as a promptfoo Python assertion (Milestone 4.1/4.2).

promptfoo's `python` assertion type calls a Python function with the
LLM's output plus a context object, and expects back either a bool, a
float score, or a dict with at least {"pass": bool, "score": float,
"reason": str} -- see promptfoo_vindex.yaml alongside this file for the
config that wires this function in as `file://./promptfoo_vindex.py`.

vindex.script_adherence needs the ORIGINAL PROMPT, not just the output
-- promptfoo exposes it via context.vars (whatever variable your test
case's prompt was built from), not as a separate function argument, so
this reads it out of vars rather than context.prompt (which is the
rendered final prompt text, not necessarily the same string
script_adherence should be comparing script against if you're using a
prompt template with a system message).

Install:
    pip install vindex
    npm install -g promptfoo   # or: npx promptfoo eval
"""

from __future__ import annotations

from typing import Any

from vindex import script_adherence


def get_assert(output: str, context: dict[str, Any]) -> dict[str, Any]:
    """promptfoo's Python-assertion entrypoint.

    output  : the LLM's response text, from promptfoo.
    context : dict with "vars" (the test case's template variables) and
              "prompt" (the fully rendered prompt) -- see
              https://www.promptfoo.dev/docs/configuration/expected-outputs/python/
    """
    prompt = context["vars"].get("prompt") or context.get("prompt", "")
    result = script_adherence(prompt, output)

    return {
        "pass": result.passed,
        "score": result.score,
        "reason": f"[{result.label}] {result.reason}",
    }
