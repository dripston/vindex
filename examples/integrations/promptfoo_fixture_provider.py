"""
Deterministic stand-in provider for promptfoo_vindex.yaml's example --
NOT part of the vindex integration itself. Returns whatever
`canned_output` var the test case sets, so the example can demonstrate
vindex scoring different responses without calling a real model or API.
"""

from __future__ import annotations

from typing import Any


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    canned_output = context["vars"].get("canned_output", "")
    return {"output": canned_output}
