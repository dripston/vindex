"""Command-line interface.

    vindex check PROMPT RESPONSE        score one pair with script_adherence
    vindex detect TEXT                  show which script(s) a text is written in
    vindex run FILE [options]           evaluate a dataset (.jsonl, .json or .csv)

`vindex run` is built for CI: it prints a per-metric summary, lists every
failing case with its reason, and exits non-zero when any metric's pass rate
falls below `--fail-under` (default 1.0, i.e. every case must pass).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from vindex import __version__
from vindex.result import MetricResult

METRICS = ("script", "similarity", "judge", "trace")

# Column aliases accepted in dataset files, so exports from DeepEval,
# promptfoo, Ragas or a spreadsheet work without renaming.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt", "question", "input", "user_input", "query"),
    "response": ("response", "answer", "output", "actual_output", "completion"),
    "gold": ("gold", "expected", "expected_output", "reference", "ground_truth"),
    "trace": ("trace", "reasoning", "judge_reasoning"),
    "language": ("language", "lang"),
    "id": ("id", "case_id", "name"),
}


# --------------------------------------------------------------------------- #
# Terminal styling
# --------------------------------------------------------------------------- #


class Style:
    def __init__(self, stream: TextIO, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = (
                hasattr(stream, "isatty")
                and stream.isatty()
                and "NO_COLOR" not in os.environ
                and os.environ.get("TERM") != "dumb"
            )
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def accent(self, text: str) -> str:
        return self._wrap("38;5;208", text)


def _utf8(stream: TextIO) -> TextIO:
    """Make sure Indic text can be printed on consoles with a legacy code page."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    return stream


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    id: str
    prompt: str
    response: str
    gold: str = ""
    trace: str = ""
    language: str = ""


def _pick(row: dict[str, Any], key: str) -> str:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for alias in FIELD_ALIASES[key]:
        value = lowered.get(alias)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def load_cases(path: Path) -> list[Case]:
    """Read a .jsonl, .json (a list of objects) or .csv dataset."""
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]]
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("cases") or data.get("data") or data.get("tests") or []
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of objects")
        rows = data
    else:
        rows = []
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc

    cases = []
    for n, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {n} is not an object")
        cases.append(
            Case(
                id=_pick(row, "id") or str(n),
                prompt=_pick(row, "prompt"),
                response=_pick(row, "response"),
                gold=_pick(row, "gold"),
                trace=_pick(row, "trace"),
                language=_pick(row, "language"),
            )
        )
    return cases


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class MetricSummary:
    name: str
    results: list[tuple[Case, MetricResult]] = field(default_factory=list)
    skipped: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for _, r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 1.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for _, r in self.results) / self.total if self.total else 0.0


def make_judge(spec: str) -> Any:
    """Build a judge model from `provider` or `provider:model_id`."""
    from vindex import judge_model

    provider, _, model_id = spec.partition(":")
    provider = provider.strip().lower()
    if provider == "groq":
        return judge_model.GroqJudge(model_id=model_id) if model_id else judge_model.GroqJudge()
    if provider == "openai":
        return judge_model.OpenAIJudge(model_id=model_id)
    if provider == "anthropic":
        return judge_model.AnthropicJudge(model_id=model_id)
    if provider == "litellm":
        return judge_model.LiteLLMJudge(model_id=model_id)
    raise ValueError(
        f"unknown judge provider {provider!r}; use groq, openai, anthropic or litellm"
    )


def evaluate(
    cases: Sequence[Case],
    metrics: Sequence[str],
    *,
    language: str = "",
    strict_language_check: bool = False,
    judge_spec: str = "groq",
    encoder: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> list[MetricSummary]:
    summaries = {name: MetricSummary(name) for name in metrics}
    judge = make_judge(judge_spec) if "judge" in metrics else None

    for i, case in enumerate(cases, start=1):
        if "script" in summaries:
            from vindex import script_adherence

            summaries["script"].results.append(
                (case, script_adherence(case.prompt, case.response, strict_language_check))
            )
        if "similarity" in summaries:
            lang = case.language or language
            if case.gold and lang:
                from vindex import calibrated_similarity

                summaries["similarity"].results.append(
                    (case, calibrated_similarity(case.gold, case.response, lang, encoder))
                )
            else:
                summaries["similarity"].skipped += 1
        if "judge" in summaries:
            from vindex import indic_judge

            summaries["judge"].results.append(
                (case, indic_judge(case.prompt, case.response, gold=case.gold or None, judge=judge))
            )
        if "trace" in summaries:
            if case.trace:
                from vindex import check_trace

                summaries["trace"].results.append((case, check_trace(case.prompt, case.trace)))
            else:
                summaries["trace"].skipped += 1
        if on_progress is not None:
            on_progress(i, len(cases))

    return list(summaries.values())


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

METRIC_TITLES = {
    "script": "script_adherence",
    "similarity": "calibrated_similarity",
    "judge": "indic_judge",
    "trace": "check_trace",
}


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _bar(rate: float, width: int = 20) -> str:
    filled = round(rate * width)
    return "█" * filled + "░" * (width - filled)


def render_table(summaries: Sequence[MetricSummary], out: TextIO, style: Style) -> None:
    out.write("\n")
    for s in summaries:
        if not s.total:
            note = f"skipped {s.skipped}" if s.skipped else "no cases"
            out.write(f"  {style.dim('○')} {METRIC_TITLES[s.name]:<22} {style.dim(note)}\n")
            continue
        ok = s.passed == s.total
        mark = style.green("✓") if ok else style.red("✗")
        colour = style.green if ok else (style.yellow if s.pass_rate >= 0.8 else style.red)
        line = (
            f"  {mark} {METRIC_TITLES[s.name]:<22} "
            f"{colour(_bar(s.pass_rate))} "
            f"{style.bold(f'{s.pass_rate:6.1%}')}  "
            f"{style.dim(f'{s.passed}/{s.total} passed')}"
        )
        if s.skipped:
            line += style.dim(f", {s.skipped} skipped")
        out.write(line + "\n")

    failures = [(s, c, r) for s in summaries for c, r in s.results if not r.passed]
    if failures:
        out.write(f"\n  {style.bold('Failures')}\n")
        for s, case, result in failures:
            out.write(
                f"\n  {style.red('●')} {style.bold(case.id)} "
                f"{style.dim('·')} {METRIC_TITLES[s.name]} "
                f"{style.dim('·')} {style.yellow(result.label)}\n"
            )
            out.write(f"    {style.dim('prompt  ')} {_clip(case.prompt, 90)}\n")
            out.write(f"    {style.dim('response')} {_clip(case.response, 90)}\n")
            out.write(f"    {style.dim('reason  ')} {result.reason}\n")
    out.write("\n")


def render_markdown(summaries: Sequence[MetricSummary], out: TextIO) -> None:
    out.write("## vindex report\n\n")
    out.write("| metric | pass rate | passed | skipped | mean score |\n")
    out.write("|---|---:|---:|---:|---:|\n")
    for s in summaries:
        rate = f"{s.pass_rate:.1%}" if s.total else "–"
        mean = f"{s.mean_score:.3f}" if s.total else "–"
        icon = "✅" if s.passed == s.total else "❌"
        out.write(
            f"| {icon} `{METRIC_TITLES[s.name]}` | {rate} | {s.passed}/{s.total} "
            f"| {s.skipped} | {mean} |\n"
        )
    failures = [(s, c, r) for s in summaries for c, r in s.results if not r.passed]
    if failures:
        out.write("\n<details><summary>Failures</summary>\n\n")
        out.write("| case | metric | label | reason |\n|---|---|---|---|\n")
        for s, case, result in failures:
            reason = result.reason.replace("|", "\\|")
            out.write(f"| {case.id} | `{METRIC_TITLES[s.name]}` | `{result.label}` | {reason} |\n")
        out.write("\n</details>\n")


def to_json(summaries: Sequence[MetricSummary]) -> dict[str, Any]:
    return {
        "vindex_version": __version__,
        "metrics": {
            METRIC_TITLES[s.name]: {
                "total": s.total,
                "passed": s.passed,
                "skipped": s.skipped,
                "pass_rate": s.pass_rate,
                "mean_score": s.mean_score,
                "cases": [
                    {
                        "id": case.id,
                        "score": r.score,
                        "passed": r.passed,
                        "label": r.label,
                        "reason": r.reason,
                        "detail": r.detail,
                    }
                    for case, r in s.results
                ],
            }
            for s in summaries
        },
    }


def render_result(result: MetricResult, out: TextIO, style: Style) -> None:
    mark = style.green("✓ PASS") if result.passed else style.red("✗ FAIL")
    score = style.dim(f"score {result.score:.2f}")
    out.write(f"\n  {mark}  {style.yellow(result.label)}  {score}\n")
    out.write(f"  {result.reason}\n\n")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _cmd_check(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    from vindex import script_adherence

    result = script_adherence(args.prompt, args.response, args.strict_language)
    if args.json:
        out.write(json.dumps(_result_dict(result), ensure_ascii=False, indent=2) + "\n")
    else:
        render_result(result, out, style)
    return 0 if result.passed else 1


def _cmd_detect(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    from vindex.language import looks_like_hinglish
    from vindex.script import classify, count_scripts

    counts = {
        k.removesuffix("_chars").removesuffix("_alpha"): v
        for k, v in count_scripts(args.text).items()
        if v and k != "other_chars"
    }
    label = classify(args.text)
    if args.json:
        payload = {"script": label, "counts": counts, "hinglish": looks_like_hinglish(args.text)}
        out.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return 0
    out.write(f"\n  {style.bold('script')}    {style.accent(label)}\n")
    if counts:
        total = sum(counts.values())
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            out.write(f"  {style.dim(f'{name:<10}')}{_bar(n / total, 16)} {n}\n")
    if label == "roman" and looks_like_hinglish(args.text):
        out.write(f"  {style.dim('note')}      looks like Romanized Hindi (Hinglish)\n")
    out.write("\n")
    return 0


def _cmd_run(args: argparse.Namespace, out: TextIO, style: Style) -> int:
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"vindex: {path}: no such file")
    cases = load_cases(path)
    if not cases:
        raise SystemExit(f"vindex: {path}: no cases found")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise SystemExit(f"vindex: unknown metric(s): {', '.join(unknown)}")
    if args.trace_auto and "trace" not in metrics and any(c.trace for c in cases):
        metrics.append("trace")

    show_progress = args.format == "table" and style.enabled and len(cases) > 1

    def progress(i: int, n: int) -> None:
        sys.stderr.write(f"\r  {style.dim(f'evaluating {i}/{n}')}")
        sys.stderr.flush()
        if i == n:
            sys.stderr.write("\r" + " " * 40 + "\r")

    if args.format == "table":
        out.write(
            f"\n  {style.accent('vindex')} {style.dim(__version__)}  "
            f"{style.bold(path.name)} {style.dim(f'· {len(cases)} cases')}\n"
        )

    summaries = evaluate(
        cases,
        metrics,
        language=args.language,
        strict_language_check=args.strict_language,
        judge_spec=args.judge,
        encoder=args.encoder,
        on_progress=progress if show_progress else None,
    )

    if args.format == "json":
        out.write(json.dumps(to_json(summaries), ensure_ascii=False, indent=2) + "\n")
    elif args.format == "markdown":
        render_markdown(summaries, out)
    else:
        render_table(summaries, out, style)

    if args.output:
        Path(args.output).write_text(
            json.dumps(to_json(summaries), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file and args.github_summary:
        with open(summary_file, "a", encoding="utf-8") as fh:
            render_markdown(summaries, fh)

    failing = [s for s in summaries if s.total and s.pass_rate < args.fail_under]
    if args.format == "table":
        if failing:
            names = ", ".join(METRIC_TITLES[s.name] for s in failing)
            out.write(f"  {style.red('FAILED')} {names} below {args.fail_under:.0%}\n\n")
        else:
            ok = style.green("PASSED")
            out.write(f"  {ok} all metrics at or above {args.fail_under:.0%}\n\n")
    return 1 if failing else 0


def _result_dict(result: MetricResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "passed": result.passed,
        "label": result.label,
        "reason": result.reason,
        "detail": result.detail,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vindex",
        description="Evals for Indian-language LLM apps.",
    )
    parser.add_argument("--version", action="version", version=f"vindex {__version__}")
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    check = sub.add_parser("check", help="score one prompt/response pair for script adherence")
    check.add_argument("prompt")
    check.add_argument("response")
    check.add_argument("--strict-language", action="store_true",
                       help="also fail Romanized-Hindi prompts answered in English")
    check.add_argument("--json", action="store_true", help="print the result as JSON")
    check.set_defaults(func=_cmd_check)

    detect = sub.add_parser("detect", help="show which script(s) a text is written in")
    detect.add_argument("text")
    detect.add_argument("--json", action="store_true", help="print the result as JSON")
    detect.set_defaults(func=_cmd_detect)

    run = sub.add_parser("run", help="evaluate a dataset file (.jsonl, .json or .csv)")
    run.add_argument("file")
    run.add_argument("-m", "--metrics", default="script",
                     help="comma-separated: script,similarity,judge,trace (default: script)")
    run.add_argument("-l", "--language", default="",
                     help="default language for similarity: en, hi or hinglish")
    run.add_argument("--judge", default="groq",
                     help="judge for indic_judge: groq, openai[:model], anthropic[:model], "
                          "litellm:<model> (default: groq)")
    run.add_argument("--encoder", default="", help="sentence-transformers encoder for similarity")
    run.add_argument("--strict-language", action="store_true",
                     help="also fail Romanized-Hindi prompts answered in English")
    run.add_argument("--fail-under", type=float, default=1.0, metavar="RATE",
                     help="exit 1 if any metric's pass rate is below RATE (default: 1.0)")
    run.add_argument("-f", "--format", choices=("table", "json", "markdown"), default="table")
    run.add_argument("-o", "--output", help="also write the full JSON report to this path")
    run.add_argument("--no-trace-auto", dest="trace_auto", action="store_false",
                     help="don't run check_trace automatically on rows that have a trace")
    run.add_argument("--no-github-summary", dest="github_summary", action="store_false",
                     help="don't append a report to $GITHUB_STEP_SUMMARY")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = _utf8(sys.stdout)
    _utf8(sys.stderr)
    style = Style(out, enabled=False if args.no_color else None)
    try:
        code: int = args.func(args, out, style)
    except ValueError as exc:
        sys.stderr.write(f"vindex: {exc}\n")
        return 2
    except ImportError as exc:
        sys.stderr.write(
            f"vindex: missing optional dependency ({exc.name or exc}). "
            "Install the extra for this metric, e.g. pip install 'vindex[similarity]' "
            "or 'vindex[judge]'.\n"
        )
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
