"""Tests for the `vindex` command-line interface."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from vindex.cli import load_cases, main


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["--no-color", *argv])
    return code, capsys.readouterr().out


def test_check_pass(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["check", "Mumbai kahan hai?", "Mumbai Maharashtra mein hai."], capsys)
    assert code == 0
    assert "PASS" in out and "matched" in out


def test_check_fail_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["check", "Mumbai kahan hai?", "मुंबई महाराष्ट्र में है।", "--json"], capsys)
    assert code == 1
    assert json.loads(out)["label"] == "script_mismatch"


def test_detect_json(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run(["detect", "Kal office aa raha hai kya?", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0
    assert payload["script"] == "roman"
    assert payload["hinglish"] is True


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_cases_aliases(tmp_path: Path) -> None:
    jsonl = _write(
        tmp_path,
        "c.jsonl",
        json.dumps({"input": "Q", "actual_output": "A", "expected_output": "G"}) + "\n\n",
    )
    case = load_cases(jsonl)[0]
    assert (case.id, case.prompt, case.response, case.gold) == ("1", "Q", "A", "G")

    csv_path = _write(tmp_path, "c.csv", "Question,Answer,lang\nQ,A,hi\n")
    case = load_cases(csv_path)[0]
    assert (case.prompt, case.response, case.language) == ("Q", "A", "hi")

    payload = {"cases": [{"prompt": "Q", "response": "A"}]}
    json_path = _write(tmp_path, "c.json", json.dumps(payload))
    assert load_cases(json_path)[0].response == "A"


def test_load_cases_bad_jsonl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=":2:"):
        load_cases(_write(tmp_path, "bad.jsonl", '{"prompt": "a"}\n{nope\n'))


def test_run_table_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        {"id": "ok", "prompt": "Mumbai kahan hai?", "response": "Mumbai Maharashtra mein hai."},
        {"id": "bad", "prompt": "Mumbai kahan hai?", "response": "मुंबई महाराष्ट्र में है।"},
    ]
    path = _write(tmp_path, "cases.jsonl", "\n".join(json.dumps(r) for r in rows))

    code, out = _run(["run", str(path), "--no-github-summary"], capsys)
    assert code == 1
    assert "50.0%" in out and "bad" in out and "script_mismatch" in out

    code, _ = _run(["run", str(path), "--fail-under", "0.5", "--no-github-summary"], capsys)
    assert code == 0


def test_run_json_and_trace_auto(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    row = {
        "prompt": "समुद्र तल पर पानी किस तापमान पर उबलता है?",
        "response": "समुद्र तल पर पानी 100 डिग्री सेल्सियस पर उबलता है।",
        "trace": "The question asks for the boiling point at the sea floor.",
    }
    path = _write(tmp_path, "cases.jsonl", json.dumps(row, ensure_ascii=False))
    report_path = tmp_path / "report.json"
    code, out = _run(
        ["run", str(path), "-f", "json", "-o", str(report_path), "--no-github-summary"], capsys
    )
    report = json.loads(out)
    assert code == 1
    assert report["metrics"]["script_adherence"]["passed"] == 1
    assert report["metrics"]["check_trace"]["cases"][0]["label"] == "misread_detected"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_run_markdown_and_github_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, "c.jsonl", json.dumps({"prompt": "Hello?", "response": "Hi there."}))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code, out = _run(["run", str(path), "-f", "markdown"], capsys)
    assert code == 0
    assert out.startswith("## vindex report")
    assert "script_adherence" in summary.read_text(encoding="utf-8")


def test_run_unknown_metric(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.jsonl", json.dumps({"prompt": "a", "response": "b"}))
    with pytest.raises(SystemExit, match="unknown metric"):
        main(["run", str(path), "-m", "script,bogus"])


def test_similarity_skips_rows_without_gold(tmp_path: Path) -> None:
    from vindex.cli import evaluate

    cases = load_cases(_write(tmp_path, "c.jsonl", json.dumps({"prompt": "a", "response": "b"})))
    (summary,) = evaluate(cases, ["similarity"], language="en")
    assert summary.total == 0 and summary.skipped == 1


def test_bad_judge_provider() -> None:
    from vindex.cli import make_judge

    with pytest.raises(ValueError, match="unknown judge provider"):
        make_judge("nope")


def test_utf8_stream_is_safe() -> None:
    from vindex.cli import _utf8

    stream = io.StringIO()
    assert _utf8(stream) is stream
