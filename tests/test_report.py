"""Tests for report rendering, persistence, and regression diffing."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from agenttrace.cli import app, execute_run, load_sample_trace
from agenttrace.report.report import (
    load_results,
    regression_diff,
    render,
    save_results,
    to_json_dict,
)
from agenttrace.schema import EvalResult, RunConfig


def _result(name: str, score: float, passed: bool | None = None) -> EvalResult:
    return EvalResult(
        evaluator=name,
        score=score,
        passed=passed if passed is not None else score >= 0.5,
        detail={"note": f"{name} detail", "score": score},
    )


# --- render / persistence -----------------------------------------------------------------------


def test_render_prints_and_returns_json_safe_dict() -> None:
    buf = io.StringIO()
    results = [_result("outcome", 1.0), _result("efficiency", 0.4)]
    out = render(results, console=Console(file=buf, width=120))

    # Printed a table.
    printed = buf.getvalue()
    assert "outcome" in printed and "efficiency" in printed

    # Returned dict is JSON-safe and round-trips.
    assert out == to_json_dict(results)
    json.dumps(out)  # does not raise
    assert out["results"][0]["evaluator"] == "outcome"


def test_save_and_load_round_trip(tmp_path) -> None:
    results = [_result("outcome", 1.0), _result("tool_call", 0.5)]
    path = save_results(results, tmp_path / "baseline.json")
    assert path.exists()

    loaded = load_results(path)
    assert loaded == results


# --- regression_diff ----------------------------------------------------------------------------


def test_identical_results_no_regression() -> None:
    results = [_result("outcome", 1.0), _result("efficiency", 0.8)]
    diff = regression_diff(results, results)
    assert diff["regressed"] is False
    assert diff["evaluators"]["outcome"]["delta"] == 0.0
    assert diff["evaluators"]["outcome"]["regressed"] is False


def test_dropped_score_is_regression() -> None:
    old = [_result("outcome", 1.0)]
    new = [_result("outcome", 0.80)]  # drop of 0.20 > 0.03
    diff = regression_diff(old, new)
    assert diff["regressed"] is True
    entry = diff["evaluators"]["outcome"]
    assert entry["delta"] == pytest.approx(-0.20)
    assert entry["regressed"] is True
    assert entry["status"] == "compared"


def test_small_drop_within_threshold_not_regression() -> None:
    old = [_result("outcome", 1.0)]
    new = [_result("outcome", 0.98)]  # drop of 0.02 <= 0.03
    diff = regression_diff(old, new)
    assert diff["regressed"] is False
    assert diff["evaluators"]["outcome"]["regressed"] is False


def test_custom_threshold() -> None:
    old = [_result("outcome", 1.0)]
    new = [_result("outcome", 0.95)]  # drop 0.05
    assert regression_diff(old, new, threshold=0.10)["regressed"] is False
    assert regression_diff(old, new, threshold=0.01)["regressed"] is True


def test_improved_score_not_regression() -> None:
    old = [_result("efficiency", 0.5)]
    new = [_result("efficiency", 0.9)]
    diff = regression_diff(old, new)
    assert diff["regressed"] is False
    assert diff["evaluators"]["efficiency"]["delta"] == pytest.approx(0.4)


def test_added_and_removed_evaluators() -> None:
    old = [_result("outcome", 1.0), _result("gone", 1.0)]
    new = [_result("outcome", 1.0), _result("new_one", 0.5)]
    diff = regression_diff(old, new)
    assert diff["evaluators"]["new_one"]["status"] == "added"
    assert diff["evaluators"]["new_one"]["regressed"] is False
    assert diff["evaluators"]["gone"]["status"] == "removed"
    assert diff["evaluators"]["gone"]["regressed"] is True
    assert diff["regressed"] is True  # a removed measurement counts


# --- CLI report ---------------------------------------------------------------------------------


def test_cli_report_renders_latest(tmp_path) -> None:
    # Produce a run so there is a results file to report on.
    config = RunConfig(agent_id="sample-langgraph", task_set="smoke")
    execute_run(
        adapter="langgraph", native_trace=load_sample_trace(), config=config, out_dir=tmp_path
    )

    res = CliRunner().invoke(app, ["report", "--results", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "cost_latency" in res.output


def test_cli_report_errors_when_empty(tmp_path) -> None:
    res = CliRunner().invoke(app, ["report", "--results", str(tmp_path)])
    assert res.exit_code != 0
