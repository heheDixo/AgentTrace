"""Tests for the `agenttrace ci` regression gate: baseline establish, pass, and fail paths."""

from __future__ import annotations

from typer.testing import CliRunner

from agenttrace.cli import app

runner = CliRunner()


def test_ci_establishes_baseline_then_passes_clean(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "runs"

    # First run: no baseline yet -> it is written, exit 0.
    first = runner.invoke(
        app, ["ci", "--baseline", str(baseline), "--out", str(out), "--reference-steps", "2"]
    )
    assert first.exit_code == 0, first.output
    assert baseline.exists()
    assert "baseline written" in first.output.lower()

    # Second run, same golden config -> identical scores -> no regression, exit 0.
    second = runner.invoke(
        app, ["ci", "--baseline", str(baseline), "--out", str(out), "--reference-steps", "2"]
    )
    assert second.exit_code == 0, second.output
    assert "no regression" in second.output.lower()


def test_ci_fails_on_regression(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "runs"

    # Establish a baseline where efficiency scores 1.0 (reference_steps == actual 2 steps).
    established = runner.invoke(
        app, ["ci", "--baseline", str(baseline), "--out", str(out), "--reference-steps", "2"]
    )
    assert established.exit_code == 0, established.output

    # New run with a tighter reference (1) -> 2 steps now over budget -> efficiency drops to 0.5.
    regressed = runner.invoke(
        app, ["ci", "--baseline", str(baseline), "--out", str(out), "--reference-steps", "1"]
    )
    assert regressed.exit_code != 0, regressed.output
    assert "regression detected" in regressed.output.lower()


def test_ci_update_baseline_flag_overwrites_and_passes(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "runs"

    # Seed a baseline at the good score.
    runner.invoke(
        app, ["ci", "--baseline", str(baseline), "--out", str(out), "--reference-steps", "2"]
    )
    # Even a worse run exits 0 when explicitly refreshing the baseline.
    refreshed = runner.invoke(
        app,
        [
            "ci",
            "--baseline",
            str(baseline),
            "--out",
            str(out),
            "--reference-steps",
            "1",
            "--update-baseline",
        ],
    )
    assert refreshed.exit_code == 0, refreshed.output
    assert "baseline written" in refreshed.output.lower()
