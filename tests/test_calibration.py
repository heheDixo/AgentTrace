"""Tests for Cohen's κ and the calibration report / CLI subcommand."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from agenttrace.cli import app
from agenttrace.judge.calibration import calibrate, calibration_report, cohen_kappa
from agenttrace.judge.llm_judge import LLMJudge
from agenttrace.schema import Outcome, RunConfig, Trajectory

_TS = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)


def _traj(success: bool) -> Trajectory:
    return Trajectory(
        config=RunConfig(agent_id="demo", task_set="smoke"),
        steps=[],
        outcome=Outcome(success=success, final_answer="x"),
        total_tokens=0,
        total_usd=0.0,
        wall_clock_s=0.0,
    )


# --- cohen_kappa --------------------------------------------------------------------------------


def test_perfect_agreement_is_one() -> None:
    labels = [True, False, True, True, False]
    assert cohen_kappa(labels, labels) == 1.0


def test_known_hand_computed_value() -> None:
    # human=[T,T,F,F], judge=[T,F,F,F]: po=0.75, pe=0.5 -> kappa=(0.75-0.5)/0.5 = 0.5.
    human = [True, True, False, False]
    judge = [True, False, False, False]
    assert cohen_kappa(human, judge) == pytest.approx(0.5)


def test_random_independent_labels_near_zero() -> None:
    rng = random.Random(1234)
    n = 5000
    human = [rng.random() < 0.5 for _ in range(n)]
    judge = [rng.random() < 0.5 for _ in range(n)]
    assert abs(cohen_kappa(human, judge)) < 0.05


def test_pe_equal_one_edge_case() -> None:
    # Both raters all-True -> pe == 1; convention returns 1.0.
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_total_disagreement_is_negative() -> None:
    # human=[T,T,F,F], judge=[F,F,T,T]: po=0, pe=0.5 -> kappa=-1.0.
    human = [True, True, False, False]
    judge = [False, False, True, True]
    assert cohen_kappa(human, judge) == pytest.approx(-1.0)


def test_length_and_empty_guards() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# --- calibration_report / calibrate -------------------------------------------------------------


def test_calibration_report_counts() -> None:
    human = [True, True, False, False]
    judge = [True, False, False, False]
    report = calibration_report(human, judge)
    assert report["n"] == 4
    assert report["kappa"] == pytest.approx(0.5)
    assert report["agreement_rate"] == pytest.approx(0.75)
    assert report["agree"] == 3
    assert report["disagree"] == 1
    assert report["human_positive"] == 2
    assert report["judge_positive"] == 1


def test_calibrate_with_mock_judge() -> None:
    trajectories = [_traj(True), _traj(True), _traj(False), _traj(False)]
    human = [True, False, False, False]

    # Mock judge: calls the final_answer "x" a success iff it is non-empty -> always True here.
    judge = LLMJudge(lambda _: {"success": True, "rationale": "non-empty answer"})
    report = calibrate(trajectories, human, judge)
    # judge_labels = [T,T,T,T], human=[T,F,F,F] -> agree on idx0 only -> 1/4.
    assert report["agree"] == 1
    assert report["agreement_rate"] == pytest.approx(0.25)


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_calibrate_with_explicit_judge_labels(tmp_path) -> None:
    payload = {
        "human_labels": [True, True, False, False],
        "judge_labels": [True, False, False, False],
    }
    path = tmp_path / "labelled.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    res = CliRunner().invoke(app, ["calibrate", "--labelled", str(path)])
    assert res.exit_code == 0, res.output
    assert "kappa" in res.output.lower()
    assert "0.500" in res.output


def test_cli_calibrate_falls_back_to_outcome_success(tmp_path) -> None:
    payload = {
        "trajectories": [_traj(True).model_dump(mode="json"), _traj(False).model_dump(mode="json")],
        "human_labels": [True, False],
    }
    path = tmp_path / "labelled.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    res = CliRunner().invoke(app, ["calibrate", "--labelled", str(path)])
    assert res.exit_code == 0, res.output
    assert "1.000" in res.output  # perfect agreement -> kappa 1.0
