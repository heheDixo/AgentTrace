"""Tests for the no-LLM evaluators: sample Trajectories through each, assert EvalResult shape."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import agenttrace.evaluators as ev
from agenttrace.evaluators.cost_latency import CostLatencyEvaluator
from agenttrace.evaluators.efficiency import EfficiencyEvaluator
from agenttrace.evaluators.tool_call import ToolCallEvaluator
from agenttrace.schema import EvalResult, Outcome, RunConfig, Step, Trajectory

_TS = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)


def _step(index: int, tool_name=None, tool_args=None, latency_ms=None) -> Step:
    return Step(
        index=index,
        role="assistant",
        tool_name=tool_name,
        tool_args=tool_args,
        tool_output=None,
        tokens=None,
        latency_ms=latency_ms,
        timestamp=_TS,
    )


def _trajectory(
    steps, *, budget=None, harness_config=None, total_tokens=0, total_usd=0.0
) -> Trajectory:
    config = RunConfig(
        agent_id="demo",
        task_set="smoke",
        harness_config=harness_config or {},
        budget=budget or {},
    )
    return Trajectory(
        config=config,
        steps=steps,
        outcome=Outcome(success=True, final_answer="ok"),
        total_tokens=total_tokens,
        total_usd=total_usd,
        wall_clock_s=0.0,
    )


def _assert_well_formed(result: EvalResult, name: str) -> None:
    assert isinstance(result, EvalResult)
    assert result.evaluator == name
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.passed, bool)
    assert isinstance(result.detail, dict)


def test_all_three_are_registered() -> None:
    names = {c.name for c in ev.get_all()}
    assert {"cost_latency", "tool_call", "efficiency"} <= names


# --- cost_latency -------------------------------------------------------------------------------


def test_cost_latency_within_budget() -> None:
    traj = _trajectory(
        [_step(0, latency_ms=100.0), _step(1, latency_ms=200.0), _step(2, latency_ms=300.0)],
        budget={"max_tokens": 1000, "max_usd": 1.0},
        total_tokens=400,
        total_usd=0.25,
    )
    r = CostLatencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "cost_latency")
    assert r.passed is True
    assert r.detail["total_tokens"] == 400
    assert r.detail["p50_latency_ms"] == 200.0
    assert r.detail["n_latency_samples"] == 3
    assert r.detail["p95_latency_ms"] >= r.detail["p50_latency_ms"]


def test_cost_latency_over_budget_fails() -> None:
    traj = _trajectory([_step(0)], budget={"max_tokens": 100}, total_tokens=500)
    r = CostLatencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "cost_latency")
    assert r.passed is False
    assert r.score == 0.0  # clamped


def test_cost_latency_no_budget_passes_with_no_latency_samples() -> None:
    traj = _trajectory([_step(0)], total_tokens=10)
    r = CostLatencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "cost_latency")
    assert r.passed is True
    assert r.score == 1.0
    assert r.detail["p50_latency_ms"] is None


# --- tool_call ----------------------------------------------------------------------------------


def test_tool_call_all_valid() -> None:
    traj = _trajectory(
        [_step(0, tool_name="calc", tool_args={"x": 1}), _step(1, tool_name="search", tool_args={})]
    )
    r = ToolCallEvaluator().evaluate(traj)
    _assert_well_formed(r, "tool_call")
    assert r.score == 1.0
    assert r.passed is True
    assert r.detail["n_tool_calls"] == 2
    assert r.detail["invalid_step_indices"] == []


def test_tool_call_detects_null_args() -> None:
    traj = _trajectory(
        [_step(0, tool_name="calc", tool_args={"x": 1}), _step(1, tool_name="bad", tool_args=None)]
    )
    r = ToolCallEvaluator().evaluate(traj)
    _assert_well_formed(r, "tool_call")
    assert r.score == pytest.approx(0.5)
    assert r.passed is False
    assert r.detail["invalid_step_indices"] == [1]


def test_tool_call_no_tools_is_vacuously_valid() -> None:
    traj = _trajectory([_step(0), _step(1)])
    r = ToolCallEvaluator().evaluate(traj)
    _assert_well_formed(r, "tool_call")
    assert r.score == 1.0
    assert r.passed is True
    assert r.detail["n_tool_calls"] == 0


# --- efficiency ---------------------------------------------------------------------------------


def test_efficiency_under_reference() -> None:
    traj = _trajectory([_step(0), _step(1)], harness_config={"reference_steps": 4})
    r = EfficiencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "efficiency")
    assert r.passed is True
    assert r.score == 1.0


def test_efficiency_over_reference_penalized() -> None:
    traj = _trajectory([_step(i) for i in range(8)], harness_config={"reference_steps": 4})
    r = EfficiencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "efficiency")
    assert r.passed is False
    assert r.score == pytest.approx(0.5)
    assert r.detail["n_steps"] == 8


def test_efficiency_no_reference_skips() -> None:
    traj = _trajectory([_step(0)])
    r = EfficiencyEvaluator().evaluate(traj)
    _assert_well_formed(r, "efficiency")
    assert r.passed is True
    assert r.score == 1.0
    assert r.detail["reference_steps"] is None
