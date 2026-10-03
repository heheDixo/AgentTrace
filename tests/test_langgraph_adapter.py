"""Adapter tests: a real captured trace from the sample agent -> a well-formed Trajectory."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from agenttrace.adapters.langgraph import LangGraphAdapter
from agenttrace.schema import RunConfig, Trajectory

# Make examples/ importable without packaging it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


@pytest.fixture(scope="module")
def native_trace() -> dict:
    """Run the deterministic sample LangGraph agent once and return its raw final state."""
    from sample_langgraph_agent import run

    return run()


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        agent_id="sample-langgraph",
        task_set="smoke",
        harness_config={"framework": "langgraph"},
        budget={"max_steps": 10},
    )


def test_native_trace_has_expected_shape(native_trace: dict) -> None:
    # Guards the fixture: if the sample agent's output shape drifts, the adapter tests below
    # are only meaningful if this still holds.
    assert isinstance(native_trace, dict)
    assert "messages" in native_trace
    assert len(native_trace["messages"]) == 6


def test_to_trajectory_is_well_formed(native_trace: dict, config: RunConfig) -> None:
    traj = LangGraphAdapter().to_trajectory(native_trace, config)

    assert isinstance(traj, Trajectory)
    assert traj.config is config

    # Two tool calls in the sample -> two steps, correctly ordered and paired with outputs.
    assert len(traj.steps) == 2
    assert [s.index for s in traj.steps] == [0, 1]

    calc, search = traj.steps
    assert calc.role == "assistant"
    assert calc.tool_name == "calculator"
    assert calc.tool_args == {"expression": "21 * 2"}
    assert calc.tool_output == "42"
    assert isinstance(calc.timestamp, datetime)

    assert search.tool_name == "fake_search"
    assert search.tool_args == {"query": "LangGraph"}
    assert "LangGraph" in search.tool_output

    # The LLM-free fixture exposes no usage/timing; those stay None / zero.
    assert calc.latency_ms is None
    assert traj.total_tokens == 0
    assert traj.total_usd == 0.0
    assert traj.wall_clock_s == 0.0


def test_outcome_from_end_state(native_trace: dict, config: RunConfig) -> None:
    traj = LangGraphAdapter().to_trajectory(native_trace, config)

    assert traj.outcome.success is True
    assert traj.outcome.final_answer == "21 * 2 = 42. LangGraph is a library for agents."
    assert traj.outcome.ground_truth is None
    assert traj.outcome.error is None


def test_audit_record_round_trips(native_trace: dict, config: RunConfig) -> None:
    traj = LangGraphAdapter().to_trajectory(native_trace, config)
    record = traj.to_audit_record()

    # JSON-serializable and reconstructable.
    reloaded = json.loads(json.dumps(record))
    assert reloaded["config"]["agent_id"] == "sample-langgraph"
    assert len(reloaded["steps"]) == 2
    assert reloaded["config_fingerprint"] == config.fingerprint()

    rebuilt = Trajectory.model_validate(traj.model_dump(mode="json"))
    assert rebuilt == traj


def test_accepts_bare_message_list(native_trace: dict, config: RunConfig) -> None:
    # The adapter also accepts a raw message list, not just the state dict.
    traj = LangGraphAdapter().to_trajectory(native_trace["messages"], config)
    assert len(traj.steps) == 2
    assert traj.outcome.final_answer is not None
