"""Adapter tests: a real captured trace from the sample OpenAI agent -> a well-formed Trajectory."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from agenttrace.adapters.openai_sdk import OpenAIAgentsAdapter
from agenttrace.schema import RunConfig, Trajectory

# Make examples/ importable without packaging it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


@pytest.fixture(scope="module")
def native_trace():
    """Run the deterministic sample OpenAI agent once and return its raw RunResult."""
    from sample_openai_agent import run

    return run()


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        agent_id="sample-openai",
        task_set="smoke",
        harness_config={"framework": "openai_sdk"},
        budget={"max_steps": 10},
    )


def test_native_trace_has_expected_shape(native_trace) -> None:
    # Guards the fixture: RunResult with 5 new_items (2 calls + 2 outputs + 1 message).
    assert hasattr(native_trace, "new_items")
    assert len(native_trace.new_items) == 5
    assert native_trace.final_output


def test_to_trajectory_is_well_formed(native_trace, config: RunConfig) -> None:
    traj = OpenAIAgentsAdapter().to_trajectory(native_trace, config)

    assert isinstance(traj, Trajectory)
    assert traj.config is config

    # Two tool calls -> two steps, correctly ordered and paired with outputs.
    assert len(traj.steps) == 2
    assert [s.index for s in traj.steps] == [0, 1]

    calc, search = traj.steps
    assert calc.role == "assistant"
    assert calc.tool_name == "calculator"
    assert calc.tool_args == {"expression": "21 * 2"}  # parsed from JSON string
    assert calc.tool_output == "42"
    assert isinstance(calc.timestamp, datetime)

    assert search.tool_name == "fake_search"
    assert search.tool_args == {"query": "OpenAI Agents SDK"}
    assert "Agents SDK" in search.tool_output

    # SDK exposes usage per response: 3 responses * 25 tokens = 75.
    assert traj.total_tokens == 75
    assert traj.total_usd == 0.0
    assert traj.wall_clock_s == 0.0
    assert calc.latency_ms is None


def test_outcome_from_final_output(native_trace, config: RunConfig) -> None:
    traj = OpenAIAgentsAdapter().to_trajectory(native_trace, config)

    assert traj.outcome.success is True
    assert traj.outcome.final_answer == "21 * 2 = 42. The OpenAI Agents SDK runs agents with tools."
    assert traj.outcome.ground_truth is None
    assert traj.outcome.error is None


def test_audit_record_round_trips(native_trace, config: RunConfig) -> None:
    traj = OpenAIAgentsAdapter().to_trajectory(native_trace, config)
    record = traj.to_audit_record()

    reloaded = json.loads(json.dumps(record))
    assert reloaded["config"]["agent_id"] == "sample-openai"
    assert len(reloaded["steps"]) == 2
    assert reloaded["config_fingerprint"] == config.fingerprint()

    rebuilt = Trajectory.model_validate(traj.model_dump(mode="json"))
    assert rebuilt == traj


def test_accepts_bare_item_list(native_trace, config: RunConfig) -> None:
    # The adapter also accepts a raw new_items list (final answer then comes from the last message).
    traj = OpenAIAgentsAdapter().to_trajectory(native_trace.new_items, config)
    assert len(traj.steps) == 2
    assert traj.outcome.final_answer is not None
