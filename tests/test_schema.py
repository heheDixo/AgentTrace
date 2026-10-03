"""Real tests for the core schema — audit round-trip, redaction, and fingerprint stability."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agenttrace.schema import Outcome, RunConfig, Step, Trajectory


def _make_trajectory(**config_overrides) -> Trajectory:
    config_kwargs = dict(
        agent_id="demo-agent",
        task_set="smoke",
        harness_config={"model": "claude-opus-4-8", "temperature": 0.0},
        judge=None,
        budget={"max_steps": 10, "max_tokens": 50_000, "max_usd": 0.5},
    )
    config_kwargs.update(config_overrides)
    config = RunConfig(**config_kwargs)

    steps = [
        Step(
            index=0,
            role="assistant",
            tool_name="search",
            tool_args={"query": "weather", "api_key": "sk-secret-123"},
            tool_output="sunny",
            tokens=42,
            latency_ms=120.5,
            timestamp=datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC),
        ),
        Step(
            index=1,
            role="assistant",
            tool_name="finish",
            tool_args={"answer": "It is sunny.", "auth_token": "abc", "password": "hunter2"},
            tool_output=None,
            tokens=8,
            latency_ms=15.0,
            timestamp=datetime(2026, 10, 1, 12, 0, 1, tzinfo=UTC),
        ),
    ]
    outcome = Outcome(success=True, final_answer="It is sunny.", ground_truth="sunny", error=None)
    return Trajectory(
        config=config,
        steps=steps,
        outcome=outcome,
        total_tokens=50,
        total_usd=0.01,
        wall_clock_s=0.14,
    )


def test_audit_record_round_trips() -> None:
    traj = _make_trajectory()
    record = traj.to_audit_record()

    # JSON-serializable.
    dumped = json.dumps(record)
    reloaded = json.loads(dumped)

    assert reloaded["config"]["agent_id"] == "demo-agent"
    assert reloaded["outcome"]["success"] is True
    assert len(reloaded["steps"]) == 2
    assert reloaded["total_tokens"] == 50

    # Reconstruct a Trajectory from the (unredacted) model dump to confirm the schema round-trips.
    rebuilt = Trajectory.model_validate(traj.model_dump(mode="json"))
    assert rebuilt == traj


def test_credential_redaction() -> None:
    traj = _make_trajectory()
    record = traj.to_audit_record()

    # Non-credential args survive.
    assert record["steps"][0]["tool_args"]["query"] == "weather"
    assert record["steps"][1]["tool_args"]["answer"] == "It is sunny."

    # Credential-like keys are redacted.
    assert record["steps"][0]["tool_args"]["api_key"] == "***REDACTED***"
    assert record["steps"][1]["tool_args"]["auth_token"] == "***REDACTED***"
    assert record["steps"][1]["tool_args"]["password"] == "***REDACTED***"

    # The original in-memory trajectory is untouched.
    assert traj.steps[0].tool_args["api_key"] == "sk-secret-123"


def test_write_audit_redacts_on_disk(tmp_path) -> None:
    traj = _make_trajectory()
    path = traj.write_audit(tmp_path / "nested" / "run.audit.json")

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "sk-secret-123" not in text
    assert "hunter2" not in text
    assert "***REDACTED***" in text


def test_fingerprint_is_stable_and_order_independent() -> None:
    a = RunConfig(
        agent_id="x",
        task_set="t",
        harness_config={"model": "m", "temperature": 0.0},
        budget={"max_steps": 5, "max_usd": 1.0},
    )
    b = RunConfig(
        agent_id="x",
        task_set="t",
        # Same content, different insertion order.
        harness_config={"temperature": 0.0, "model": "m"},
        budget={"max_usd": 1.0, "max_steps": 5},
    )
    assert a.fingerprint() == b.fingerprint()
    # Stable across repeated calls.
    assert a.fingerprint() == a.fingerprint()


def test_fingerprint_changes_when_config_changes() -> None:
    base = RunConfig(agent_id="x", task_set="t", harness_config={"model": "m"})
    changed_model = RunConfig(agent_id="x", task_set="t", harness_config={"model": "m2"})
    changed_agent = RunConfig(agent_id="y", task_set="t", harness_config={"model": "m"})
    changed_budget = RunConfig(
        agent_id="x", task_set="t", harness_config={"model": "m"}, budget={"max_steps": 1}
    )

    assert base.fingerprint() != changed_model.fingerprint()
    assert base.fingerprint() != changed_agent.fingerprint()
    assert base.fingerprint() != changed_budget.fingerprint()
