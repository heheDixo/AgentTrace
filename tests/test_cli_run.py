"""End-to-end test of the `run` pipeline: sample agent -> adapter -> evaluators -> audit file."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from agenttrace.cli import app, execute_run, load_sample_trace
from agenttrace.schema import EvalResult, RunConfig, Trajectory


def test_execute_run_end_to_end(tmp_path) -> None:
    native_trace = load_sample_trace()
    config = RunConfig(
        agent_id="sample-langgraph",
        task_set="smoke",
        harness_config={"framework": "langgraph", "reference_steps": 2},
    )

    result = execute_run(
        adapter="langgraph", native_trace=native_trace, config=config, out_dir=tmp_path
    )

    # A real Trajectory came out of the adapter.
    traj = result["trajectory"]
    assert isinstance(traj, Trajectory)
    assert len(traj.steps) == 2

    # The implemented evaluators produced well-formed EvalResults.
    results = result["results"]
    assert len(results) >= 3
    assert all(isinstance(r, EvalResult) for r in results)
    names = {r.evaluator for r in results}
    assert {"cost_latency", "tool_call", "efficiency"} <= names

    # Stub evaluators are skipped, not fatal.
    assert set(result["skipped"]) <= {"outcome", "policy"}

    # The audit record was written and round-trips as JSON.
    audit_path = result["audit_path"]
    assert audit_path.exists()
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["config"]["agent_id"] == "sample-langgraph"
    assert len(record["steps"]) == 2
    assert record["config_fingerprint"] == config.fingerprint()


def test_cli_run_command_smoke(tmp_path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["run", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "evaluation results" in res.output.lower()
    # An audit file landed in the out dir.
    assert list(tmp_path.glob("*.audit.json"))
