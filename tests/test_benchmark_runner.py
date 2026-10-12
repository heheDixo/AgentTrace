"""End-to-end benchmark runner tests with the deterministic oracle agent (no LLM)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from agenttrace.adapters.langgraph import LangGraphAdapter
from agenttrace.benchmarks.runner import (
    BenchmarkReport,
    OracleAgent,
    run_suite,
)
from agenttrace.cli import app


def test_oracle_runs_small_suite_securely() -> None:
    report = run_suite(
        "banking",
        LangGraphAdapter(),
        OracleAgent(),
        k=2,
        limit_user_tasks=2,
        limit_injection_tasks=2,
    )
    assert isinstance(report, BenchmarkReport)

    # 2 user tasks (benign) x k=2, and 2 tasks * 2 injections = 4 injected cases x k=2.
    assert report.n_benign_cases == 2
    assert report.n_injected_cases == 4
    assert report.n_benign_runs == 4
    assert report.n_injected_runs == 8

    # A benign oracle completes every user task and follows no injection.
    assert report.task_utility == 1.0
    assert report.attack_success_rate == 0.0
    assert report.utility_pass_hat_k == 1.0
    assert report.security_pass_hat_k == 1.0

    # Report is JSON-safe.
    json.dumps(report.model_dump())


def test_compromised_oracle_shows_attack_success() -> None:
    report = run_suite(
        "banking",
        LangGraphAdapter(),
        OracleAgent(follow_injections=True),
        k=1,
        limit_user_tasks=2,
        limit_injection_tasks=2,
    )
    # A compromised oracle follows every injection: attack rate 1.0, security pass^k 0.0.
    assert report.attack_success_rate == 1.0
    assert report.security_pass_hat_k == 0.0
    # It still completes the user task.
    assert report.task_utility == 1.0
    assert report.agent == "oracle-compromised"


def test_bench_cli_smoke(tmp_path) -> None:
    out = tmp_path / "bench.json"
    res = CliRunner().invoke(
        app,
        [
            "bench",
            "--suite",
            "banking",
            "--limit-user-tasks",
            "1",
            "--limit-injection-tasks",
            "1",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "attack success rate" in res.output.lower()
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["suite"] == "banking"
    assert data["adapter"] == "langgraph"
