"""Benchmark runner: run a suite of AgentDojo tasks through an adapter + evaluators and aggregate.

Lives in ``benchmarks/`` and MAY import agentdojo (a dataset, not an agent framework). It still
produces only :mod:`agenttrace.schema`-compatible data and drives the *generic* evaluators; the
framework boundary is unchanged (the agent emits a framework-native trace, the adapter normalizes
it, the evaluators never learn which framework or benchmark produced it).

The agent is pluggable: :class:`OracleAgent` replays ground-truth calls deterministically (no LLM)
so the whole suite runs in tests/CI. A real LLM agent implements the same :class:`BenchmarkAgent`
protocol, returning a native trace for whichever adapter is chosen.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from agenttrace import evaluators
from agenttrace.adapters.base import Adapter
from agenttrace.benchmarks.agentdojo import (
    BenchmarkTask,
    InjectionMeta,
    benchmark_block,
    load_tasks,
)
from agenttrace.schema import EvalResult, RunConfig, Trajectory


@runtime_checkable
class BenchmarkAgent(Protocol):
    """An agent the runner can drive: given a task (+ optional injection), return a native trace."""

    name: str

    def run(self, task: BenchmarkTask, injection: InjectionMeta | None) -> Any: ...


def _oracle_langgraph_trace(
    task: BenchmarkTask, injection: InjectionMeta | None, follow_injection: bool
) -> dict[str, Any]:
    """Build a LangGraph-shaped native trace that replays the task's ground-truth calls.

    A perfectly *benign* oracle ignores injections; a *compromised* oracle also replays the
    attacker's call(s) — used to exercise the security metric against a known-bad run.
    """
    messages: list[dict[str, Any]] = [{"type": "human", "content": task.goal}]
    calls = list(task.ground_truth)
    if follow_injection and injection is not None:
        calls = calls + injection.calls

    for i, call in enumerate(calls):
        call_id = f"call_{i}"
        messages.append(
            {
                "type": "ai",
                "content": "",
                "tool_calls": [{"name": call.function, "args": call.args, "id": call_id}],
            }
        )
        messages.append(
            {"type": "tool", "content": "", "name": call.function, "tool_call_id": call_id}
        )
    messages.append({"type": "ai", "content": "done"})
    return {"messages": messages}


class OracleAgent:
    """Deterministic ground-truth agent (no LLM), emitting LangGraph-shaped traces."""

    def __init__(self, *, follow_injections: bool = False, name: str | None = None) -> None:
        self.follow_injections = follow_injections
        self.name = name or ("oracle-compromised" if follow_injections else "oracle")

    def run(self, task: BenchmarkTask, injection: InjectionMeta | None) -> dict[str, Any]:
        return _oracle_langgraph_trace(task, injection, self.follow_injections)


class BenchmarkReport(BaseModel):
    """Aggregate results for a suite run. JSON-safe (primitives only)."""

    suite: str
    adapter: str
    agent: str
    k: int
    version: str

    n_benign_cases: int
    n_injected_cases: int
    n_benign_runs: int
    n_injected_runs: int

    # Headline numbers.
    task_utility: float  # mean utility over benign runs
    attack_success_rate: float  # mean attack success over injected runs

    # pass^k reliability: fraction of cases that succeeded on ALL k repeats.
    utility_pass_hat_k: float
    security_pass_hat_k: float

    # Cost / latency means across every run.
    mean_tokens: float
    mean_usd: float
    mean_latency_ms: float | None = None

    detail: dict[str, Any] = Field(default_factory=dict)


def _run_evaluators(trajectory: Trajectory) -> list[EvalResult]:
    """Run every registered evaluator, skipping ones not applicable (NotImplementedError)."""
    out: list[EvalResult] = []
    for cls in evaluators.get_all():
        try:
            out.append(cls().evaluate(trajectory))
        except NotImplementedError:
            continue
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _run_once(
    suite: str,
    task: BenchmarkTask,
    injection: InjectionMeta | None,
    adapter: Adapter,
    agent: BenchmarkAgent,
) -> dict[str, Any]:
    """One scenario run: agent -> adapter -> evaluators -> extracted metrics."""
    scenario = injection.injection_task_id if injection else "benign"
    config = RunConfig(
        agent_id=f"{agent.name}-{suite}-{task.user_task_id}-{scenario}",
        task_set=f"agentdojo/{suite}",
        harness_config={
            "framework": adapter.framework,
            "task": task.goal,
            "benchmark": benchmark_block(task, injection),
        },
    )
    native = agent.run(task, injection)
    trajectory = adapter.to_trajectory(native, config)
    by_name = {r.evaluator: r for r in _run_evaluators(trajectory)}
    policy = by_name.get("policy")

    return {
        "task_utility": policy.detail["task_utility"] if policy else 0.0,
        "attack_success": policy.detail["attack_success_rate"] if policy else 0.0,
        "total_tokens": float(trajectory.total_tokens),
        "total_usd": float(trajectory.total_usd),
        "latencies": [s.latency_ms for s in trajectory.steps if s.latency_ms is not None],
    }


def run_suite(
    suite: str,
    adapter: Adapter,
    agent: BenchmarkAgent,
    *,
    k: int = 1,
    version: str = "v1",
    limit_user_tasks: int | None = None,
    limit_injection_tasks: int | None = None,
) -> BenchmarkReport:
    """Run all benign + injected cases in ``suite``; return an aggregated report."""
    tasks = load_tasks(suite, version=version)
    if limit_user_tasks is not None:
        tasks = tasks[:limit_user_tasks]

    benign_runs: list[dict[str, Any]] = []
    injected_runs: list[dict[str, Any]] = []
    benign_case_passed: list[bool] = []
    injected_case_secure: list[bool] = []

    for task in tasks:
        # Benign case: measure task utility over k repeats.
        utility_successes = 0
        for _ in range(k):
            m = _run_once(suite, task, None, adapter, agent)
            benign_runs.append(m)
            if m["task_utility"] == 1.0:
                utility_successes += 1
        benign_case_passed.append(utility_successes == k)

        # Injected cases: each injection fixture is a separate security test case.
        injections = task.injections
        if limit_injection_tasks is not None:
            injections = injections[:limit_injection_tasks]
        for injection in injections:
            resisted = 0
            for _ in range(k):
                m = _run_once(suite, task, injection, adapter, agent)
                injected_runs.append(m)
                if m["attack_success"] == 0.0:
                    resisted += 1
            injected_case_secure.append(resisted == k)

    all_runs = benign_runs + injected_runs
    all_latencies = [lat for m in all_runs for lat in m["latencies"]]

    return BenchmarkReport(
        suite=suite,
        adapter=adapter.framework,
        agent=agent.name,
        k=k,
        version=version,
        n_benign_cases=len(tasks),
        n_injected_cases=len(injected_case_secure),
        n_benign_runs=len(benign_runs),
        n_injected_runs=len(injected_runs),
        task_utility=_mean([m["task_utility"] for m in benign_runs]),
        attack_success_rate=_mean([m["attack_success"] for m in injected_runs]),
        utility_pass_hat_k=_mean([1.0 if p else 0.0 for p in benign_case_passed]),
        security_pass_hat_k=(
            _mean([1.0 if s else 0.0 for s in injected_case_secure])
            if injected_case_secure
            else 1.0
        ),
        mean_tokens=_mean([m["total_tokens"] for m in all_runs]),
        mean_usd=_mean([m["total_usd"] for m in all_runs]),
        mean_latency_ms=_mean(all_latencies) if all_latencies else None,
    )


def render_report(report: BenchmarkReport, *, console: Console | None = None) -> None:
    """Print a rich summary table for a :class:`BenchmarkReport`."""
    console = console or Console()
    title = f"AgentTrace bench — {report.suite} via {report.adapter} ({report.agent}, k={report.k})"
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    lat = "—" if report.mean_latency_ms is None else f"{report.mean_latency_ms:.1f} ms"
    rows = [
        ("benign cases / runs", f"{report.n_benign_cases} / {report.n_benign_runs}"),
        ("injected cases / runs", f"{report.n_injected_cases} / {report.n_injected_runs}"),
        ("task utility (benign)", f"{report.task_utility:.3f}"),
        ("attack success rate (injected)", f"{report.attack_success_rate:.3f}"),
        ("utility pass^k", f"{report.utility_pass_hat_k:.3f}"),
        ("security pass^k", f"{report.security_pass_hat_k:.3f}"),
        ("mean tokens / run", f"{report.mean_tokens:.1f}"),
        ("mean $ / run", f"{report.mean_usd:.4f}"),
        ("mean step latency", lat),
    ]
    for metric, value in rows:
        table.add_row(metric, value)
    console.print(table)
