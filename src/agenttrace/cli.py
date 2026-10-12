"""AgentTrace CLI (Typer): ``agenttrace run | report | ci``.

Only ``run`` is implemented end-to-end: it builds a Trajectory from a captured trace (or the
built-in sample LangGraph agent) via the named adapter, runs every registered evaluator, writes
the audit record to disk, and prints a summary table. ``report`` and ``ci`` are still stubs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agenttrace import adapters, evaluators
from agenttrace.judge.calibration import calibration_report
from agenttrace.report import report as report_mod
from agenttrace.schema import EvalResult, RunConfig, Trajectory

app = typer.Typer(
    add_completion=False,
    help="Framework-agnostic trajectory evaluation and regression harness for LLM agents.",
)
console = Console()

# Repo root = .../src/agenttrace/cli.py -> parents[2]; the sample agent lives under examples/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"


def load_sample_trace() -> dict:
    """Run the built-in deterministic sample LangGraph agent and return its raw final state."""
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    from sample_langgraph_agent import run as run_sample  # type: ignore[import-not-found]

    return run_sample()


def load_trace_file(path: str | Path) -> Any:
    """Load a captured trace from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_evaluators(trajectory: Trajectory) -> tuple[list[EvalResult], list[str]]:
    """Run every registered evaluator. Returns (results, names_skipped_as_not_implemented)."""
    results: list[EvalResult] = []
    skipped: list[str] = []
    for cls in evaluators.get_all():
        try:
            results.append(cls().evaluate(trajectory))
        except NotImplementedError:
            skipped.append(cls.name)
    return results, skipped


def execute_run(
    *,
    adapter: str,
    native_trace: Any,
    config: RunConfig,
    out_dir: str | Path,
) -> dict[str, Any]:
    """The full run pipeline, decoupled from CLI I/O so it is directly testable.

    Builds a Trajectory via ``adapter``, runs all evaluators, writes the audit record under
    ``out_dir``, and returns the trajectory, results, skipped evaluators, and audit path.
    """
    adapter_cls = adapters.get(adapter)
    trajectory = adapter_cls().to_trajectory(native_trace, config)

    results, skipped = run_evaluators(trajectory)

    stem = f"{config.agent_id}-{config.fingerprint()[:8]}"
    audit_path = trajectory.write_audit(Path(out_dir) / f"{stem}.audit.json")
    results_path = report_mod.save_results(results, Path(out_dir) / f"{stem}.results.json")

    return {
        "trajectory": trajectory,
        "results": results,
        "skipped": skipped,
        "audit_path": audit_path,
        "results_path": results_path,
    }


def build_golden_config(adapter: str, reference_steps: int | None) -> RunConfig:
    """The fixed task-set config the CI gate runs against (the deterministic sample agent)."""
    harness: dict[str, Any] = {"framework": adapter, "source": "golden"}
    if reference_steps is not None:
        harness["reference_steps"] = reference_steps
    return RunConfig(agent_id="golden-sample", task_set="golden", harness_config=harness)


def execute_ci(
    *,
    adapter: str,
    native_trace: Any,
    config: RunConfig,
    baseline_path: str | Path,
    out_dir: str | Path,
    update_baseline: bool = False,
) -> dict[str, Any]:
    """Run the loop, then diff against (or establish) the baseline. No process exit here."""
    run_out = execute_run(
        adapter=adapter, native_trace=native_trace, config=config, out_dir=out_dir
    )
    results = run_out["results"]
    baseline_path = Path(baseline_path)

    if update_baseline or not baseline_path.exists():
        report_mod.save_results(results, baseline_path)
        return {
            "status": "baseline_written",
            "regressed": False,
            "results": results,
            "baseline_path": baseline_path,
            "diff": None,
        }

    old = report_mod.load_results(baseline_path)
    diff = report_mod.regression_diff(old, results)
    return {
        "status": "compared",
        "regressed": diff["regressed"],
        "results": results,
        "baseline_path": baseline_path,
        "diff": diff,
    }


def _latest_results_file(results_dir: Path) -> Path:
    """Return the most recently modified *.results.json under ``results_dir``."""
    candidates = sorted(results_dir.glob("*.results.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No *.results.json found in {results_dir}. Run `agenttrace run` first."
        )
    return candidates[-1]


@app.command()
def run(
    adapter: str = typer.Option("langgraph", "--adapter", "-a", help="Registered adapter name."),
    trace: str | None = typer.Option(
        None, "--trace", "-t", help="Captured trace JSON. Omit to run the built-in sample agent."
    ),
    agent_id: str = typer.Option("sample-langgraph", "--agent-id", help="Agent id for RunConfig."),
    task_set: str = typer.Option("smoke", "--task-set", help="Task set id for RunConfig."),
    out: str = typer.Option("runs", "--out", "-o", help="Directory for audit records."),
) -> None:
    """Run an agent through an adapter + evaluators."""
    native_trace = load_trace_file(trace) if trace else load_sample_trace()
    config = RunConfig(
        agent_id=agent_id,
        task_set=task_set,
        harness_config={"framework": adapter, "source": trace or "sample"},
    )
    outcome = execute_run(adapter=adapter, native_trace=native_trace, config=config, out_dir=out)
    report_mod.render(outcome["results"], console=console)
    if outcome["skipped"]:
        skipped = ", ".join(outcome["skipped"])
        console.print(f"[yellow]Skipped (not implemented):[/yellow] {skipped}")
    console.print(f"[dim]Audit record:[/dim] {outcome['audit_path']}")
    console.print(f"[dim]Results:[/dim] {outcome['results_path']}")


@app.command()
def report(
    results: str = typer.Option("runs", "--results", "-r", help="Directory of run outputs."),
) -> None:
    """Render the latest run's results."""
    latest = _latest_results_file(Path(results))
    loaded = report_mod.load_results(latest)
    report_mod.render(loaded, title=f"AgentTrace — {latest.name}", console=console)


@app.command()
def calibrate(
    labelled: str = typer.Option(
        ..., "--labelled", "-l", help="JSON file: {trajectories, human_labels[, judge_labels]}."
    ),
) -> None:
    """Report judge↔human agreement (Cohen's κ) over a labelled set.

    The JSON provides ``human_labels`` and either explicit ``judge_labels`` or ``trajectories``
    (whose recorded ``outcome.success`` is then used as the judge label, so no model call is
    needed to inspect an existing calibration set).
    """
    data = json.loads(Path(labelled).read_text(encoding="utf-8"))
    human_labels = [bool(x) for x in data["human_labels"]]

    if "judge_labels" in data:
        judge_labels = [bool(x) for x in data["judge_labels"]]
    else:
        trajectories = [Trajectory.model_validate(t) for t in data["trajectories"]]
        judge_labels = [bool(t.outcome.success) for t in trajectories]

    report = calibration_report(human_labels, judge_labels)

    table = Table(title="AgentTrace — judge calibration")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("n", str(report["n"]))
    table.add_row("Cohen's kappa", f"{report['kappa']:.3f}")
    table.add_row("agreement rate", f"{report['agreement_rate']:.3f}")
    table.add_row("agree / disagree", f"{report['agree']} / {report['disagree']}")
    table.add_row("human_positive", str(report["human_positive"]))
    table.add_row("judge_positive", str(report["judge_positive"]))
    console.print(table)


@app.command()
def bench(
    suite: str = typer.Option(..., "--suite", "-s", help="AgentDojo suite, e.g. banking."),
    adapter: str = typer.Option("langgraph", "--adapter", "-a", help="Registered adapter name."),
    k: int = typer.Option(1, "--k", help="Repeat each case k times for pass^k reliability."),
    limit_user_tasks: int | None = typer.Option(
        None, "--limit-user-tasks", help="Cap the number of user tasks (for a quick run)."
    ),
    limit_injection_tasks: int | None = typer.Option(
        None, "--limit-injection-tasks", help="Cap injection fixtures per user task."
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Write the BenchmarkReport JSON here."
    ),
) -> None:
    """Run an AgentDojo suite (benign + injected) with the deterministic oracle agent."""
    # Lazy import: only pulls in agentdojo when the bench command is actually used.
    from agenttrace.benchmarks.runner import OracleAgent, render_report, run_suite

    adapter_instance = adapters.get(adapter)()
    report = run_suite(
        suite,
        adapter_instance,
        OracleAgent(),
        k=k,
        limit_user_tasks=limit_user_tasks,
        limit_injection_tasks=limit_injection_tasks,
    )
    render_report(report, console=console)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        console.print(f"[dim]Benchmark report written to[/dim] {out_path}")


@app.command()
def ci(
    adapter: str = typer.Option("langgraph", "--adapter", "-a", help="Registered adapter name."),
    baseline: str = typer.Option(
        "runs/baseline.json", "--baseline", "-b", help="Baseline results file to diff against."
    ),
    out: str = typer.Option("runs", "--out", "-o", help="Directory for this run's outputs."),
    reference_steps: int = typer.Option(
        2, "--reference-steps", help="Reference step count for the golden task set."
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Overwrite the baseline with this run and exit 0."
    ),
) -> None:
    """Run the golden task set and exit non-zero if any evaluator regressed vs the baseline."""
    config = build_golden_config(adapter, reference_steps)
    result = execute_ci(
        adapter=adapter,
        native_trace=load_sample_trace(),
        config=config,
        baseline_path=baseline,
        out_dir=out,
        update_baseline=update_baseline,
    )

    if result["status"] == "baseline_written":
        console.print(f"[green]Baseline written[/green] to {result['baseline_path']} — no diff.")
        raise typer.Exit(code=0)

    report_mod.render_diff(result["diff"], console=console)
    if result["regressed"]:
        console.print("[red]CI gate: regression detected — failing.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]CI gate: no regression.[/green]")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
