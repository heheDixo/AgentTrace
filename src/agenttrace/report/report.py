"""Human-readable + JSON reporting and regression diffing.

Depends only on :mod:`agenttrace.schema`. ``render`` both prints a rich table and returns a
JSON-safe dict, so the same call that shows a run can also persist it as a baseline for later
regression gating.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agenttrace.schema import EvalResult

# A new score may dip below the old by at most this much before it counts as a regression.
DEFAULT_REGRESSION_THRESHOLD = 0.03


def to_json_dict(results: Sequence[EvalResult]) -> dict[str, Any]:
    """Return a JSON-safe dict for a set of results (round-trips via :func:`load_results`)."""
    return {"results": [r.model_dump(mode="json") for r in results]}


def _detail_summary(detail: dict[str, Any], *, limit: int = 4) -> str:
    """A compact one-line summary of the most useful detail fields."""
    items = list(detail.items())[:limit]
    return ", ".join(f"{k}={v}" for k, v in items)


def render(
    results: Sequence[EvalResult],
    *,
    title: str = "AgentTrace — evaluation results",
    console: Console | None = None,
) -> dict[str, Any]:
    """Print a clean rich table of results and return a JSON-safe dict of them."""
    console = console or Console()
    table = Table(title=title)
    table.add_column("Evaluator", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Result", justify="center")
    table.add_column("Detail", overflow="fold")

    for r in results:
        table.add_row(
            r.evaluator,
            f"{r.score:.3f}",
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            _detail_summary(r.detail),
        )
    console.print(table)
    return to_json_dict(results)


def render_diff(diff: dict[str, Any], *, console: Console | None = None) -> None:
    """Print a regression diff (from :func:`regression_diff`) as a rich table."""
    console = console or Console()
    table = Table(title=f"AgentTrace — regression diff (threshold {diff['threshold']})")
    table.add_column("Evaluator", style="bold")
    table.add_column("Old", justify="right")
    table.add_column("New", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status", justify="center")

    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.3f}"

    for name, e in diff["evaluators"].items():
        if e["regressed"]:
            status = "[red]REGRESSED[/red]"
        elif e["status"] == "added":
            status = "[cyan]added[/cyan]"
        else:
            status = "[green]ok[/green]"
        table.add_row(name, _fmt(e["old_score"]), _fmt(e["new_score"]), _fmt(e["delta"]), status)

    console.print(table)
    verdict = "[red]REGRESSION[/red]" if diff["regressed"] else "[green]no regression[/green]"
    console.print(f"Overall: {verdict}")


def save_results(results: Sequence[EvalResult], path: str | Path) -> Path:
    """Write results to ``path`` as JSON (e.g. runs/baseline.json). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(results), indent=2), encoding="utf-8")
    return path


def load_results(path: str | Path) -> list[EvalResult]:
    """Load results previously written by :func:`save_results`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalResult.model_validate(r) for r in data["results"]]


def regression_diff(
    old: Sequence[EvalResult],
    new: Sequence[EvalResult],
    *,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> dict[str, Any]:
    """Per-evaluator score diff between an old (baseline) and new result set.

    For each evaluator: ``delta = new_score - old_score`` and ``regressed`` is ``True`` when the
    new score drops below the old by more than ``threshold``. Evaluators present only in ``new``
    are ``"added"`` (never a regression); evaluators lost from ``old`` are ``"removed"`` and
    counted as a regression, since we lost that measurement.
    """
    old_by = {r.evaluator: r for r in old}
    new_by = {r.evaluator: r for r in new}

    evaluators: dict[str, Any] = {}
    for name in sorted(old_by.keys() | new_by.keys()):
        o = old_by.get(name)
        n = new_by.get(name)
        if o is not None and n is not None:
            delta = n.score - o.score
            entry = {
                "status": "compared",
                "old_score": o.score,
                "new_score": n.score,
                "delta": delta,
                "regressed": delta < -threshold,
            }
        elif n is not None:
            entry = {
                "status": "added",
                "old_score": None,
                "new_score": n.score,
                "delta": None,
                "regressed": False,
            }
        else:  # present in old, gone from new
            entry = {
                "status": "removed",
                "old_score": o.score,
                "new_score": None,
                "delta": None,
                "regressed": True,
            }
        evaluators[name] = entry

    return {
        "threshold": threshold,
        "regressed": any(e["regressed"] for e in evaluators.values()),
        "evaluators": evaluators,
    }
