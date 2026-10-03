"""Cost/latency evaluator — tokens, $/run, and p50/p95 latency vs the RunConfig budget.

Depends only on :mod:`agenttrace.schema`. No LLM calls.
"""

from __future__ import annotations

from collections.abc import Sequence

from agenttrace.evaluators.base import Evaluator, register
from agenttrace.schema import EvalResult, Trajectory


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (``pct`` in [0, 100]); ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


@register
class CostLatencyEvaluator(Evaluator):
    """Pass when the run stays within ``budget.max_tokens`` and ``budget.max_usd``.

    ``score`` is normalized budget headroom in [0, 1]: ``1.0`` means well under budget (or no
    budget set), ``0.0`` means at/over budget. ``detail`` carries the raw numbers.
    """

    name = "cost_latency"

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        budget = trajectory.config.budget or {}
        max_tokens = budget.get("max_tokens")
        max_usd = budget.get("max_usd")

        latencies = [s.latency_ms for s in trajectory.steps if s.latency_ms is not None]
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)

        ratios: list[float] = []
        token_ratio = usd_ratio = None
        if max_tokens:
            token_ratio = trajectory.total_tokens / max_tokens
            ratios.append(token_ratio)
        if max_usd:
            usd_ratio = trajectory.total_usd / max_usd
            ratios.append(usd_ratio)

        worst_ratio = max(ratios) if ratios else 0.0
        passed = worst_ratio <= 1.0
        score = max(0.0, min(1.0, 1.0 - worst_ratio))

        return EvalResult(
            evaluator=self.name,
            score=score,
            passed=passed,
            detail={
                "total_tokens": trajectory.total_tokens,
                "total_usd": trajectory.total_usd,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "n_latency_samples": len(latencies),
                "max_tokens": max_tokens,
                "max_usd": max_usd,
                "token_ratio": token_ratio,
                "usd_ratio": usd_ratio,
            },
        )
