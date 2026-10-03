"""Efficiency evaluator — step count vs a reference count (flags wasteful agents).

The reference is passed via ``RunConfig.harness_config["reference_steps"]``. Depends only on
:mod:`agenttrace.schema`. No LLM calls.
"""

from __future__ import annotations

from agenttrace.evaluators.base import Evaluator, register
from agenttrace.schema import EvalResult, Trajectory


@register
class EfficiencyEvaluator(Evaluator):
    """Compare ``len(steps)`` to ``reference_steps``.

    ``score`` = ``reference / max(n_steps, 1)`` capped at ``1.0``: at or under the reference scores
    ``1.0``; taking more steps scales the score down proportionally. ``passed`` iff at/under the
    reference. With no reference provided, the check is skipped (score ``1.0``, passed ``True``).
    """

    name = "efficiency"

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        reference = trajectory.config.harness_config.get("reference_steps")
        n_steps = len(trajectory.steps)

        if reference is None:
            return EvalResult(
                evaluator=self.name,
                score=1.0,
                passed=True,
                detail={
                    "n_steps": n_steps,
                    "reference_steps": None,
                    "ratio": None,
                    "note": "no reference_steps in harness_config; efficiency not scored",
                },
            )

        ratio = reference / max(n_steps, 1)
        score = min(1.0, ratio)
        passed = n_steps <= reference

        return EvalResult(
            evaluator=self.name,
            score=score,
            passed=passed,
            detail={
                "n_steps": n_steps,
                "reference_steps": reference,
                "ratio": ratio,
            },
        )
