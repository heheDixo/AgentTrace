"""Outcome evaluator — task success vs ground truth, falling back to the judge.

If the trajectory's :class:`Outcome` carries a ``ground_truth``, success is decided by a direct
(lenient) comparison against ``final_answer``. Otherwise, an injected :class:`LLMJudge` decides.
With no ground truth and no judge, the evaluator cannot score and raises ``NotImplementedError``
(the CLI treats that as "skipped" rather than fatal).

Depends only on :mod:`agenttrace.schema` and the judge interface.
"""

from __future__ import annotations

from agenttrace.evaluators.base import Evaluator, register
from agenttrace.judge.llm_judge import LLMJudge
from agenttrace.schema import EvalResult, Trajectory


def _matches(final_answer: str | None, ground_truth: str | None) -> bool:
    """Lenient match: case-insensitive equality or ground_truth contained in the answer."""
    answer = (final_answer or "").strip().lower()
    truth = (ground_truth or "").strip().lower()
    if not truth:
        return False
    return truth == answer or truth in answer


@register
class OutcomeEvaluator(Evaluator):
    name = "outcome"

    def __init__(self, judge: LLMJudge | None = None) -> None:
        self._judge = judge

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        outcome = trajectory.outcome

        if outcome.ground_truth is not None:
            success = _matches(outcome.final_answer, outcome.ground_truth)
            return EvalResult(
                evaluator=self.name,
                score=1.0 if success else 0.0,
                passed=success,
                detail={
                    "method": "ground_truth",
                    "final_answer": outcome.final_answer,
                    "ground_truth": outcome.ground_truth,
                    "match": success,
                },
            )

        if self._judge is None:
            raise NotImplementedError(
                "Outcome evaluator needs a judge when the trajectory has no ground_truth."
            )

        verdict = self._judge.judge(trajectory)
        return EvalResult(
            evaluator=self.name,
            score=1.0 if verdict.success else 0.0,
            passed=verdict.success,
            detail={
                "method": "judge",
                "model": self._judge.model,
                "rationale": verdict.rationale,
                "final_answer": outcome.final_answer,
            },
        )
