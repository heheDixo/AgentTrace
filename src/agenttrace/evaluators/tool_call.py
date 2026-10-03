"""Tool-call evaluator — each tool step must carry non-null ``tool_args`` that are a dict.

Depends only on :mod:`agenttrace.schema`. No LLM calls. A deeper check (validating arguments
against each tool's JSON schema) can layer on later; this is the structural baseline.
"""

from __future__ import annotations

from agenttrace.evaluators.base import Evaluator, register
from agenttrace.schema import EvalResult, Trajectory

# Minimum fraction of valid tool calls to pass; overridable via harness_config.
_DEFAULT_THRESHOLD = 1.0


@register
class ToolCallEvaluator(Evaluator):
    """``score`` = fraction of tool calls with valid (dict, non-null) args; pass above threshold.

    A trajectory with no tool calls scores ``1.0`` (vacuously valid).
    """

    name = "tool_call"

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        threshold = trajectory.config.harness_config.get("tool_call_threshold", _DEFAULT_THRESHOLD)

        tool_steps = [s for s in trajectory.steps if s.tool_name is not None]
        invalid_indices = [
            s.index for s in tool_steps if not isinstance(s.tool_args, dict)
        ]
        n_calls = len(tool_steps)
        n_valid = n_calls - len(invalid_indices)

        score = 1.0 if n_calls == 0 else n_valid / n_calls
        passed = score >= threshold

        return EvalResult(
            evaluator=self.name,
            score=score,
            passed=passed,
            detail={
                "n_tool_calls": n_calls,
                "n_valid": n_valid,
                "invalid_step_indices": invalid_indices,
                "threshold": threshold,
            },
        )
