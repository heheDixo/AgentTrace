"""Policy evaluator — OWASP ASI compliance checks on the trajectory. STUB.

Detection/measurement only — never exploitation. This measures an agent's policy-compliance and
robustness; it does not craft or extend attacks.
"""

from __future__ import annotations

from agenttrace.evaluators.base import Evaluator, register
from agenttrace.schema import EvalResult, Trajectory


@register
class PolicyEvaluator(Evaluator):
    name = "policy"

    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        raise NotImplementedError("Policy evaluator not implemented yet.")
