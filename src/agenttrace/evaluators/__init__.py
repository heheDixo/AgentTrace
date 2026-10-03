"""Evaluators — each operates ONLY on the normalized schema, never on a framework's native shape.

Importing this package registers all built-in evaluators via their modules.
"""

from agenttrace.evaluators import (  # noqa: F401  (import for registration side effects)
    cost_latency,
    efficiency,
    outcome,
    policy,
    tool_call,
)
from agenttrace.evaluators.base import Evaluator, get_all, register

__all__ = ["Evaluator", "register", "get_all"]
