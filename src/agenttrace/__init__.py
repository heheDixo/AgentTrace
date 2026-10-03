"""AgentTrace — framework-agnostic trajectory evaluation and regression harness for LLM agents."""

from agenttrace.schema import (
    EvalResult,
    Outcome,
    RunConfig,
    Step,
    Trajectory,
)

__all__ = [
    "EvalResult",
    "Outcome",
    "RunConfig",
    "Step",
    "Trajectory",
]

__version__ = "0.0.1"
