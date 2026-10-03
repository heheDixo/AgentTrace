"""CrewAI adapter (breadth / comparison target). STUB."""

from __future__ import annotations

from typing import Any

from agenttrace.adapters.base import Adapter, register
from agenttrace.schema import RunConfig, Trajectory


@register
class CrewAIAdapter(Adapter):
    """Normalize a CrewAI run into a Trajectory."""

    framework = "crewai"

    def to_trajectory(self, native_trace: Any, config: RunConfig) -> Trajectory:
        raise NotImplementedError("CrewAI adapter not implemented yet.")
