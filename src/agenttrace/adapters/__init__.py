"""Framework adapters — the ONLY place that knows which framework produced a trajectory.

Each adapter converts a framework's native trace into a normalized
:class:`agenttrace.schema.Trajectory`. Nothing outside this package may import or branch on a
specific framework.

Importing this package registers all built-in adapters via their modules.
"""

from agenttrace.adapters import (  # noqa: F401  (import for registration side effects)
    crewai,
    langgraph,
    openai_sdk,
)
from agenttrace.adapters.base import Adapter, get, get_all, register

__all__ = ["Adapter", "register", "get", "get_all"]
