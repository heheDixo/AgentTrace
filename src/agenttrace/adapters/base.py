"""Adapter ABC — native trace → normalized trajectory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agenttrace.schema import RunConfig, Trajectory

_REGISTRY: dict[str, type[Adapter]] = {}


class Adapter(ABC):
    """Converts one framework's native trace into a normalized :class:`Trajectory`.

    Subclasses are the only code in AgentTrace allowed to know a framework's native shape.
    """

    #: Human-readable name of the framework this adapter handles (e.g. ``"langgraph"``).
    framework: str = "base"

    @abstractmethod
    def to_trajectory(self, native_trace: Any, config: RunConfig) -> Trajectory:
        """Normalize ``native_trace`` into a :class:`Trajectory` under ``config``."""
        raise NotImplementedError


def register(cls: type[Adapter]) -> type[Adapter]:
    """Class decorator that registers an adapter by its ``framework`` name."""
    name = cls.framework
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Adapter framework {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> type[Adapter]:
    """Return the adapter class registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown adapter {name!r}. Registered: {known}.") from None


def get_all() -> list[type[Adapter]]:
    """Return all registered adapter classes."""
    return list(_REGISTRY.values())
