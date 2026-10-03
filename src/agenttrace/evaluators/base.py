"""Evaluator ABC + a simple registry.

New capability = a new registered class, never a new ``if`` branch. Evaluators depend only on
:mod:`agenttrace.schema`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agenttrace.schema import EvalResult, Trajectory

_REGISTRY: dict[str, type[Evaluator]] = {}


class Evaluator(ABC):
    """Scores one aspect of a :class:`Trajectory`."""

    #: Unique evaluator name, used in reports and as the registry key.
    name: str = "base"

    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> EvalResult:
        """Return an :class:`EvalResult` for ``trajectory``."""
        raise NotImplementedError


def register(cls: type[Evaluator]) -> type[Evaluator]:
    """Class decorator that registers an evaluator by its ``name``."""
    name = cls.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Evaluator name {name!r} is already registered.")
    _REGISTRY[name] = cls
    return cls


def get_all() -> list[type[Evaluator]]:
    """Return all registered evaluator classes."""
    return list(_REGISTRY.values())
