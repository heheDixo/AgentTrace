"""LLM-as-judge for outcomes without hard ground truth.

The model call is pluggable: :class:`LLMJudge` takes a ``complete`` callable so no vendor is
hard-wired and tests can inject a fake with no network. The callable receives a rendered prompt
string and returns the model's verdict, either as a JSON string ``{"success": ..., "rationale":
...}`` (a real LLM wrapper) or directly as a dict / :class:`Verdict` (a mock).

Depends only on :mod:`agenttrace.schema`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agenttrace.schema import Trajectory

#: A pluggable model call: rendered prompt -> model output (JSON str, dict, or Verdict).
JudgeFn = Callable[[str], Any]


class Verdict(BaseModel):
    """A structured judge verdict. Outcome-like, but scoped to what the judge decides."""

    success: bool
    rationale: str = ""


def _parse_json_blob(text: str) -> dict[str, Any]:
    """Tolerantly extract a JSON object from model text (handles ```json fences / prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop a leading ```json / ``` fence and any trailing fence.
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


class LLMJudge:
    """Judges a trajectory's outcome when there is no hard ground truth to compare against."""

    def __init__(self, complete: JudgeFn, model: str = "unspecified") -> None:
        self._complete = complete
        self.model = model

    def build_prompt(self, trajectory: Trajectory) -> str:
        """Render a judging prompt from the normalized trajectory (no framework specifics)."""
        cfg = trajectory.config
        # The natural-language task is not a first-class schema field; use an explicit task from
        # harness_config when provided, else fall back to the task_set identifier.
        task = cfg.harness_config.get("task") or cfg.task_set
        steps_summary = "\n".join(
            f"  - {s.tool_name}({s.tool_args}) -> {s.tool_output}"
            for s in trajectory.steps
            if s.tool_name is not None
        )
        return (
            "You are grading whether an AI agent succeeded at its task.\n"
            f"Task: {task}\n"
            f"Agent steps:\n{steps_summary or '  (none)'}\n"
            f"Final answer: {trajectory.outcome.final_answer}\n\n"
            'Respond with JSON: {"success": <true|false>, "rationale": "<one sentence>"}'
        )

    def judge(self, trajectory: Trajectory) -> Verdict:
        """Return a structured :class:`Verdict` for ``trajectory``."""
        raw = self._complete(self.build_prompt(trajectory))
        return self._coerce(raw)

    @staticmethod
    def _coerce(raw: Any) -> Verdict:
        if isinstance(raw, Verdict):
            return raw
        if isinstance(raw, dict):
            return Verdict.model_validate(raw)
        if isinstance(raw, str):
            return Verdict.model_validate(_parse_json_blob(raw))
        raise TypeError(f"Judge callable returned unsupported type: {type(raw)!r}")
