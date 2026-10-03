"""OpenAI Agents SDK adapter: native SDK run -> normalized Trajectory.

This is the ONLY module allowed to know the OpenAI Agents SDK's native shape. A run
(``Runner.run_sync(agent, task)``) returns a ``RunResult`` object (not a dict, unlike LangGraph):

  * ``result.new_items`` — typed ``RunItem`` wrappers, each with a ``.raw_item``:
      - ``ToolCallItem``        raw = ResponseFunctionToolCall(name, arguments=JSON str, call_id)
      - ``ToolCallOutputItem``  raw = dict {"call_id": ..., "output": ...}
      - ``MessageOutputItem``   raw = ResponseOutputMessage(content=[ResponseOutputText], role)
  * ``result.final_output``  — the terminal answer string.
  * ``result.raw_responses`` — one ModelResponse per model turn, each carrying ``usage``.

We normalize that into the SAME :class:`~agenttrace.schema.Trajectory` the LangGraph adapter
produces:

  * One :class:`Step` per tool call — a ``ToolCallItem`` paired with its ``ToolCallOutputItem``
    (matched by ``call_id``). ``arguments`` is a JSON *string* here, so it is parsed into a dict.
  * ``final_output`` becomes :attr:`Outcome.final_answer`.
  * ``total_tokens`` is summed from ``raw_responses[*].usage`` (the SDK exposes per-turn usage, not
    per-item), so unlike the LLM-free LangGraph fixture this adapter can report real token counts.
    ``latency_ms`` / per-item timestamps are not exposed by ``run_sync`` -> ``None`` / ingest-time.
  * ``total_usd`` is ``0.0`` (not derivable); ``wall_clock_s`` is ``0.0`` from ``run_sync``.

Everything is read through small accessors that tolerate both SDK *objects* and dict-serialized
items, so the adapter works on a live run or a captured/replayed trace. No agents-SDK import is
needed at module load.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from agenttrace.adapters.base import Adapter, register
from agenttrace.schema import Outcome, RunConfig, Step, Trajectory


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an SDK object or a serialized dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_args(raw_args: Any) -> dict[str, Any] | None:
    """Parse a tool call's ``arguments`` (a JSON string in this SDK) into a dict.

    Empty args -> ``{}``. Malformed JSON or non-object args -> ``None`` so the tool_call evaluator
    flags the defect honestly rather than hiding it behind an empty dict.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        if not raw_args.strip():
            return {}
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return {} if raw_args is None else None


def _message_text(raw: Any) -> str:
    """Flatten a ResponseOutputMessage's content (list of text blocks) to a string."""
    content = _attr(raw, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = _attr(block, "text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def _response_tokens(resp: Any) -> int | None:
    """Total tokens for one model response from its ``usage``, if present."""
    usage = _attr(resp, "usage")
    if usage is None:
        return None
    total = _attr(usage, "total_tokens")
    if total is None:
        total = (_attr(usage, "input_tokens") or 0) + (_attr(usage, "output_tokens") or 0) or None
    return total


def _unpack(native_trace: Any) -> tuple[list[Any], str | None, list[Any]]:
    """Return (new_items, final_output, raw_responses) from a RunResult, dict, or bare item list."""
    if isinstance(native_trace, (list, tuple)):
        return list(native_trace), None, []
    items = _attr(native_trace, "new_items")
    if items is None:
        raise ValueError("OpenAI Agents trace has no 'new_items'.")
    final_output = _attr(native_trace, "final_output")
    raw_responses = list(_attr(native_trace, "raw_responses") or [])
    return list(items), final_output, raw_responses


@register
class OpenAIAgentsAdapter(Adapter):
    """Normalize an OpenAI Agents SDK run into a :class:`Trajectory`."""

    framework = "openai_sdk"

    def to_trajectory(self, native_trace: Any, config: RunConfig) -> Trajectory:
        items, final_output, raw_responses = _unpack(native_trace)
        # run_sync exposes no per-item timestamps; stamp all steps with one ingest time so the
        # required Step.timestamp field is populated without inventing fake per-step ordering.
        ingested_at = datetime.now(UTC)

        # Map each tool-call output back to the call it answers, keyed by call_id.
        tool_results: dict[str, Any] = {}
        for item in items:
            if _attr(item, "type") == "tool_call_output_item":
                raw = _attr(item, "raw_item")
                call_id = _attr(raw, "call_id")
                if call_id is not None:
                    tool_results[call_id] = raw

        steps: list[Step] = []
        for item in items:
            if _attr(item, "type") != "tool_call_item":
                continue
            raw = _attr(item, "raw_item")
            call_id = _attr(raw, "call_id")
            result = tool_results.get(call_id)
            steps.append(
                Step(
                    index=len(steps),
                    role="assistant",
                    tool_name=_attr(raw, "name"),
                    tool_args=_parse_args(_attr(raw, "arguments")),
                    tool_output=(str(_attr(result, "output")) if result is not None else None),
                    tokens=None,  # SDK reports usage per model response, not per tool call
                    latency_ms=None,
                    timestamp=ingested_at,
                )
            )

        outcome = self._build_outcome(items, final_output)
        total_tokens = sum(
            t for r in raw_responses if (t := _response_tokens(r)) is not None
        )

        return Trajectory(
            config=config,
            steps=steps,
            outcome=outcome,
            total_tokens=total_tokens,
            total_usd=0.0,
            wall_clock_s=0.0,
        )

    @staticmethod
    def _build_outcome(items: list[Any], final_output: str | None) -> Outcome:
        """Derive the terminal Outcome from final_output (or the last message item)."""
        final_answer = final_output
        if not final_answer:
            for item in reversed(items):
                if _attr(item, "type") == "message_output_item":
                    text = _message_text(_attr(item, "raw_item"))
                    if text:
                        final_answer = text
                        break

        # No hard ground truth in a raw SDK trace; downstream evaluators/judge handle that.
        success = bool(final_answer)
        return Outcome(success=success, final_answer=final_answer, ground_truth=None, error=None)
