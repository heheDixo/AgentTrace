"""LangGraph (+deepagents) adapter: native LangGraph run -> normalized Trajectory.

This is the ONLY module allowed to know LangGraph's / LangChain's native shape. A LangGraph run
(``graph.invoke(...)``) returns the final graph *state*: a dict whose ``messages`` key holds a flat,
ordered list of LangChain ``BaseMessage`` objects —

    HumanMessage                                   # the task
    AIMessage(content="", tool_calls=[{name, args, id}])
    ToolMessage(content=..., name=..., tool_call_id=...)
    ... (repeated) ...
    AIMessage(content="<final answer>")            # terminal, no tool_calls

We normalize that into :class:`~agenttrace.schema.Trajectory`:

  * One :class:`Step` per tool call — an ``AIMessage`` tool call paired with its answering
    ``ToolMessage`` (matched by ``tool_call_id``). That is the schema's unit of "one agent action":
    ``{tool_name, tool_args, tool_output}``.
  * The terminal ``AIMessage`` text becomes :attr:`Outcome.final_answer`.
  * ``tokens`` is read from each assistant turn's ``usage_metadata`` when present; ``latency_ms``
    and per-message timestamps are not exposed by ``invoke()``, so they are ``None`` / ingest-time.
  * ``total_usd`` is ``0.0`` (not derivable from the trace); ``wall_clock_s`` comes from the trace
    when wrapped (see :meth:`to_trajectory`), else ``0.0``.

Messages are read through small accessors that tolerate both LangChain message *objects* and their
dict-serialized form, so the adapter works on a live run or a captured/replayed trace.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agenttrace.adapters.base import Adapter, register
from agenttrace.schema import Outcome, RunConfig, Step, Trajectory

# LangChain message ``.type`` -> normalized role (OTel gen_ai.message.role style).
_ROLE_BY_TYPE = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "system": "system",
    "function": "tool",
}
# Fallback: map class name -> role, for objects without a ``.type``.
_ROLE_BY_CLASS = {
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "ToolMessage": "tool",
    "SystemMessage": "system",
    "FunctionMessage": "tool",
}


def _get(msg: Any, attr: str, default: Any = None) -> Any:
    """Read ``attr`` from a message, whether it is an object or a serialized dict."""
    if isinstance(msg, dict):
        if attr in msg:
            return msg[attr]
        return msg.get("kwargs", {}).get(attr, default)
    return getattr(msg, attr, default)


def _role(msg: Any) -> str:
    mtype = _get(msg, "type")
    if mtype in _ROLE_BY_TYPE:
        return _ROLE_BY_TYPE[mtype]
    return _ROLE_BY_CLASS.get(type(msg).__name__, type(msg).__name__.lower())


def _turn_tokens(msg: Any) -> int | None:
    """Total tokens for an assistant turn from its ``usage_metadata``, if exposed."""
    usage = _get(msg, "usage_metadata")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total is None:
            inp = usage.get("input_tokens") or 0
            out = usage.get("output_tokens") or 0
            total = (inp + out) or None
        return total
    return None


def _extract_messages(native_trace: Any) -> list[Any]:
    """Pull the ordered message list out of whatever LangGraph handed us."""
    if isinstance(native_trace, dict):
        msgs = native_trace.get("messages")
        if msgs is not None:
            return list(msgs)
        # A streamed run may be wrapped as {"state": {...}}.
        state = native_trace.get("state")
        if isinstance(state, dict) and "messages" in state:
            return list(state["messages"])
        raise ValueError("LangGraph trace dict has no 'messages' key.")
    if isinstance(native_trace, (list, tuple)):
        return list(native_trace)
    raise TypeError(f"Unsupported LangGraph trace type: {type(native_trace)!r}")


def _as_text(content: Any) -> str:
    """Flatten LangChain message content (str or list of content blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


@register
class LangGraphAdapter(Adapter):
    """Normalize a LangGraph run into a :class:`Trajectory`."""

    framework = "langgraph"

    def to_trajectory(self, native_trace: Any, config: RunConfig) -> Trajectory:
        messages = _extract_messages(native_trace)
        # invoke() exposes no per-message timestamps; stamp all steps with one ingest time so the
        # required Step.timestamp field is populated without inventing fake per-step ordering.
        ingested_at = datetime.now(UTC)

        # Map each ToolMessage back to the tool call it answers, keyed by tool_call_id.
        tool_results: dict[str, Any] = {}
        tool_errored = False
        for msg in messages:
            if _role(msg) == "tool":
                call_id = _get(msg, "tool_call_id")
                if call_id is not None:
                    tool_results[call_id] = msg
                if _get(msg, "status") == "error":
                    tool_errored = True

        steps: list[Step] = []
        for msg in messages:
            tool_calls = _get(msg, "tool_calls") or []
            if not tool_calls:
                continue
            turn_tokens = _turn_tokens(msg)
            for call in tool_calls:
                call_id = _get(call, "id")
                result = tool_results.get(call_id)
                steps.append(
                    Step(
                        index=len(steps),
                        role=_role(msg),
                        tool_name=_get(call, "name"),
                        tool_args=_get(call, "args") or {},
                        tool_output=(
                            _as_text(_get(result, "content")) if result is not None else None
                        ),
                        # Turn-level usage is annotated on the first resulting step only, so the
                        # per-step tokens never double-count a multi-call turn.
                        tokens=turn_tokens if call is tool_calls[0] else None,
                        latency_ms=None,
                        timestamp=ingested_at,
                    )
                )

        outcome = self._build_outcome(messages, config, tool_errored)
        total_tokens = sum(t for m in messages if (t := _turn_tokens(m)) is not None)

        # When the run is wrapped with timing metadata, honor it; otherwise it is not derivable.
        wall_clock_s = 0.0
        if isinstance(native_trace, dict):
            wall_clock_s = float(native_trace.get("wall_clock_s") or 0.0)

        return Trajectory(
            config=config,
            steps=steps,
            outcome=outcome,
            total_tokens=total_tokens,
            total_usd=0.0,
            wall_clock_s=wall_clock_s,
        )

    @staticmethod
    def _build_outcome(messages: list[Any], config: RunConfig, tool_errored: bool) -> Outcome:
        """Derive the terminal Outcome from the graph end state."""
        final_answer: str | None = None
        for msg in reversed(messages):
            if _role(msg) == "assistant":
                text = _as_text(_get(msg, "content"))
                if text:
                    final_answer = text
                    break

        error = "tool call reported an error status" if tool_errored else None
        # No hard ground truth in a raw LangGraph trace; a judge/ground-truth comparison happens
        # downstream in the evaluators. Here, success means the graph produced a final answer and
        # no tool error surfaced.
        success = bool(final_answer) and error is None
        return Outcome(success=success, final_answer=final_answer, ground_truth=None, error=error)
