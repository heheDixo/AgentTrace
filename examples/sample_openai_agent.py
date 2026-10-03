"""A tiny, deterministic OpenAI Agents SDK agent used to explore the SDK's native run shape.

Like the LangGraph sample, this is deliberately LLM-free: instead of calling a real model, a
``ScriptedModel`` implements the SDK's ``Model`` interface and returns a fixed sequence of tool
calls followed by a final message. That keeps the run fully deterministic and runnable with no API
key (so it can be a test fixture), while still driving the *real* Agents SDK ``Runner`` loop —
real ``@function_tool`` execution, real ``RunResult`` / response items.

This file is EXPLORE-ONLY: it exists to reveal the native shape the SDK returns. No adapter here.

Run directly to print the raw shape::

    python examples/sample_openai_agent.py
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any

from agents import Agent, ModelResponse, Runner, function_tool
from agents.models.interface import Model, ModelTracing
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

TASK = "What is 21 * 2, and what is the OpenAI Agents SDK?"

# --- Tools (same two as the LangGraph sample, for an apples-to-apples comparison) -----------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError("unsupported expression")


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '21 * 2'."""
    return str(_safe_eval(ast.parse(expression, mode="eval")))


@function_tool
def fake_search(query: str) -> str:
    """Return a canned, deterministic search snippet for a query."""
    snippets = {
        "openai agents sdk": "The OpenAI Agents SDK runs agents with tools via a Runner loop.",
    }
    return snippets.get(query.strip().lower(), f"No results for {query!r}.")


# --- A scripted, pluggable model (implements the SDK Model interface) -----------------------------


def _count_tool_outputs(input_items: Any) -> int:
    """How many tool results are already in the running input (drives the next scripted step)."""
    if not isinstance(input_items, list):
        return 0
    return sum(
        1
        for it in input_items
        if isinstance(it, dict) and it.get("type") == "function_call_output"
    )


class ScriptedModel(Model):
    """Deterministic stand-in for an LLM: emits calculator -> fake_search -> final answer."""

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing: ModelTracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> ModelResponse:
        step = _count_tool_outputs(input)
        usage = Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25)

        if step == 0:
            item = ResponseFunctionToolCall(
                id="fc_1",
                call_id="call_1",
                name="calculator",
                arguments=json.dumps({"expression": "21 * 2"}),
                type="function_call",
            )
        elif step == 1:
            item = ResponseFunctionToolCall(
                id="fc_2",
                call_id="call_2",
                name="fake_search",
                arguments=json.dumps({"query": "OpenAI Agents SDK"}),
                type="function_call",
            )
        else:
            item = ResponseOutputMessage(
                id="msg_1",
                role="assistant",
                status="completed",
                type="message",
                content=[
                    ResponseOutputText(
                        text="21 * 2 = 42. The OpenAI Agents SDK runs agents with tools.",
                        type="output_text",
                        annotations=[],
                    )
                ],
            )

        return ModelResponse(output=[item], usage=usage, response_id=f"resp_{step}")

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("ScriptedModel is non-streaming (explore fixture).")


def build_agent() -> Agent:
    return Agent(
        name="sample-openai-agent",
        instructions="Answer the user's question, using tools when helpful.",
        tools=[calculator, fake_search],
        model=ScriptedModel(),
    )


def run():
    """Run the single task end-to-end and return the SDK's RunResult."""
    return Runner.run_sync(build_agent(), TASK)


if __name__ == "__main__":
    result = run()
    print("=== type ===")
    print(type(result))
    print("\n=== final_output ===")
    print(repr(result.final_output))

    print(f"\n=== new_items ({len(result.new_items)}) ===")
    for i, item in enumerate(result.new_items):
        print(f"\n[{i}] RunItem={type(item).__name__} (.type={getattr(item, 'type', None)!r})")
        raw = getattr(item, "raw_item", None)
        print(f"    raw_item type = {type(raw).__name__}")
        for attr in ("name", "arguments", "call_id", "output", "content", "role"):
            if hasattr(raw, attr):
                print(f"    raw_item.{attr} = {getattr(raw, attr)!r}")
            elif isinstance(raw, dict) and attr in raw:
                print(f"    raw_item[{attr!r}] = {raw[attr]!r}")

    print(f"\n=== raw_responses ({len(result.raw_responses)}) ===")
    for i, resp in enumerate(result.raw_responses):
        print(f"[{i}] response_id={resp.response_id!r} usage="
              f"(in={resp.usage.input_tokens}, out={resp.usage.output_tokens}, "
              f"total={resp.usage.total_tokens}, requests={resp.usage.requests})")

    print("\n=== to_input_list() (flat conversation) ===")
    for i, item in enumerate(result.to_input_list()):
        if isinstance(item, dict):
            kind = item.get("type") or item.get("role")
        else:
            kind = type(item).__name__
        print(f"[{i}] {kind}")
