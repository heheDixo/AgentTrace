"""A tiny, deterministic LangGraph agent used as a fixture for building the adapter.

It is deliberately LLM-free: instead of an API-backed chat model, a scripted ``agent`` node emits
a fixed sequence of tool calls and then a final answer. That keeps the run fully deterministic (no
API key, no sampling) while still producing a *genuine* LangGraph message trace — real
``AIMessage`` tool calls routed through a real ``ToolNode`` and answered with ``ToolMessage``s.
That trace is exactly the native shape the LangGraph adapter will normalize.

Run directly to see the raw result LangGraph returns::

    python examples/sample_langgraph_agent.py
"""

from __future__ import annotations

import ast
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# The single task this fixture runs. Chosen so the agent must use both tools, in order.
TASK = "What is 21 * 2, and what is LangGraph?"

# --- Tools ---------------------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a tiny arithmetic AST (numbers and + - * / ** only)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '21 * 2'."""
    return str(_safe_eval(ast.parse(expression, mode="eval")))


@tool
def fake_search(query: str) -> str:
    """Return a canned, deterministic search snippet for a query."""
    snippets = {
        "langgraph": "LangGraph is a library for building stateful, multi-step agents as graphs.",
    }
    return snippets.get(query.strip().lower(), f"No results for {query!r}.")


TOOLS = [calculator, fake_search]


# --- Graph ---------------------------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent(state: State) -> dict:
    """Scripted 'model': decide the next action from which tools have already answered."""
    answered = {m.name for m in state["messages"] if isinstance(m, ToolMessage)}
    if "calculator" not in answered:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "args": {"expression": "21 * 2"}, "id": "call_1"}
                    ],
                )
            ]
        }
    if "fake_search" not in answered:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "fake_search", "args": {"query": "LangGraph"}, "id": "call_2"}
                    ],
                )
            ]
        }
    return {"messages": [AIMessage(content="21 * 2 = 42. LangGraph is a library for agents.")]}


def route(state: State) -> str:
    """Go to the tools node while the last message has tool calls, else finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    """Compile and return the deterministic agent graph."""
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run() -> dict:
    """Run the single task end-to-end and return LangGraph's full final state."""
    return build_graph().invoke({"messages": [HumanMessage(content=TASK)]})


if __name__ == "__main__":
    result = run()
    print("=== type ===")
    print(type(result))
    print("\n=== keys ===")
    print(list(result.keys()))
    print(f"\n=== messages ({len(result['messages'])}) ===")
    for i, m in enumerate(result["messages"]):
        print(f"\n[{i}] {type(m).__name__}")
        print(f"    content   = {m.content!r}")
        tc = getattr(m, "tool_calls", None)
        if tc:
            print(f"    tool_calls= {tc}")
        if isinstance(m, ToolMessage):
            print(f"    name      = {m.name}")
            print(f"    tool_call_id = {m.tool_call_id}")
        if getattr(m, "response_metadata", None):
            print(f"    response_metadata = {m.response_metadata}")
        if getattr(m, "usage_metadata", None):
            print(f"    usage_metadata    = {m.usage_metadata}")
