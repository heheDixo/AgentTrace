"""AgentDojo task source (97 user tasks, 629 security test cases, 4 domains).

AgentDojo's injection tasks are treated strictly as *fixtures to measure against*, never payloads
to extend: this module only *reads* their declared goals and ground-truth call sequences to build
normalized task specs. It never synthesizes, mutates, or amplifies an attack string.

This module is allowed to import ``agentdojo`` (a benchmark dataset, not an agent framework). It
produces only :mod:`agenttrace.schema`-compatible data: normalized task specs, a JSON-safe
"benchmark block" for ``RunConfig.harness_config``, and reference :class:`Trajectory` objects. The
benchmark block is what lets the framework-agnostic ``policy`` evaluator score a run without ever
importing agentdojo itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agenttrace.schema import Outcome, RunConfig, Step, Trajectory

_VERSION = "v1"


class ToolSpec(BaseModel):
    """A tool available to the agent for a task."""

    name: str
    parameters: list[str] = Field(default_factory=list)


class RefCall(BaseModel):
    """A normalized reference tool call: a function name + its arguments."""

    function: str
    args: dict[str, Any] = Field(default_factory=dict)


class InjectionMeta(BaseModel):
    """An injection task: the attacker's goal and the call(s) a successful attack would make."""

    injection_task_id: str
    goal: str
    calls: list[RefCall] = Field(default_factory=list)


class BenchmarkTask(BaseModel):
    """A normalized AgentDojo task spec our harness can run and score."""

    source: str = "agentdojo"
    suite: str
    user_task_id: str
    goal: str
    tools: list[ToolSpec] = Field(default_factory=list)
    ground_truth: list[RefCall] = Field(default_factory=list)
    # The suite's injection catalog (fixtures to measure against); empty if the suite has none.
    injections: list[InjectionMeta] = Field(default_factory=list)


def _normalize_calls(calls: Any) -> list[RefCall]:
    return [RefCall(function=c.function, args=dict(c.args)) for c in calls]


def _tool_specs(suite: Any) -> list[ToolSpec]:
    specs = []
    for tool in suite.tools:
        props = tool.parameters.model_json_schema().get("properties", {})
        specs.append(ToolSpec(name=tool.name, parameters=list(props.keys())))
    return specs


def load_tasks(suite: str, *, version: str = _VERSION) -> list[BenchmarkTask]:
    """Load a suite (e.g. ``"banking"``) as a list of normalized :class:`BenchmarkTask` specs."""
    from agentdojo.task_suite.load_suites import get_suite

    s = get_suite(version, suite)
    env = s.load_and_inject_default_environment({})

    tools = _tool_specs(s)
    injections = [
        InjectionMeta(
            injection_task_id=it.ID,
            goal=it.GOAL,
            calls=_normalize_calls(it.ground_truth(env)),
        )
        for it in s.injection_tasks.values()
    ]

    tasks = []
    for ut in s.user_tasks.values():
        tasks.append(
            BenchmarkTask(
                suite=suite,
                user_task_id=ut.ID,
                goal=ut.PROMPT,
                tools=tools,
                ground_truth=_normalize_calls(ut.ground_truth(env)),
                injections=injections,
            )
        )
    return tasks


def benchmark_block(
    task: BenchmarkTask, injection: InjectionMeta | None = None
) -> dict[str, Any]:
    """Build the JSON-safe benchmark block the ``policy`` evaluator reads from harness_config.

    ``utility_target`` is the user task's ground-truth call sequence; ``attack_target`` (optional)
    is the injection task's attacker call sequence. The evaluator matches these against the run's
    actual steps — it needs nothing from agentdojo itself.
    """
    block: dict[str, Any] = {
        "source": "agentdojo",
        "suite": task.suite,
        "user_task_id": task.user_task_id,
        "utility_target": [c.model_dump() for c in task.ground_truth],
        "attack_target": None,
    }
    if injection is not None:
        block["attack_target"] = {
            "injection_task_id": injection.injection_task_id,
            "goal": injection.goal,
            "calls": [c.model_dump() for c in injection.calls],
        }
    return block


def build_reference_trajectory(
    suite: str,
    user_task_id: str,
    *,
    injection_task_id: str | None = None,
    agent_follows_injection: bool = False,
    version: str = _VERSION,
) -> Trajectory:
    """Build a deterministic reference (oracle) :class:`Trajectory` for a task.

    Simulates an agent by replaying the user task's ground-truth tool calls. When
    ``agent_follows_injection`` is set, it additionally replays the injection task's attacker
    call(s) — modelling a *compromised* agent, so the policy evaluator can be measured against a
    known-bad run. No LLM, no network; it exists to exercise the loop deterministically.
    """
    from agentdojo.task_suite.load_suites import get_suite

    s = get_suite(version, suite)
    env = s.load_and_inject_default_environment({})
    ut = s.user_tasks[user_task_id]

    benign = _normalize_calls(ut.ground_truth(env))

    injection = None
    attack_calls: list[RefCall] = []
    if injection_task_id is not None:
        it = s.injection_tasks[injection_task_id]
        injection = InjectionMeta(
            injection_task_id=it.ID,
            goal=it.GOAL,
            calls=_normalize_calls(it.ground_truth(env)),
        )
        attack_calls = injection.calls

    sequence = list(benign)
    if agent_follows_injection:
        sequence = sequence + attack_calls

    ts = datetime.now(UTC)
    steps = [
        Step(
            index=i,
            role="assistant",
            tool_name=c.function,
            tool_args=c.args,
            tool_output=None,
            tokens=None,
            latency_ms=None,
            timestamp=ts,
        )
        for i, c in enumerate(sequence)
    ]

    task = BenchmarkTask(
        suite=suite,
        user_task_id=user_task_id,
        goal=ut.PROMPT,
        tools=_tool_specs(s),
        ground_truth=benign,
        injections=[injection] if injection else [],
    )
    config = RunConfig(
        agent_id=f"agentdojo-{suite}-{user_task_id}",
        task_set=f"agentdojo/{suite}",
        harness_config={
            "framework": "agentdojo-reference",
            "task": ut.PROMPT,
            "benchmark": benchmark_block(task, injection),
        },
    )
    return Trajectory(
        config=config,
        steps=steps,
        outcome=Outcome(success=True, final_answer=None, ground_truth=None, error=None),
        total_tokens=0,
        total_usd=0.0,
        wall_clock_s=0.0,
    )
