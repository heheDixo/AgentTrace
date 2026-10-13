---
name: adapter-authoring
description: The house pattern for writing an AgentTrace framework adapter. Use when creating or implementing any adapter under src/agenttrace/adapters/ — turning a framework's native run (LangGraph, OpenAI Agents SDK, CrewAI, …) into a normalized Trajectory. Complements the /new-adapter scaffolder.
---

# Writing an AgentTrace adapter the house way

An adapter converts one framework's **native trace** into a normalized
`agenttrace.schema.Trajectory`. It is the ONLY place in the codebase allowed to know a framework's
native shape. Read `CLAUDE.md` first; the normalization rule is non-negotiable.

## The boundary (do not break it)

- Everything outside `adapters/` — the evaluators, judge, benchmarks, report — depends ONLY on
  `schema.py`. Your adapter must **not** make anything outside `adapters/` import or branch on a
  framework.
- Do not edit `schema.py` to make an adapter fit. If the normalized schema genuinely cannot
  represent something, stop and raise it with a maintainer — that is a schema decision, not an
  adapter workaround. (A hook in this repo makes editing `schema.py` require confirmation.)
- Prefer **duck-typed accessors** over importing the framework SDK at module load, so the adapter
  module imports cleanly even when that SDK is absent and works on both live objects and
  dict-serialized/replayed traces. See `adapters/langgraph.py` and `adapters/openai_sdk.py`.

## Registration

Subclass `Adapter` (`adapters/base.py`), set `framework = "<name>"`, and decorate with
`@register`. Add `<name>` to the sorted submodule imports in `adapters/__init__.py` so importing
the package registers it. Callers resolve it by string via `agenttrace.adapters.get("<name>")` —
never by importing the class.

## Mapping native → Trajectory

1. **Steps = agent actions, one per tool call.** Pair each tool *call* (tool name + arguments)
   with its *result* by the framework's call id (LangGraph: `tool_call_id`; OpenAI SDK: `call_id`).
   Each becomes a `Step(index, role, tool_name, tool_args, tool_output, tokens, latency_ms,
   timestamp)`.
2. **`tool_args` must be a dict or `None`.** If the framework gives arguments as a JSON string
   (OpenAI SDK), `json.loads` it. If parsing fails or the result is not an object, use `None` so
   the `tool_call` evaluator honestly flags the defect — never hide it behind an empty dict.
3. **Outcome from the end state.** Take the final assistant answer as `Outcome.final_answer`
   (`result.final_output`, or the last text message). `ground_truth` stays `None` in a raw trace;
   downstream evaluators/judge handle correctness. Set `error` if the trace signals a tool error.
4. **Fill what the framework exposes; leave the rest `None`.** Sum per-turn `usage` into
   `total_tokens` when available (OpenAI SDK exposes it; an LLM-free fixture may not). `total_usd`
   is `0.0` when not derivable. `latency_ms` / per-step timestamps are usually absent from a
   non-streaming `invoke`/`run` result — leave `latency_ms=None` and stamp `Step.timestamp` with
   one ingest time rather than inventing per-step ordering.

## Pin it with a test

Mirror `tests/test_langgraph_adapter.py` / `tests/test_openai_adapter.py`: feed a **captured real
trace** (run a tiny deterministic sample agent once) through the adapter and assert the normalized
shape — step count/order, paired tool outputs, the final answer, token totals, and that the audit
record round-trips. A freshly scaffolded stub gets a registration test plus a `NotImplementedError`
test until the mapping lands.

## Workflow

Explore first (write a tiny sample agent, print the raw native shape), then implement
`to_trajectory` against `schema.py`, then test. `/new-adapter <name>` scaffolds the stub + test to
start from.
