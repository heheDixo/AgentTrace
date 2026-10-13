---
description: Scaffold a new framework adapter (stub + test) against the Adapter ABC.
argument-hint: <framework_name>  e.g. /new-adapter autogen
---

Scaffold a new AgentTrace framework adapter for the framework named **`$ARGUMENTS`**.

## Argument

- `$ARGUMENTS` — the framework's short, snake_case name (e.g. `crewai`, `autogen`,
  `llama_index`). This becomes the module filename, the `framework` class attribute, and the test
  filename. If no argument was given, stop and ask the user for the framework name.

## Rules (read `CLAUDE.md` first)

- The adapter is the **only** place allowed to know this framework's native shape. Everything
  outside `adapters/` depends only on `schema.py` — do not touch the schema, evaluators, judge,
  benchmarks, or report.
- Follow the **exact pattern** of the existing adapters:
  `src/agenttrace/adapters/langgraph.py` and `src/agenttrace/adapters/openai_sdk.py` (full
  implementations), and `src/agenttrace/adapters/crewai.py` (the stub shape). Subclass `Adapter`
  from `src/agenttrace/adapters/base.py` and register it.
- The house pattern is captured in the **adapter-authoring** skill (bundled in this plugin); follow
  it.
- Scaffold a **stub only** — `to_trajectory` raises `NotImplementedError`. Do not implement the
  normalization yet; that is a deliberate separate step so the mapping gets real attention.

## Steps

1. Derive `NAME = "$ARGUMENTS"` (snake_case) and a PascalCase class name
   (e.g. `llama_index` → `LlamaIndexAdapter`, appending `Adapter`).

2. Create `src/agenttrace/adapters/<NAME>.py` following this shape (mirror the existing stubs):

   ```python
   """<Framework> adapter: native <Framework> run -> normalized Trajectory. STUB.

   This is the ONLY module allowed to know <Framework>'s native shape. Implement `to_trajectory`
   to pair tool calls with their outputs into Steps, take the final answer as the Outcome, and
   sum usage into total_tokens — mirroring adapters/langgraph.py and adapters/openai_sdk.py.
   """

   from __future__ import annotations

   from typing import Any

   from agenttrace.adapters.base import Adapter, register
   from agenttrace.schema import RunConfig, Trajectory


   @register
   class <ClassName>Adapter(Adapter):
       """Normalize a <Framework> run into a :class:`Trajectory`."""

       framework = "<NAME>"

       def to_trajectory(self, native_trace: Any, config: RunConfig) -> Trajectory:
           raise NotImplementedError("<Framework> adapter not implemented yet.")
   ```

3. Register it for discovery by adding `<NAME>` to the submodule imports in
   `src/agenttrace/adapters/__init__.py` (keep the list sorted, matching how `crewai`, `langgraph`,
   `openai_sdk` are imported there).

4. Create `tests/test_<NAME>_adapter.py` mirroring `tests/test_langgraph_adapter.py`: a test that
   the adapter is registered and discoverable via `agenttrace.adapters.get("<NAME>")`, and a test
   asserting `to_trajectory(...)` raises `NotImplementedError` for now (so the stub is pinned until
   implemented). Add a `TODO` comment pointing at the real end-to-end test to write once the
   mapping lands.

5. Run `ruff check .` and `pytest -q` and make them pass.

6. Print the two new file paths and a one-line next step: "explore the framework's native trace
   shape (write a tiny sample agent), then implement `to_trajectory` against `schema.py`."
