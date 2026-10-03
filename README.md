# AgentTrace

A **framework-agnostic trajectory evaluation and regression harness for LLM agents.** It takes
any agent (LangGraph, deepagents, the OpenAI Agents SDK, CrewAI, …), captures its full run as a
normalized *trajectory*, runs a suite of *evaluators* over that trajectory, and gates regressions
in CI — moving an agent from "it works" to "it scores X% success, p95 latency Ys, $Z/run, and this
commit regressed tool-call correctness by 3%."

## The pipeline

```
Agent (any framework)
   │ native trace
   ▼
Adapter  (native → normalized)   ← one small adapter per framework
   ▼
Normalized Trajectory            ← the common schema; the heart of the tool
   ▼
Evaluators · Judge · Benchmarks  ← all operate ONLY on the normalized schema
   ▼
Report + CI gate
```

**Normalization rule:** evaluators, the judge, benchmarks, and reporting depend only on
`schema.py`. They never import or branch on a specific framework. Framework-specific code lives
exclusively in `adapters/`. Adding a new framework = writing one new adapter, nothing else.

## Status: scaffold

Only the core schema is fully implemented and tested. Everything else is a typed stub that raises
`NotImplementedError`. Build order:

- [x] `schema.py` — `RunConfig`, `Trajectory`, `Step`, `Outcome`, `EvalResult` (+ audit record,
      credential redaction, config fingerprint)
- [ ] LangGraph adapter (end-to-end, against `schema.py`)
- [ ] Evaluators (outcome, tool_call, efficiency, cost_latency, policy)
- [ ] Judge + calibration (Cohen's κ)
- [ ] More adapters (OpenAI Agents SDK, CrewAI)
- [ ] CI regression gate

## Install

```bash
pip install -e ".[dev]"
```

## Develop

```bash
ruff check .
pytest
```
