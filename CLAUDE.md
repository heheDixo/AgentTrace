# CLAUDE.md — AgentTrace

Context for any coding agent working in this repo. Read this first, every session.

## What this project is

**AgentTrace** is a **framework-agnostic trajectory evaluation and regression harness for LLM agents.**

It takes any agent (built in LangGraph, deepagents, the OpenAI Agents SDK, CrewAI, …), captures its full run as a normalized *trajectory*, runs a suite of *evaluators* over that trajectory, and gates regressions in CI. The goal is to move an agent from "it works" to "it scores X% success, p95 latency Ys, $Z/run, and this commit regressed tool-call correctness by 3%."

It is deliberately an *engineering + evaluation* tool, not an attack tool. Security evaluation here means measuring an agent's robustness and policy-compliance on public benchmarks — never producing working exploits.

## The one idea that matters: normalization

Every framework emits traces in its own shape. AgentTrace never lets a framework's format leak past the edge. The pipeline is:

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

**Rule:** evaluators, the judge, benchmarks, and reporting must depend only on `schema.py`. They must never import or branch on a specific framework. Framework-specific code lives exclusively in `adapters/`. Adding a new framework = writing one new adapter, nothing else.

## Core schema (schema.py)

Two levels, both as typed models (Pydantic v2):

- **`RunConfig`** — the reproducibility tuple `⟨agent_id, task_set, harness_config, judge, budget⟩`. This captures *everything needed to reproduce a run*. Inspired by the auditable `⟨M,A,D,S,J⟩` tuple from the PromptSecurity SoK (arXiv:2510.15476): a result is only comparable to another when both configs match.
- **`Trajectory`** — `RunConfig` + an ordered list of `Step` + a final `Outcome` + run-level metrics (total tokens, cost, wall-clock).
  - **`Step`** — one agent action: `{index, role, tool_name, tool_args, tool_output, tokens, latency_ms, timestamp}`.
  - **`Outcome`** — `{success: bool, final_answer, ground_truth, error}`.

Every run also writes a complete **audit record** to disk (prompts, outputs, judge decisions, settings, token/call counts, timestamps). Audit records are first-class — they're what make a result checkable.

**Align the schema with OpenTelemetry GenAI/agent semantic conventions** wherever possible (span names, tool-call attributes). Many frameworks already emit OTel; matching it keeps adapters thin and signals that we follow an industry standard rather than inventing a bespoke format.

## Evaluators (evaluators/)

Each evaluator is a class implementing a common `Evaluator` interface: `evaluate(trajectory) -> EvalResult`. The initial five:

1. **outcome** — task success vs ground truth (or via the judge when no ground truth).
2. **tool_call** — right tool chosen, arguments valid against the tool's schema.
3. **efficiency** — step count vs a reference trajectory (flags wasteful agents).
4. **cost_latency** — tokens, $/run, p50/p95 latency.
5. **policy** — OWASP ASI (Agentic Security Initiative) compliance checks on the trajectory. This is the security angle; keep it to detection/measurement, not exploitation.

## Judge (judge/)

- **`llm_judge.py`** — LLM-as-judge for outcomes without hard ground truth.
- **`calibration.py`** — measures judge↔human agreement (Cohen's κ) on a hand-labelled set of 50–100 trajectories. The SoK paper shows judge choice materially changes measured results, so calibration is not optional — it's a headline feature and a potential paper result. Never trust an uncalibrated judge's numbers.

## Benchmarks (benchmarks/)

- **`agentdojo.py`** — integrate AgentDojo (97 user tasks, 629 security test cases, 4 domains) as a built-in task source. This gives security + utility evaluation out of the box and strong credibility. Treat AgentDojo's own injection tasks as *fixtures to measure against*, not payloads to extend.

## Tech stack

- Python 3.11+, typed throughout, Pydantic v2 for all schema models.
- Adapters target: LangGraph (+deepagents) first, then OpenAI Agents SDK, then CrewAI.
- Tracing/observability: compatible with Langfuse and LangSmith; OTel as the lingua franca.
- Tests: pytest. Lint/format: ruff. Packaging: a real `pyproject.toml` (installable, `pip install -e .`).
- **Do not** take AutoGen as a target (maintenance mode). CrewAI is supported mainly as a breadth/comparison adapter.

## Repo layout

```
src/agenttrace/
  schema.py              # RunConfig, Trajectory, Step, Outcome, EvalResult
  adapters/              # base.py (ABC) + one file per framework
  evaluators/            # base.py (ABC) + outcome/tool_call/efficiency/cost_latency/policy
  judge/                 # llm_judge.py, calibration.py
  benchmarks/            # agentdojo.py
  report/                # report.py (human-readable + JSON), regression diff
  cli.py                 # `agenttrace run`, `agenttrace report`, `agenttrace ci`
tests/
.github/workflows/eval.yml   # CI: run evals, fail PR on regression
```

## Conventions for any code written here

- Keep the framework boundary clean: nothing outside `adapters/` knows which framework produced a trajectory.
- Prefer small, composable classes behind the `Evaluator` / `Adapter` ABCs. New capability = new class, registered — not a new `if` branch.
- Everything that affects a result (model, decoding, budget, judge, dataset sample id) goes into `RunConfig` and the audit record.
- Metrics report with uncertainty where it's cheap to (run k trials, report pass@k and pass^k; pass^k = all k runs succeed, the reliability metric).
- Build in stages: schema → one adapter end-to-end → evaluators → judge+calibration → more adapters → CI. Don't scaffold everything half-working at once.
- No fabricated numbers in README or docs. Every metric shown must come from a real run with its audit record.

## Non-goals / guardrails

- Not an exploit generator. The `policy` evaluator and AgentDojo integration *measure* robustness; they do not craft working attacks.
- Keep secrets out of the repo and out of audit records (redact tool args that look like credentials).
