# Scaffold prompt for AgentTrace

Paste the block below into Claude Code (or any coding agent) from an empty folder to generate the initial repo. It only builds the skeleton + the real core schema — not the whole tool at once, on purpose. After it runs, you'll flesh out one adapter end-to-end before anything else.

---

You are scaffolding a Python project called **AgentTrace**: a framework-agnostic trajectory evaluation and regression harness for LLM agents. A `CLAUDE.md` in this folder describes the full design — read it first and follow it exactly, especially the normalization rule (nothing outside `adapters/` may know which framework produced a trajectory).

Build ONLY the scaffold described below. Do not implement adapters, evaluators, the judge, or benchmarks beyond typed stubs that raise `NotImplementedError` — except `schema.py`, which must be fully implemented and tested. Keep the framework boundary clean.

### 1. Project setup
- `pyproject.toml`: installable package (`agenttrace`), Python 3.11+, deps: `pydantic>=2`, `typer` (CLI), `rich`; dev deps: `pytest`, `ruff`. Console script `agenttrace = "agenttrace.cli:app"`.
- `README.md`: one-paragraph description, the pipeline ASCII diagram from CLAUDE.md, a "Status: scaffold" note, and a build-order checklist (schema ✅ → LangGraph adapter → evaluators → judge+calibration → more adapters → CI).
- `.gitignore` (Python), `ruff.toml` (sensible defaults, line length 100).

### 2. Fully implement `src/agenttrace/schema.py` (Pydantic v2)
- `Step`: `index:int, role:str, tool_name:str|None, tool_args:dict|None, tool_output:str|None, tokens:int|None, latency_ms:float|None, timestamp:datetime`.
- `Outcome`: `success:bool, final_answer:str|None, ground_truth:str|None, error:str|None`.
- `RunConfig`: the reproducibility tuple — `agent_id:str, task_set:str, harness_config:dict, judge:str|None, budget:dict` (e.g. max_steps, max_tokens, max_usd). Add a `fingerprint()` method returning a stable hash of the config so two runs can be checked for comparability.
- `Trajectory`: `config:RunConfig, steps:list[Step], outcome:Outcome, total_tokens:int, total_usd:float, wall_clock_s:float`. Add `to_audit_record()` → dict suitable for JSON, and `write_audit(path)` that dumps it. Redact any `tool_args` values whose key matches a credential-like pattern (token, key, secret, password) before writing.
- `EvalResult`: `evaluator:str, score:float, passed:bool, detail:dict`.
- Align field names with OpenTelemetry GenAI span/tool-call attribute names where a natural match exists; add a short comment noting the mapping.

### 3. Stubs (typed, documented, raise `NotImplementedError`)
- `adapters/base.py`: `Adapter` ABC with `to_trajectory(native_trace, config) -> Trajectory`.
- `adapters/langgraph.py`, `adapters/openai_sdk.py`, `adapters/crewai.py`: subclasses, stubbed.
- `evaluators/base.py`: `Evaluator` ABC with `evaluate(trajectory) -> EvalResult`. Plus a simple registry (`register`/`get_all`).
- `evaluators/outcome.py`, `tool_call.py`, `efficiency.py`, `cost_latency.py`, `policy.py`: stubbed subclasses, each registered.
- `judge/llm_judge.py` and `judge/calibration.py` (Cohen's κ): stubbed.
- `benchmarks/agentdojo.py`: stubbed loader `load_tasks() -> list[...]`.
- `report/report.py`: stubbed `render(results)` and `regression_diff(old, new)`.

### 4. CLI (`cli.py`, Typer)
Three commands, wired to the stubs so `--help` works and the structure is visible:
- `agenttrace run` — run an agent through an adapter + evaluators (stub).
- `agenttrace report` — render the latest results (stub).
- `agenttrace ci` — run evals and exit non-zero on regression (stub).

### 5. Tests + CI
- `tests/test_schema.py`: real tests for `schema.py` — round-trip a `Trajectory` through `to_audit_record()`, assert credential redaction works, assert `RunConfig.fingerprint()` is stable and changes when config changes.
- `.github/workflows/eval.yml`: install, run `ruff check` and `pytest`. Leave a commented placeholder for the future regression gate.

### Finish by
- Running `ruff check` and `pytest` and making them pass.
- Printing the resulting file tree and a short "next step: implement the LangGraph adapter end-to-end against schema.py" note.

Do not add features beyond this. Stop after the scaffold is green.
