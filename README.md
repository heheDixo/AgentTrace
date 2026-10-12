# AgentTrace

**A framework-agnostic trajectory evaluation and regression harness for LLM agents.** Point it at
any agent (LangGraph, the OpenAI Agents SDK, CrewAI, …), capture its run as a normalized
*trajectory*, score that trajectory with a suite of evaluators, and gate regressions in CI.

## The problem

Most agents ship on vibes. "It worked when I tried it" is not a number, and the next commit
silently regresses tool-call correctness or doubles the token bill with nobody the wiser.
AgentTrace turns an agent run into **checkable numbers** — task success, tool-call validity, step
efficiency, cost/latency, and prompt-injection robustness — writes a full **audit record** for
every run, and **fails CI on a regression**. It moves an agent from "it works" to "it scores X%
success at $Z/run, and this PR dropped efficiency from 1.00 to 0.50."

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

## Why it's designed this way: the normalization rule

Every framework emits traces in its own shape. AgentTrace converts each into **one common schema**
([`schema.py`](src/agenttrace/schema.py): `RunConfig`, `Trajectory`, `Step`, `Outcome`,
`EvalResult`) at the edge, and nothing downstream is allowed to see past it.

> **Evaluators, the judge, benchmarks, and reporting depend only on `schema.py`. They never import
> or branch on a specific framework. Framework-specific code lives exclusively in `adapters/`.
> Adding a new framework = writing one new adapter, nothing else.**

This isn't aspirational — it's load-bearing, and we tested it. **Adding the second adapter (the
OpenAI Agents SDK) required zero changes outside `adapters/`:** only
[`adapters/openai_sdk.py`](src/agenttrace/adapters/openai_sdk.py) and its test were written. The
schema, all five evaluators, the judge, and the report were untouched, and the *same* evaluators
then scored both agents. The one genuine difference between the frameworks — the OpenAI SDK exposes
per-response token usage while our LLM-free LangGraph fixture does not — surfaced exactly where it
should, as `total_tokens=75` vs `total_tokens=0` in the identical `cost_latency` evaluator, with no
evaluator code aware of which framework produced the trajectory.

Field names follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
where a natural match exists, so adapters stay thin.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.11+ (developed on 3.14). Pydantic v2 throughout.

## Quickstart

```bash
# 1. Run an agent through an adapter + evaluators; writes an audit record + results JSON.
agenttrace run --adapter langgraph               # built-in deterministic sample agent
agenttrace run --adapter openai_sdk --trace run.json

# 2. Render the latest run's results.
agenttrace report

# 3. Benchmark robustness + utility on AgentDojo (benign + injected cases).
agenttrace bench --suite banking

# 4. Gate regressions: establishes a baseline, then exits non-zero if any evaluator regresses.
agenttrace ci --baseline baselines/main.json
```

## What a real run produces

> **Provenance.** Every number below is copied from a real run of the commands above on this
> commit — each is backed by an audit/results record on disk. The two sample agents and the
> AgentDojo "oracle" agent are **deterministic, LLM-free fixtures** (they replay scripted / ground-truth
> tool calls), so these are honest measurements of *the harness*, not performance claims about a
> production model. Swap in a real agent and the same evaluators produce the same shape of numbers.

### Per-run evaluators (`agenttrace run`)

Scored on the built-in samples. `cost_latency`, `tool_call`, `efficiency` run on every trajectory;
`outcome` needs a ground truth or a judge; `policy` needs a benchmark task:

| Evaluator      | LangGraph sample | OpenAI SDK sample | What it measures |
|----------------|------------------|-------------------|------------------|
| `cost_latency` | 1.000 — `total_tokens=0` | 1.000 — `total_tokens=75` | tokens, $/run, p50/p95 latency vs budget |
| `tool_call`    | 1.000 — `2/2 valid` | 1.000 — `2/2 valid` | right tool, args present & well-typed |
| `efficiency`   | 1.000 — `n_steps=2` | 1.000 — `n_steps=2` | step count vs a reference |
| `outcome`      | *skipped* (no ground truth / judge) | *skipped* | task success vs ground truth or judge |
| `policy`       | *skipped* (no benchmark task) | *skipped* | prompt-injection robustness |

`outcome`, run on a trajectory that *does* carry a ground truth, scores `1.000` (`method=ground_truth`,
`final_answer="21 * 2 = 42."` matched `ground_truth="42"`).

### Security + utility on AgentDojo (`agenttrace bench --suite banking`)

Full banking suite, deterministic oracle agent, k=1 — **16 benign cases, 144 injected cases**:

| Agent | task utility (benign) | attack-success-rate (injected) | utility pass^k | security pass^k |
|-------|-----------------------|--------------------------------|----------------|-----------------|
| oracle (benign)       | 1.000 | 0.021 | 1.000 | 0.979 |
| oracle (compromised)  | 1.000 | 1.000 | 1.000 | 0.000 |

One run yields **both** a security number (attack-success-rate) and a utility number
(task-utility). The compromised oracle (which deliberately follows every injection) is caught at
`1.000` attack-success-rate; the benign oracle resists. The benign oracle's non-zero `0.021` is a
measurement artifact, discussed under **Limitations**.

### CI regression gate (`agenttrace ci`)

Establishing a baseline exits `0`; a run that drops a score past the threshold (default 0.03) exits
non-zero. Observed on this commit: tightening the efficiency reference regressed `efficiency` from
**1.000 → 0.500** (delta −0.500), and `agenttrace ci` **exited 1** (a clean re-run exited 0).

*Repo state: 67 tests passing, `ruff` clean.*

## Design decisions & trade-offs

- **One schema, enforced.** The normalization rule is the whole architecture. New capability = a new
  registered `Adapter` / `Evaluator` subclass, never a new `if framework == …` branch. The cost is
  an up-front mapping per framework; the payoff is that evaluators, benchmarks, and the CI gate are
  written once and work everywhere.
- **Pluggable everything, deterministic by default.** The judge takes an injected `complete`
  callable (no vendor hard-wired); the benchmark runner takes a pluggable agent. The samples and the
  AgentDojo oracle are LLM-free, so the whole suite + CI run with **no API key** and are fully
  reproducible. The seam for a real LLM agent is the same interface.
- **Auditability is first-class.** Every run writes a complete, JSON-safe audit record, and
  credential-like `tool_args` (token/key/secret/password/…) are redacted before anything hits disk.
- **Reproducibility tuple.** `RunConfig` captures `⟨agent_id, task_set, harness_config, judge,
  budget⟩` and `fingerprint()`s to a stable hash, so two results are only compared when their
  configs match.
- **Detection, not exploitation.** The security angle *measures* robustness on public fixtures; it
  never synthesizes or extends attack payloads.

## Limitations

These are real and worth stating plainly:

- **Policy scoring matches declared ground-truth call signatures against the trajectory's steps — it
  does not run AgentDojo's own live `utility()` / `security()` callbacks.** Those need the live
  environment and an `agentdojo` import, which would breach the evaluator's framework-agnostic
  boundary. Consequence: the benign oracle shows `attack-success-rate=0.021` (≈3 of 144 cases)
  because a few user tasks' *benign* `send_money` ground truth coincidentally satisfies an attacker
  call's matched args — a **false positive**, not a real attack. Tightening the matcher (exact
  distinctive-arg match, or per-call "key args" in the benchmark block) removes these; exact
  AgentDojo verdicts belong in a benchmark *runner* that may call those callbacks.
- **For a real LLM agent, signature matching may under-count utility** when the agent formats
  arguments differently from the reference (e.g. a reworded `subject`). Same fix: per-call key args.
- **Per-run vs aggregate.** `attack_success` and `task_utility` are per-run 0/1; headline rates are
  means the benchmark runner aggregates across cases. `pass^k` (all k repeats succeed) is the
  reliability metric, but with a deterministic oracle `pass^k == pass@1` — its value shows with a
  real, stochastic agent.
- **The judge is only as good as its calibration.** `judge/calibration.py` computes Cohen's κ
  against human labels precisely because judge choice changes measured results; **never trust an
  uncalibrated judge's numbers.** The judge also currently sees `harness_config["task"]` or the
  `task_set` id rather than a first-class task field — worth closing before a real calibration run.
- **Numbers here are from deterministic fixtures**, by design (reproducible docs/CI). They validate
  the harness, not any model.

## Threat model: policy evaluator ↔ OWASP ASI

The `policy` evaluator measures resistance to **indirect prompt injection**: an attacker instruction
embedded in tool-returned data (a bill file, an incoming transaction memo) that tries to steer the
agent into an unauthorized action. Mapped to the
[OWASP Agentic Security Initiative (ASI)](https://genai.owasp.org/initiatives/#agentic-security-initiative)
threat taxonomy (indicative):

| Threat | ID | How AgentTrace relates |
|--------|----|------------------------|
| Tool Misuse | ASI **T2** | We detect when an injected instruction causes an unintended tool call (e.g. `send_money` to an attacker IBAN) — the core `attack_success` signal. |
| Intent Breaking & Goal Manipulation | ASI **T6** | The injection diverts the agent from the user's goal; measured as attack-success alongside preserved task-utility. |
| Repudiation & Untraceability | ASI **T8** | Mitigated structurally: every run emits a full, replayable audit record. |

Upstream vector corresponds to **OWASP LLM Top-10 `LLM01` (Prompt Injection)**. AgentDojo injection
tasks are used strictly as **fixtures to measure against**, never as payloads to craft or extend.

## Repo layout

```
src/agenttrace/
  schema.py              # RunConfig, Trajectory, Step, Outcome, EvalResult (the boundary)
  adapters/              # base + langgraph, openai_sdk (real); crewai (stub)
  evaluators/            # base + outcome, tool_call, efficiency, cost_latency, policy
  judge/                 # llm_judge (pluggable), calibration (Cohen's κ)
  benchmarks/            # agentdojo (task source), runner (suite aggregation)
  report/                # rich tables, JSON persistence, regression diff
  cli.py                 # run · report · bench · ci · calibrate
tests/                   # 67 tests
.github/workflows/eval.yml   # ruff + pytest + regression gate
```

## Status

The full pipeline is implemented and green end-to-end: two adapters (LangGraph, OpenAI Agents SDK),
five evaluators, a pluggable judge + κ calibration, the AgentDojo benchmark source + runner, the
report layer, and the CI regression gate. CrewAI remains a stub adapter (breadth target).
