"""Normalized trajectory schema — the heart of AgentTrace.

Everything downstream (evaluators, judge, benchmarks, reporting) depends ONLY on the models in
this module. Framework-specific shapes are converted into these models by code in ``adapters/``
and never leak past that boundary.

Field names are aligned with the OpenTelemetry GenAI semantic conventions where a natural match
exists; the mapping is noted inline next to each field. See
https://opentelemetry.io/docs/specs/semconv/gen-ai/ .
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Keys whose values look credential-like are redacted from audit records. Matches substrings so
# e.g. "api_key", "access_token", "client_secret", "db_password" are all caught.
_CREDENTIAL_KEY_RE = re.compile(r"(token|key|secret|password|passwd|pwd|credential)", re.IGNORECASE)
_REDACTED = "***REDACTED***"


def _redact(value: Any) -> Any:
    """Recursively redact credential-like keys in a dict/list structure."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _CREDENTIAL_KEY_RE.search(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class Step(BaseModel):
    """One agent action within a trajectory.

    OTel GenAI mapping:
      - ``role``        -> gen_ai.message.role / span kind
      - ``tool_name``   -> gen_ai.tool.name
      - ``tool_args``   -> gen_ai.tool.call.arguments
      - ``tool_output`` -> gen_ai.tool.call.result
      - ``tokens``      -> gen_ai.usage.input_tokens + gen_ai.usage.output_tokens (combined)
    """

    index: int
    role: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: str | None = None
    tokens: int | None = None
    latency_ms: float | None = None
    timestamp: datetime


class Outcome(BaseModel):
    """The terminal result of a trajectory."""

    success: bool
    final_answer: str | None = None
    ground_truth: str | None = None
    error: str | None = None


class RunConfig(BaseModel):
    """The reproducibility tuple ``⟨agent_id, task_set, harness_config, judge, budget⟩``.

    Captures everything needed to reproduce a run. Two results are only comparable when their
    configs match (compare via :meth:`fingerprint`). Inspired by the auditable ``⟨M,A,D,S,J⟩``
    tuple from the PromptSecurity SoK (arXiv:2510.15476).
    """

    agent_id: str
    task_set: str
    # Harness settings that affect a result: model, decoding params, seeds, tool set, etc.
    harness_config: dict[str, Any] = Field(default_factory=dict)
    judge: str | None = None
    # Budget ceilings, e.g. {"max_steps": 20, "max_tokens": 100_000, "max_usd": 1.0}.
    budget: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        """A stable hash of the config, used to check two runs for comparability.

        Stable across processes and insertion order: the config is serialized with sorted keys
        before hashing, so semantically identical configs produce identical fingerprints.
        """
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Trajectory(BaseModel):
    """A normalized agent run: config, ordered steps, terminal outcome, and run-level metrics."""

    config: RunConfig
    steps: list[Step] = Field(default_factory=list)
    outcome: Outcome
    total_tokens: int = 0
    total_usd: float = 0.0
    wall_clock_s: float = 0.0

    def to_audit_record(self) -> dict[str, Any]:
        """Return a complete, JSON-serializable audit record for this run.

        Audit records are first-class: they are what make a result checkable. Credential-like
        values in each step's ``tool_args`` are redacted so secrets never land on disk.
        """
        record = self.model_dump(mode="json")
        record["config_fingerprint"] = self.config.fingerprint()
        for step in record.get("steps", []):
            if step.get("tool_args"):
                step["tool_args"] = _redact(step["tool_args"])
        return record

    def write_audit(self, path: str | Path) -> Path:
        """Write the audit record to ``path`` as JSON and return the resolved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_audit_record(), indent=2, default=str), encoding="utf-8")
        return path


class EvalResult(BaseModel):
    """The output of a single evaluator over a single trajectory."""

    evaluator: str
    score: float
    passed: bool
    detail: dict[str, Any] = Field(default_factory=dict)
