"""Judge + outcome evaluator tests, all with an injected mock judge callable (no network)."""

from __future__ import annotations

from datetime import UTC, datetime

from agenttrace.evaluators.outcome import OutcomeEvaluator
from agenttrace.judge.llm_judge import LLMJudge, Verdict
from agenttrace.schema import EvalResult, Outcome, RunConfig, Step, Trajectory

_TS = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)


def _trajectory(*, final_answer=None, ground_truth=None, task=None) -> Trajectory:
    harness = {"task": task} if task else {}
    return Trajectory(
        config=RunConfig(agent_id="demo", task_set="smoke", harness_config=harness),
        steps=[
            Step(
                index=0,
                role="assistant",
                tool_name="calculator",
                tool_args={"expression": "21 * 2"},
                tool_output="42",
                tokens=None,
                latency_ms=None,
                timestamp=_TS,
            )
        ],
        outcome=Outcome(success=True, final_answer=final_answer, ground_truth=ground_truth),
        total_tokens=0,
        total_usd=0.0,
        wall_clock_s=0.0,
    )


# --- judge --------------------------------------------------------------------------------------


def test_judge_accepts_json_string() -> None:
    calls = []

    def fake_complete(prompt: str) -> str:
        calls.append(prompt)
        return '{"success": true, "rationale": "answer matches the task"}'

    verdict = LLMJudge(fake_complete, model="mock-1").judge(_trajectory(final_answer="42"))
    assert isinstance(verdict, Verdict)
    assert verdict.success is True
    assert "matches" in verdict.rationale
    # The prompt was actually rendered and passed to the callable.
    assert "Final answer: 42" in calls[0]


def test_judge_accepts_fenced_json() -> None:
    def fake_complete(_: str) -> str:
        return '```json\n{"success": false, "rationale": "wrong"}\n```'

    verdict = LLMJudge(fake_complete).judge(_trajectory(final_answer="oops"))
    assert verdict.success is False


def test_judge_accepts_dict_and_verdict() -> None:
    v_dict = LLMJudge(lambda _: {"success": True, "rationale": "ok"}).judge(_trajectory())
    assert v_dict.success is True

    v_obj = LLMJudge(lambda _: Verdict(success=False, rationale="no")).judge(_trajectory())
    assert v_obj.success is False


# --- outcome evaluator --------------------------------------------------------------------------


def test_outcome_ground_truth_match() -> None:
    traj = _trajectory(final_answer="It is sunny.", ground_truth="sunny")
    r = OutcomeEvaluator().evaluate(traj)  # no judge needed
    assert isinstance(r, EvalResult)
    assert r.passed is True
    assert r.score == 1.0
    assert r.detail["method"] == "ground_truth"


def test_outcome_ground_truth_mismatch() -> None:
    traj = _trajectory(final_answer="rainy", ground_truth="sunny")
    r = OutcomeEvaluator().evaluate(traj)
    assert r.passed is False
    assert r.score == 0.0


def test_outcome_falls_back_to_judge() -> None:
    traj = _trajectory(final_answer="42", task="What is 21 * 2?")  # no ground_truth

    def fake_complete(prompt: str) -> dict:
        assert "What is 21 * 2?" in prompt  # task threaded through
        return {"success": True, "rationale": "42 is correct"}

    r = OutcomeEvaluator(judge=LLMJudge(fake_complete, model="mock-judge")).evaluate(traj)
    assert r.passed is True
    assert r.detail["method"] == "judge"
    assert r.detail["model"] == "mock-judge"
    assert r.detail["rationale"] == "42 is correct"


def test_outcome_without_judge_or_ground_truth_is_skippable() -> None:
    traj = _trajectory(final_answer="42")  # no ground_truth, no judge
    try:
        OutcomeEvaluator().evaluate(traj)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError so the CLI can skip it")
