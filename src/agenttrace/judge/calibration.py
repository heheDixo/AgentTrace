"""Judge↔human agreement (Cohen's κ) on a hand-labelled set.

Calibration is not optional: never trust an uncalibrated judge's numbers. Computed from scratch
(no sklearn). Depends only on :mod:`agenttrace.schema` and the judge interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agenttrace.judge.llm_judge import LLMJudge
from agenttrace.schema import Trajectory


def cohen_kappa(human_labels: Sequence[bool], judge_labels: Sequence[bool]) -> float:
    """Cohen's κ between two sets of binary labels over the same items.

    κ = (po - pe) / (1 - pe), where ``po`` is observed agreement and ``pe`` is the agreement
    expected by chance from each rater's marginal rates. When ``pe == 1`` (both raters put every
    item in a single, identical category) the formula is undefined; by convention κ is ``1.0``
    for that (necessarily perfect-agreement) case.
    """
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must have equal length.")
    n = len(human_labels)
    if n == 0:
        raise ValueError("Cannot compute kappa over an empty label set.")

    agree = sum(1 for h, j in zip(human_labels, judge_labels, strict=True) if h == j)
    po = agree / n

    h_pos = sum(1 for h in human_labels if h) / n
    j_pos = sum(1 for j in judge_labels if j) / n
    # Expected agreement by chance across the two categories (True / False).
    pe = h_pos * j_pos + (1 - h_pos) * (1 - j_pos)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def calibration_report(
    human_labels: Sequence[bool], judge_labels: Sequence[bool]
) -> dict[str, Any]:
    """Build a calibration report dict from already-paired human and judge labels."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must have equal length.")
    n = len(human_labels)
    if n == 0:
        raise ValueError("Cannot calibrate over an empty label set.")

    agree = sum(1 for h, j in zip(human_labels, judge_labels, strict=True) if h == j)
    disagree = n - agree
    return {
        "n": n,
        "kappa": cohen_kappa(human_labels, judge_labels),
        "agreement_rate": agree / n,
        "agree": agree,
        "disagree": disagree,
        "human_positive": sum(1 for h in human_labels if h),
        "judge_positive": sum(1 for j in judge_labels if j),
    }


def calibrate(
    trajectories: Sequence[Trajectory],
    human_labels: Sequence[bool],
    judge: LLMJudge,
) -> dict[str, Any]:
    """Run ``judge`` over ``trajectories`` and report its agreement with ``human_labels``."""
    if len(trajectories) != len(human_labels):
        raise ValueError("trajectories and human_labels must have equal length.")
    judge_labels = [bool(judge.judge(t).success) for t in trajectories]
    return calibration_report(human_labels, judge_labels)
