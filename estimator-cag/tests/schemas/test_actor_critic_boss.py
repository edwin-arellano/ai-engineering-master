"""Tests de los schemas del Actor-Critic-Boss."""

from app.schemas.actor_critic_boss import (
    CriticFeedback,
    CriticIssue,
    CriticVerdict,
    IssueCategory,
    IssueSeverity,
)


def _issue(severity: IssueSeverity) -> CriticIssue:
    return CriticIssue(
        category=IssueCategory.ARITHMETIC_ERROR,
        severity=severity,
        field_path="total_cost_eur",
        description="Sum of phases does not match total.",
        suggested_fix="Recompute total_cost_eur as the sum of phase costs.",
    )


def test_empty_accept_fallback() -> None:
    feedback = CriticFeedback.empty_accept()
    assert feedback.verdict is CriticVerdict.ACCEPT
    assert feedback.issues == []


def test_has_critical() -> None:
    feedback = CriticFeedback(
        verdict=CriticVerdict.NEEDS_ITERATION,
        issues=[_issue(IssueSeverity.CRITICAL)],
        review_confidence_pct=80,
    )
    assert feedback.has_critical() is True
    assert feedback.has_blocking() is True


def test_minor_only_is_not_blocking() -> None:
    feedback = CriticFeedback(
        verdict=CriticVerdict.ACCEPT,
        issues=[_issue(IssueSeverity.MINOR)],
        review_confidence_pct=90,
    )
    assert feedback.has_blocking() is False
