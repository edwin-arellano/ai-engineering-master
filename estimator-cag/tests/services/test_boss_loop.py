"""Tests del loop del boss con dependencias mockeadas."""

from unittest.mock import MagicMock

import pytest

from app.foundations.config import get_settings
from app.domain.actor_critic_boss import (
    CriticFeedback,
    CriticIssue,
    CriticVerdict,
    IssueCategory,
    IssueSeverity,
)
from app.domain.estimation import EstimationResult, Phase
from app.domain.session import EstimationMode, Session
from app.domain.tier import UserTier
from app.generation.agentic.boss import BossService


def _result(confidence: int = 80) -> EstimationResult:
    return EstimationResult(
        summary="Test estimation",
        total_duration_weeks=7,
        total_cost_eur=44000,
        confidence_pct=confidence,
        phases=[
            Phase(
                name="Discovery",
                duration_weeks=3,
                cost_eur=18000,
                confidence_pct=80,
                assumptions=[],
            ),
            Phase(
                name="Build",
                duration_weeks=4,
                cost_eur=26000,
                confidence_pct=80,
                assumptions=[],
            ),
        ],
    )


def _blocking_feedback() -> CriticFeedback:
    return CriticFeedback(
        verdict=CriticVerdict.NEEDS_ITERATION,
        issues=[
            CriticIssue(
                category=IssueCategory.SCOPE_MISMATCH,
                severity=IssueSeverity.MAJOR,
                field_path="phases",
                description="Missing integration phase.",
                suggested_fix="Add an integration phase.",
            )
        ],
        review_confidence_pct=85,
    )


@pytest.fixture()
def session() -> Session:
    return Session(estimation_mode=EstimationMode.ACTOR_CRITIC_BOSS)


def test_boss_accepts_on_clean_review(session: Session) -> None:
    wrapper = MagicMock()
    boss = BossService(wrapper=wrapper, settings=get_settings())
    boss.critic.review = MagicMock(return_value=CriticFeedback.empty_accept())

    run_actor = MagicMock(return_value=_result())

    result = boss.run(
        session=session,
        transcript="t",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        attachments_text="",
        tier=UserTier.DEVELOPER,
        run_actor=run_actor,
    )
    assert result.converged is True
    assert result.total_iterations == 1
    run_actor.assert_called_once()


def test_boss_synthesizes_when_actor_fails_after_first_iteration(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el actor rompe en una iteración >0 (típico con critic_feedback que
    induce drift), el boss sintetiza desde el último draft bueno en vez de
    propagar el 500."""
    wrapper = MagicMock()
    boss = BossService(wrapper=wrapper, settings=get_settings())
    boss.critic.review = MagicMock(return_value=_blocking_feedback())
    monkeypatch.setattr(
        boss, "_synthesize", lambda last, fb, metrics=None: _result(confidence=40)
    )

    call_count = {"n": 0}

    def flaky_actor(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _result(confidence=80)
        raise RuntimeError("Instructor retries exhausted")

    result = boss.run(
        session=session,
        transcript="t",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        attachments_text="",
        tier=UserTier.DEVELOPER,
        run_actor=flaky_actor,
    )
    assert result.converged is False
    assert result.total_iterations == 2  # éxito en 1, fallo en 2 → synthesize
    assert result.final_result.confidence_pct == 40


def test_boss_synthesizes_when_never_converges(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = MagicMock()
    boss = BossService(wrapper=wrapper, settings=get_settings())
    # El crítico siempre devuelve issues bloqueantes → nunca acepta.
    boss.critic.review = MagicMock(return_value=_blocking_feedback())
    # La síntesis devuelve un resultado con confianza reducida.
    monkeypatch.setattr(
        boss, "_synthesize", lambda last, fb, metrics=None: _result(confidence=55)
    )

    run_actor = MagicMock(return_value=_result())

    result = boss.run(
        session=session,
        transcript="t",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        attachments_text="",
        tier=UserTier.DEVELOPER,
        run_actor=run_actor,
    )
    assert result.converged is False
    # Con ACB_MAX_ITERATIONS=3, el actor se llama 3 veces antes de sintetizar.
    assert run_actor.call_count == get_settings().acb_max_iterations
    assert result.final_result.confidence_pct == 55
