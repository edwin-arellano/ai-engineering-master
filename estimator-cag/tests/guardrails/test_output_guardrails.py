"""Tests de los output guardrails (filtro de cache)."""

from app.foundations.config import Settings
from app.generation.cag.guardrails.output_guardrails import (
    is_low_confidence,
    is_out_of_scope,
    should_cache_result,
)
from app.domain.estimation import EstimationResult, Phase


def _result(**overrides) -> EstimationResult:
    defaults = dict(
        summary="Normal estimation",
        total_duration_weeks=7,
        total_cost_eur=44000,
        confidence_pct=80,
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
    defaults.update(overrides)
    return EstimationResult(**defaults)


def _settings(min_confidence_pct: int = 30) -> Settings:
    return Settings(
        MIN_CONFIDENCE_PCT=min_confidence_pct,
        MODERATION_ENABLED=False,
        ANTHROPIC_API_KEY="test-key",
    )


def test_is_out_of_scope_detects_prefix() -> None:
    result = _result(
        summary="Out of scope: not a software project",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=0,
        phases=[],
    )
    assert is_out_of_scope(result) is True


def test_is_out_of_scope_false_for_normal_summary() -> None:
    assert is_out_of_scope(_result()) is False


def test_is_low_confidence_true_below_threshold() -> None:
    result = _result(
        summary="Out of scope: low confidence",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=20,
        phases=[],
    )
    assert is_low_confidence(result, _settings(min_confidence_pct=30)) is True


def test_is_low_confidence_false_at_threshold() -> None:
    # 30 está exactamente en el umbral; el guardrail no dispara.
    result = _result(confidence_pct=30)
    assert is_low_confidence(result, _settings(min_confidence_pct=30)) is False


def test_should_cache_normal_response() -> None:
    assert should_cache_result(_result(), _settings()) is True


def test_should_not_cache_out_of_scope() -> None:
    result = _result(
        summary="Out of scope: bathroom remodel",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=0,
        phases=[],
    )
    assert should_cache_result(result, _settings()) is False


def test_should_not_cache_low_confidence() -> None:
    result = _result(
        summary="Out of scope: vague description",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=15,
        phases=[],
    )
    assert should_cache_result(result, _settings()) is False
