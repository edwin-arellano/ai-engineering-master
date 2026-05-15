"""Tests de los `model_validator` de EstimationResult."""

import pytest

from app.schemas.estimation import EstimationResult, Phase


def _phase(**kwargs) -> Phase:
    """Helper para construir Phase con valores razonables por defecto."""
    defaults = dict(
        name="Discovery",
        duration_weeks=3,
        cost_eur=18000,
        confidence_pct=80,
        assumptions=["Single tech lead", "Sandbox creds ready"],
    )
    defaults.update(kwargs)
    return Phase(**defaults)


def test_total_must_match_sum_of_phases_ok_with_exact_sums() -> None:
    result = EstimationResult(
        summary="Test project",
        total_duration_weeks=7,
        total_cost_eur=44000,
        confidence_pct=80,
        phases=[
            _phase(name="Discovery", duration_weeks=3, cost_eur=18000),
            _phase(name="Build", duration_weeks=4, cost_eur=26000),
        ],
    )
    assert result.total_duration_weeks == 7


def test_total_must_match_sum_of_phases_ok_within_tolerances() -> None:
    # Duración: total 7, suma de fases 8 → diferencia de 1, dentro de tolerancia.
    # Coste: total 44000, suma de fases 45500 → diferencia 3.4%, dentro de ±5%.
    result = EstimationResult(
        summary="Test project",
        total_duration_weeks=7,
        total_cost_eur=44000,
        confidence_pct=80,
        phases=[
            _phase(duration_weeks=3, cost_eur=18500),
            _phase(name="Build", duration_weeks=5, cost_eur=27000),
        ],
    )
    assert result.confidence_pct == 80


def test_total_must_match_sum_of_phases_fails_on_duration() -> None:
    with pytest.raises(ValueError, match="total_duration_weeks"):
        EstimationResult(
            summary="Test project",
            total_duration_weeks=10,
            total_cost_eur=44000,
            confidence_pct=80,
            phases=[
                _phase(duration_weeks=3, cost_eur=18000),
                _phase(name="Build", duration_weeks=4, cost_eur=26000),
            ],
        )


def test_total_must_match_sum_of_phases_fails_on_cost() -> None:
    with pytest.raises(ValueError, match="total_cost_eur"):
        EstimationResult(
            summary="Test project",
            total_duration_weeks=7,
            total_cost_eur=50000,
            confidence_pct=80,
            phases=[
                _phase(duration_weeks=3, cost_eur=18000),
                _phase(name="Build", duration_weeks=4, cost_eur=26000),
            ],
        )


def test_empty_phases_skip_total_validation() -> None:
    # Caso out-of-scope: phases vacío, totales a cero, no se valida coherencia.
    result = EstimationResult(
        summary="Out of scope: not a software project",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=0,
        phases=[],
    )
    assert result.phases == []


def test_low_confidence_requires_out_of_scope_prefix() -> None:
    with pytest.raises(ValueError, match="Out of scope"):
        EstimationResult(
            summary="A perfectly normal-sounding estimation",
            total_duration_weeks=0,
            total_cost_eur=0,
            confidence_pct=20,
            phases=[],
        )


def test_low_confidence_ok_with_explicit_out_of_scope() -> None:
    result = EstimationResult(
        summary="Out of scope: description too vague to estimate",
        total_duration_weeks=0,
        total_cost_eur=0,
        confidence_pct=0,
        phases=[],
    )
    assert result.confidence_pct == 0


def test_confidence_at_threshold_does_not_require_prefix() -> None:
    # 30 está justo en el límite; el validator solo dispara con < 30.
    result = EstimationResult(
        summary="Borderline estimation",
        total_duration_weeks=3,
        total_cost_eur=15000,
        confidence_pct=30,
        phases=[_phase(duration_weeks=3, cost_eur=15000)],
    )
    assert result.confidence_pct == 30
