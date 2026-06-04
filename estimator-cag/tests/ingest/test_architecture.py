"""Tabla de verdad de recommend_architecture y viabilidad con baseline saturado."""

from __future__ import annotations

from app.ingest.architecture import (
    Architecture,
    BaselineSummary,
    CorpusProfile,
    IngestionArchitecture,
    ModelProfile,
    recommend_architecture,
)

_MODEL = ModelProfile(context_window=200_000, cost_per_million_input_tokens=0.80)


def _corpus(**overrides) -> CorpusProfile:
    base = dict(
        total_tokens=50_000,
        update_frequency_days=30,
        requires_source_attribution=False,
        requires_per_user_access_control=False,
    )
    base.update(overrides)
    return CorpusProfile(**base)


def test_attribution_forces_rag():
    assert (
        recommend_architecture(_corpus(requires_source_attribution=True), _MODEL)
        is Architecture.PURE_RAG
    )


def test_access_control_forces_rag():
    assert (
        recommend_architecture(
            _corpus(requires_per_user_access_control=True), _MODEL
        )
        is Architecture.PURE_RAG
    )


def test_context_overflow_forces_rag():
    # total_tokens > context_window * usable_ratio -> usage > 1
    assert (
        recommend_architecture(_corpus(total_tokens=500_000), _MODEL)
        is Architecture.PURE_RAG
    )


def test_high_frequency_forces_rag():
    assert (
        recommend_architecture(_corpus(update_frequency_days=3), _MODEL)
        is Architecture.PURE_RAG
    )


def test_stable_and_small_is_cag():
    arch = recommend_architecture(
        _corpus(total_tokens=10_000, update_frequency_days=120), _MODEL
    )
    assert arch is Architecture.PURE_CAG


def test_default_is_hybrid():
    assert recommend_architecture(_corpus(), _MODEL) is Architecture.HYBRID_CAG_RAG


def test_viability_false_under_saturated_baseline():
    # baseline saturado: P95 muy por encima del SLA -> no viable.
    baseline = BaselineSummary(
        latency_p50=19.4, latency_p95=69.6, cost_per_turn_mean=0.02, turns=72
    )
    arch = IngestionArchitecture(
        corpus=_corpus(total_tokens=2_000_000, requires_source_attribution=True),
        model=_MODEL,
        baseline=baseline,
        latency_sla_seconds=4.0,
        cost_per_turn_budget_usd=0.05,
    )
    viability = arch.viability()
    assert viability.latency_acceptable is False
    assert viability.is_viable() is False
    assert arch.recommend() is Architecture.PURE_RAG
