"""Ponderación blanda: decaimiento temporal por año y reordenación de finalistas.
Unit puro (sin DB ni modelo)."""

from __future__ import annotations

from app.foundations.config import get_settings
from app.generation.rag.retrieval.weighting import apply_soft_weighting, temporal_weight
from app.generation.rag.schemas import ReformulatedQuery, RetrievedChunk


def test_temporal_weight_current_year_is_one():
    assert temporal_weight(2026, half_life_years=2.5, today_year=2026) == 1.0


def test_temporal_weight_half_life_is_half():
    # A una semivida exacta de distancia, el peso es 0.5.
    assert temporal_weight(2024, half_life_years=2.0, today_year=2026) == 0.5


def test_temporal_weight_none_is_one():
    assert temporal_weight(None, half_life_years=2.5) == 1.0


def _chunk(cid: int, *, distance: float, year: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=cid, chunk_type="historical_task",
        content=f"c{cid}", distance=distance, metadata={"year": year},
    )


def _reformulated() -> ReformulatedQuery:
    return ReformulatedQuery(
        project_function="x", sector="finance", scale="medium", search_text="q"
    )


def test_recent_beats_old_at_equal_distance():
    """Con decaimiento activo, un histórico reciente NO debe quedar por detrás de uno
    viejo equivalente (misma distancia)."""
    settings = get_settings().model_copy(
        update={"temporal_decay_enabled": True, "temporal_half_life_years": 2.0}
    )
    old = _chunk(1, distance=0.30, year=2019)
    recent = _chunk(2, distance=0.30, year=2026)
    # Entrada con el viejo primero a propósito; la ponderación debe reordenar.
    ranked = apply_soft_weighting([old, recent], reformulated=_reformulated(), settings=settings)
    assert [c.chunk_id for c in ranked] == [2, 1]


def test_does_not_expel_candidates():
    """Solo reordena: el nº de finalistas se conserva (no filtra)."""
    settings = get_settings().model_copy(update={"temporal_decay_enabled": True})
    chunks = [_chunk(i, distance=0.1 * i, year=2020 + i) for i in range(1, 5)]
    ranked = apply_soft_weighting(chunks, reformulated=_reformulated(), settings=settings)
    assert len(ranked) == len(chunks)
    assert {c.chunk_id for c in ranked} == {c.chunk_id for c in chunks}


def test_disabled_keeps_distance_order():
    """Sin toggles, el orden por distancia (menor primero) se preserva."""
    settings = get_settings().model_copy(
        update={"temporal_decay_enabled": False, "contextual_weighting_enabled": False}
    )
    chunks = [_chunk(1, distance=0.5, year=2019), _chunk(2, distance=0.1, year=2019)]
    ranked = apply_soft_weighting(chunks, reformulated=_reformulated(), settings=settings)
    assert [c.chunk_id for c in ranked] == [2, 1]  # menor distancia primero
