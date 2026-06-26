"""Flujo invertido — horas por-tarea por consenso determinista de vecinos. Unit puro:
la fiabilidad por nº/cercanía y el consenso mediana, con el pipeline mockeado (sin DB)."""

from __future__ import annotations

from app.domain.structured_estimation import Reliability, TaskNeighbor
from app.foundations.config import get_settings
from app.generation.rag.retrieval.per_task import _reliability, estimate_task_hours
from app.generation.rag.schemas import (
    MetadataFilters,
    RetrievalResult,
    RetrievedChunk,
)


def _settings():
    return get_settings().model_copy(
        update={
            "per_task_top_k": 5,
            "per_task_close_distance": 0.45,
            "per_task_min_neighbors_high": 2,
        }
    )


def _neighbor(distance: float) -> TaskNeighbor:
    return TaskNeighbor(budget_id="BUD", chunk_ref="r", estimated_hours=10.0, distance=distance)


def test_reliability_none_without_neighbors():
    assert _reliability([], _settings()) == Reliability.NONE


def test_reliability_high_with_enough_close():
    neighbors = [_neighbor(0.1), _neighbor(0.2)]  # 2 cercanos (>= min_neighbors_high)
    assert _reliability(neighbors, _settings()) == Reliability.HIGH


def test_reliability_medium_with_one_close():
    neighbors = [_neighbor(0.1), _neighbor(0.9)]  # 1 cercano
    assert _reliability(neighbors, _settings()) == Reliability.MEDIUM


def test_reliability_low_when_all_far():
    neighbors = [_neighbor(0.8), _neighbor(0.9)]  # ninguno cercano
    assert _reliability(neighbors, _settings()) == Reliability.LOW


class _FakePipeline:
    """Pipeline mock: devuelve chunks con `estimated_hours` en metadata."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(self, *, reformulated, settings, **_kwargs) -> RetrievalResult:
        return RetrievalResult(
            reformulated=reformulated, filters=MetadataFilters(),
            top_k=settings.rag_top_k, distance_threshold=0.0,
            chunks=self._chunks, search_time_ms=1,
        )


def _hist_chunk(cid: int, hours: float, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=cid, chunk_type="historical_task",
        content="c", distance=distance,
        metadata={"estimated_hours": hours, "budget_id": f"BUD-{cid}", "chunk_id": f"t{cid}"},
    )


async def test_consensus_is_median_of_neighbor_hours():
    chunks = [
        _hist_chunk(1, 20.0, 0.1),
        _hist_chunk(2, 25.0, 0.2),
        _hist_chunk(3, 90.0, 0.3),  # outlier: la mediana lo resiste
    ]
    estimate = await estimate_task_hours(
        "Implementar OAuth", pipeline=_FakePipeline(chunks), settings=_settings()
    )
    assert estimate.suggested_hours == 25.0  # mediana de [20, 25, 90]
    assert estimate.needs_human_input is False
    assert estimate.reliability == Reliability.HIGH  # 3 vecinos cercanos
    assert len(estimate.neighbors) == 3


async def test_no_neighbors_needs_human_input():
    estimate = await estimate_task_hours(
        "Tarea sin histórico", pipeline=_FakePipeline([]), settings=_settings()
    )
    assert estimate.suggested_hours is None
    assert estimate.needs_human_input is True
    assert estimate.reliability == Reliability.NONE


async def test_chunks_without_hours_are_ignored():
    chunk_no_hours = RetrievedChunk(
        chunk_id=9, document_id=9, chunk_type="historical_task", content="c",
        distance=0.1, metadata={"budget_id": "BUD-9"},  # sin estimated_hours
    )
    estimate = await estimate_task_hours(
        "Tarea", pipeline=_FakePipeline([chunk_no_hours]), settings=_settings()
    )
    assert estimate.needs_human_input is True
    assert estimate.neighbors == []
