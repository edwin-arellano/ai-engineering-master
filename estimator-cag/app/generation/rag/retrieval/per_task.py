"""Fase 2 del flujo invertido: para cada tarea del esqueleto, busca vecinos en la
colección de budgets (historical_task) y DERIVA las horas de su metadata
`estimated_hours` (consenso determinista — NUNCA inferencia del modelo). La fiabilidad
sale del nº de vecinos y su cercanía. Sin vecinos → needs_human_input."""

from __future__ import annotations

from statistics import median

import structlog

from app.domain.structured_estimation import Reliability, TaskEstimate, TaskNeighbor
from app.foundations.config import Settings
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.schemas import MetadataFilters, ReformulatedQuery, SearchTarget

logger = structlog.get_logger(__name__)
# Horas en metadata están en horas-persona; el esqueleto no lleva unidad → devolvemos horas.


def _reliability(neighbors: list[TaskNeighbor], settings: Settings) -> Reliability:
    """Fiabilidad por nº de vecinos cercanos. Heurística explicable (regla de humildad):
    varios vecinos cercanos → high; alguno cercano → medium; lejanos → low; ninguno → none."""
    if not neighbors:
        return Reliability.NONE
    close = [n for n in neighbors if n.distance <= settings.per_task_close_distance]
    if len(close) >= settings.per_task_min_neighbors_high:
        return Reliability.HIGH
    if close:
        return Reliability.MEDIUM
    return Reliability.LOW


async def estimate_task_hours(
    task_title: str, *, pipeline: RetrievalPipeline, settings: Settings
) -> TaskEstimate:
    """Horas de UNA tarea por consenso (mediana) de vecinos históricos. No llama al LLM."""
    # Búsqueda explícita sobre budgets/historical_task (routing nivel 0).
    reformulated = ReformulatedQuery(
        project_function=task_title, technologies=[], sector="other",
        scale="small", countries=[], constraints=[], search_text=task_title,
    )
    filters = MetadataFilters(chunk_types=["historical_task"])
    retrieval = await pipeline.retrieve(
        reformulated=reformulated, settings=settings,
        search_mode=settings.per_task_search_mode, reranking=settings.per_task_reranking,
        routing=False, query_transform=False, temporal_decay=settings.temporal_decay_enabled,
        explicit_targets=[SearchTarget.BUDGETS], filters=filters,
    )
    neighbors: list[TaskNeighbor] = []
    for chunk in retrieval.chunks[: settings.per_task_top_k]:
        hours = chunk.metadata.get("estimated_hours")
        if hours is None:
            continue
        neighbors.append(
            TaskNeighbor(
                budget_id=str(chunk.metadata.get("budget_id", "")),
                chunk_ref=chunk.chunk_ref,
                estimated_hours=float(hours),
                distance=chunk.distance,
            )
        )
    reliability = _reliability(neighbors, settings)
    # Consenso = MEDIANA de las horas de los vecinos (robusta a outliers: "20h vs 25h").
    suggested = median([n.estimated_hours for n in neighbors]) if neighbors else None
    return TaskEstimate(
        title=task_title, suggested_hours=suggested, reliability=reliability,
        neighbors=neighbors, needs_human_input=(suggested is None),
    )
