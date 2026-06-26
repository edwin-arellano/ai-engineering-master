"""Endpoint de producto del flujo RAG end-to-end (S09) + endpoint de debug de
retrieval (S10). NO sustituye a /api/v1/sessions/{id}/estimate (CAG conversacional)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.domain.structured_estimation import StructuredEstimate
from app.generation.rag.retrieval import (
    EstimateFromTranscriptResult,
    estimate_from_transcript,
    estimate_structured_from_transcript,
)
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.retrieval.service import dedup_budget_ids
from app.generation.rag.schemas import MetadataFilters

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["rag-estimation"])


class EstimateFromTranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=60000)
    # None → toma el default de Settings (rag_search_mode / reranking_enabled).
    search_mode: Literal["vector", "hybrid"] | None = None
    reranking: bool | None = None


class EstimateStructuredRequest(BaseModel):
    """Request del flujo invertido (S10): esqueleto CAG → horas por-tarea RAG. El
    search_mode/reranking aplican a la búsqueda de vecinos por-tarea (default per_task_*)."""

    transcript: str = Field(..., min_length=10, max_length=60000)
    search_mode: Literal["vector", "hybrid"] | None = None
    reranking: bool | None = None


class RetrieveDebugRequest(BaseModel):
    """Request del endpoint de debug de retrieval (S10): mide recuperación sin pagar
    la generación LLM. Mismas palancas search_mode/reranking que el endpoint de producto.

    `apply_metadata_filters=False` desactiva el filtro de sector de S09 (mantiene solo
    el de chunk_types): útil para MEDIR la calidad de ranking sobre todo el corpus, donde
    híbrida/reranking tienen margen — con el filtro de sector el conjunto cabe entero en
    el top-5 y precision@5 se vuelve insensible al orden."""

    transcript: str = Field(..., min_length=10, max_length=60000)
    search_mode: Literal["vector", "hybrid"] | None = None
    reranking: bool | None = None
    apply_metadata_filters: bool = True
    # Toggles del pipeline avanzado (S10): medición por etapa.
    routing: bool = False
    query_transform: bool = False
    temporal_decay: bool = False


class RetrieveDebugResult(BaseModel):
    """Ranking recuperado (sin generación). `search_time_ms` es el tiempo del pipeline
    de retrieval (excluye reformulación y HTTP): latencia limpia para la medición.
    `targets`/`routing_level`/`technique` permiten medir accuracy de routing y técnica."""

    retrieved_budget_ids: list[str]
    retrieved_chunks: int
    search_mode: str
    reranking: bool
    search_time_ms: int
    targets: list[str] = []
    routing_level: str = ""
    technique: str = "direct"


@lru_cache(maxsize=1)
def _wrapper() -> LLMWrapper:
    return LLMWrapper(get_settings())


@router.post("/estimate-from-transcript", response_model=EstimateFromTranscriptResult)
async def estimate_from_transcript_endpoint(
    request: EstimateFromTranscriptRequest,
) -> EstimateFromTranscriptResult:
    settings = get_settings()
    try:
        return await estimate_from_transcript(
            transcript=request.transcript,
            wrapper=_wrapper(),
            embedder=LiteLLMEmbedder(),
            session_factory=AsyncSessionLocal,
            settings=settings,
            search_mode=request.search_mode,
            reranking=request.reranking,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rag.estimate_from_transcript_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generando la estimación desde la transcripción",
        )


@router.post("/estimate-structured", response_model=StructuredEstimate)
async def estimate_structured_endpoint(
    request: EstimateStructuredRequest,
) -> StructuredEstimate:
    """Flujo invertido (S10): genera el esqueleto de módulos/tareas con CAG (sin horas)
    y deriva las horas por-tarea del histórico (consenso determinista de vecinos +
    fiabilidad). Conserva intactos /estimate-from-transcript y /retrieve-debug."""
    settings = get_settings()
    try:
        return await estimate_structured_from_transcript(
            transcript=request.transcript,
            wrapper=_wrapper(),
            embedder=LiteLLMEmbedder(),
            session_factory=AsyncSessionLocal,
            settings=settings,
            search_mode=request.search_mode,
            reranking=request.reranking,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rag.estimate_structured_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generando la estimación estructurada",
        )


@router.post("/retrieve-debug", response_model=RetrieveDebugResult)
async def retrieve_debug_endpoint(request: RetrieveDebugRequest) -> RetrieveDebugResult:
    """Solo retrieval: reformula + recupera (vector|hybrid + rerank) y devuelve el
    ranking de budget_id, SIN augmentation ni generación. Herramienta de medición
    (el arnés recorre las 4 configuraciones contra este endpoint)."""
    settings = get_settings()
    search_mode = request.search_mode or settings.rag_search_mode
    reranking = settings.reranking_enabled if request.reranking is None else request.reranking
    # Sin filtro de sector: solo restringe a ambos chunk_types (mide ranking sobre
    # todo el corpus). Con filtro: None → el pipeline aplica filters_from_reformulation.
    filters = (
        None
        if request.apply_metadata_filters
        else MetadataFilters(chunk_types=["budget_component", "historical_task"])
    )
    try:
        reformulated = reformulate_transcript(
            transcript=request.transcript, wrapper=_wrapper(), settings=settings
        )
        # El wrapper es necesario si se activa routing o query_transform (etapas con LLM).
        pipeline = RetrievalPipeline(
            embedder=LiteLLMEmbedder(),
            session_factory=AsyncSessionLocal,
            wrapper=_wrapper(),
        )
        retrieval = await pipeline.retrieve(
            reformulated=reformulated, settings=settings,
            search_mode=search_mode, reranking=reranking, filters=filters,
            routing=request.routing, query_transform=request.query_transform,
            temporal_decay=request.temporal_decay,
        )
        return RetrieveDebugResult(
            retrieved_budget_ids=dedup_budget_ids(retrieval.chunks),
            retrieved_chunks=len(retrieval.chunks),
            search_mode=search_mode,
            reranking=reranking,
            search_time_ms=retrieval.search_time_ms,
            targets=retrieval.targets,
            routing_level=retrieval.routing_level,
            technique=retrieval.technique,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rag.retrieve_debug_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error recuperando candidatos",
        )
