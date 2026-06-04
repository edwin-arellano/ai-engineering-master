"""Endpoint de ingesta de embeddings. Orquesta chunker (según strategy) → embedder
→ IngestResponse. def síncrono a propósito (FastAPI lo corre en threadpool):
litellm.embedding es bloqueante; no queremos bloquear el event loop.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.chunking.registry import LLM_BASED, build_chunker
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.schemas import IngestRequest, IngestResponse, IngestStats

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def _build_wrapper_if_needed(strategy: str) -> LLMWrapper | None:
    """Las estrategias LLM (propositional, contextual_retrieval) necesitan wrapper;
    mecánicas y semantic, no."""
    return LLMWrapper(get_settings()) if strategy in LLM_BASED else None


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    embedder = LiteLLMEmbedder()  # instancia por request → stats aisladas
    chunker = build_chunker(
        request.strategy,
        embedder=embedder,
        wrapper=_build_wrapper_if_needed(request.strategy),
    )
    try:
        # no vectorizar huérfanos: no metemos ruido en la futura BD vectorial.
        chunks = [c for c in chunker.chunk(request.budgets) if not c.is_orphan]
        embedded = embedder.embed_many(chunks)
    except Exception:  # noqa: BLE001 — error no controlado de la API de embeddings
        logger.exception(
            "embeddings.ingest_failed",
            strategy=request.strategy,
            budgets=len(request.budgets),
        )
        raise HTTPException(status_code=500, detail="Error generando embeddings")

    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=embedder.last_run_total_tokens,
        estimated_cost_usd=round(embedder.last_run_cost_usd, 6),
    )
    logger.info("embeddings.ingest_done", strategy=request.strategy, **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
