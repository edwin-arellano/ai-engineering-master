"""Endpoint de ingesta: persiste un presupuesto como un `document` y sus chunks
(cada uno con embedding) en una sola transacción. `async def` para usar la sesión
SQLAlchemy async; el embedder es bloqueante (litellm.embedding) y se ejecuta en un
thread vía asyncio.to_thread para no bloquear el event loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.chunking.registry import build_chunker
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import get_db_session
from app.generation.rag.persistence.models import EMBEDDING_DIM
from app.generation.rag.persistence.repository import (
    get_document_id_by_source_path,
    ingest_document,
)
from app.generation.rag.schemas import (
    Budget,
    DocumentIngestRequest,
    DocumentIngestResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def _build_document_metadata(budget: Budget, document_type: str) -> dict[str, Any]:
    """Metadata a nivel documento (JSONB). Incluye los hechos del presupuesto que el
    directo usará para filtrar (sector, año, tecnología...)."""
    return {
        "budget_id": budget.budget_id,
        "client_name": budget.client_metadata.name,
        "sector": budget.client_metadata.sector,
        "country": budget.client_metadata.country,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "project_summary": budget.project_summary,
        "total_estimated_hours": budget.total_estimated_hours,
        "document_type": document_type,
    }


@router.post("/ingest")
async def ingest(
    request: DocumentIngestRequest,
    session: AsyncSession = Depends(get_db_session),
):
    started = time.perf_counter()

    # 1. ¿Ya existe? (idempotencia por source_path)
    existing_id = await get_document_id_by_source_path(session, request.source_path)
    if existing_id is not None:
        return JSONResponse(
            status_code=409,
            content={"detail": "Document already ingested", "document_id": existing_id},
        )

    # 2. content → Budget
    try:
        budget = Budget.model_validate(request.content)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"`content` no es un Budget válido: {exc.errors()}",
        )

    # 3. chunking estructural (huérfanos fuera: no los vectorizamos)
    chunker = build_chunker("structural")
    chunks = [c for c in chunker.chunk([budget]) if not c.is_orphan]

    # 4. embeddings por lotes (embedder bloqueante → thread)
    embedder = LiteLLMEmbedder()
    try:
        embedded = await asyncio.to_thread(embedder.embed_many, chunks)
    except Exception:  # noqa: BLE001
        logger.exception("embeddings.ingest_failed", source_path=request.source_path)
        raise HTTPException(status_code=500, detail="Error generando embeddings")

    # 5 + 6. persistir en una transacción
    document_metadata = _build_document_metadata(budget, request.document_type)
    document_id, chunks_created = await ingest_document(
        session,
        source_path=request.source_path,
        document_type=request.document_type,
        document_metadata=document_metadata,
        embedded_chunks=embedded,
    )

    dimension = len(embedded[0].embedding) if embedded else EMBEDDING_DIM
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "embeddings.ingest_done",
        document_id=document_id,
        chunks_created=chunks_created,
        embedding_dimension=dimension,
        ingestion_time_ms=elapsed_ms,
    )
    return DocumentIngestResponse(
        document_id=document_id,
        chunks_created=chunks_created,
        embedding_dimension=dimension,
        ingestion_time_ms=elapsed_ms,
    )
