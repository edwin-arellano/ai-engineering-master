"""Endpoint de ingesta: persiste un documento y sus chunks (cada uno con embedding) en
una sola transacción. `async def` para usar la sesión SQLAlchemy async; el embedder es
bloqueante (litellm.embedding) y se ejecuta en un thread vía asyncio.to_thread para no
bloquear el event loop.

S10: parametrizado por colección. Default `budgets` = camino histórico (presupuesto →
chunker estructural/tareas). `transcripts`/`technical_docs` ingestan texto plano
(`content_text`) con su chunker de texto a la tabla correspondiente.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.chunking.registry import build_chunker
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.collections import COLLECTION_MODELS
from app.generation.rag.persistence.database import get_db_session
from app.generation.rag.persistence.models import EMBEDDING_DIM
from app.generation.rag.persistence.repository import (
    BUDGET_COMPONENT,
    HISTORICAL_TASK,
    TECHNICAL_REFERENCE,
    TRANSCRIPT_SEGMENT,
    get_document_id_by_source_path,
    ingest_document,
)
from app.generation.rag.schemas import (
    Budget,
    Chunk,
    DocumentIngestRequest,
    DocumentIngestResponse,
    SearchTarget,
)
from app.ingest.collections.technical_docs import chunk_technical_doc
from app.ingest.collections.transcripts import chunk_transcript

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Mapea la estrategia de chunking solicitada al chunk_type que se persiste (S09).
_STRATEGY_TO_CHUNK_TYPE = {
    "structural": BUDGET_COMPONENT,
    "historical_task": HISTORICAL_TASK,
}


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


def _chunk_text_collection(
    collection: SearchTarget, request: DocumentIngestRequest
) -> tuple[list[Chunk], str, dict[str, Any]]:
    """Trocea texto plano para las colecciones transcripts/technical_docs. Devuelve
    (chunks, chunk_type, document_metadata)."""
    text = (request.content_text or "").strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"`content_text` es obligatorio para collection={collection.value}",
        )
    name = Path(request.source_path).name
    if collection is SearchTarget.TRANSCRIPTS:
        meta = {"year": date.today().year, "source": name}
        chunks = chunk_transcript(text, source_path=request.source_path, meta=meta)
        return chunks, TRANSCRIPT_SEGMENT, {**meta, "collection": "transcripts"}
    # technical_docs
    meta = {"technology": Path(request.source_path).stem, "year": date.today().year}
    chunks = chunk_technical_doc(text, source_path=request.source_path, meta=meta)
    return chunks, TECHNICAL_REFERENCE, {**meta, "collection": "technical_docs"}


def _chunk_budget(
    request: DocumentIngestRequest,
) -> tuple[list[Chunk], str, dict[str, Any]]:
    """Camino histórico: valida el Budget y trocea con la estrategia pedida."""
    if request.content is None:
        raise HTTPException(
            status_code=422, detail="`content` (Budget) es obligatorio para collection=budgets"
        )
    try:
        budget = Budget.model_validate(request.content)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"`content` no es un Budget válido: {exc.errors()}",
        )
    chunk_type = _STRATEGY_TO_CHUNK_TYPE.get(request.chunk_strategy)
    if chunk_type is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"chunk_strategy no soportada para ingesta: '{request.chunk_strategy}'. "
                f"Válidas: {sorted(_STRATEGY_TO_CHUNK_TYPE)}"
            ),
        )
    chunker = build_chunker(request.chunk_strategy)
    chunks = chunker.chunk([budget])
    document_metadata = _build_document_metadata(budget, request.document_type)
    return chunks, chunk_type, document_metadata


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

    # 2 + 3. chunking según la colección destino (huérfanos fuera: no los vectorizamos)
    if request.collection is SearchTarget.BUDGETS:
        raw_chunks, chunk_type, document_metadata = _chunk_budget(request)
    else:
        raw_chunks, chunk_type, document_metadata = _chunk_text_collection(
            request.collection, request
        )
    chunks = [c for c in raw_chunks if not c.is_orphan]
    model = COLLECTION_MODELS[request.collection]

    # 4. embeddings por lotes (embedder bloqueante → thread)
    embedder = LiteLLMEmbedder()
    try:
        embedded = await asyncio.to_thread(embedder.embed_many, chunks)
    except Exception:  # noqa: BLE001
        logger.exception("embeddings.ingest_failed", source_path=request.source_path)
        raise HTTPException(status_code=500, detail="Error generando embeddings")

    # 5 + 6. persistir en una transacción
    document_id, chunks_created = await ingest_document(
        session,
        model=model,
        source_path=request.source_path,
        document_type=request.document_type,
        document_metadata=document_metadata,
        embedded_chunks=embedded,
        chunk_type=chunk_type,
    )

    dimension = len(embedded[0].embedding) if embedded else EMBEDDING_DIM
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "embeddings.ingest_done",
        collection=request.collection.value,
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
