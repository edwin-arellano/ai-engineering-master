"""Búsqueda semántica: embebe la query con el MISMO modelo que la ingesta
(text-embedding-3-small) y devuelve los k chunks más cercanos por distancia coseno.
La búsqueda usa el índice HNSW half-vec (ver repository._build_halfvec_search_stmt):
operador `<=>` alineado con `halfvec_cosine_ops`. `ef_search` (HNSW_EF_SEARCH) ajusta
el balance recall/latencia y se pasa desde settings.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.foundations.config import get_settings
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import get_db_session
from app.generation.rag.persistence.models import BudgetChunkRow
from app.generation.rag.persistence.repository import search_chunks
from app.generation.rag.schemas import SearchRequest, SearchResponse, SearchResultItem

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    embedder = LiteLLMEmbedder()
    started = time.perf_counter()
    try:
        query_vector = await asyncio.to_thread(embedder.embed_one, request.query)
    except Exception:  # noqa: BLE001
        logger.exception("search.embed_failed")
        raise HTTPException(status_code=500, detail="Error embebiendo la query")

    rows = await search_chunks(
        session,
        model=BudgetChunkRow,
        query_vector=query_vector,
        k=request.k,
        ef_search=get_settings().hnsw_ef_search,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    results = [
        SearchResultItem(
            chunk_id=row._mapping["chunk_id"],
            document_id=row._mapping["document_id"],
            chunk_type=row._mapping["chunk_type"],
            content=row._mapping["content"],
            distance=round(float(row._mapping["distance"]), 4),
            metadata=row._mapping["metadata"],
        )
        for row in rows
    ]
    logger.info("search.done", k=request.k, hits=len(results), search_time_ms=elapsed_ms)
    return SearchResponse(
        query=request.query, k=request.k, search_time_ms=elapsed_ms, results=results
    )
