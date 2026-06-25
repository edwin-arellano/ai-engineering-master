"""Fase 2 del RAG: retrieval como servicio (data-as-a-service). Embede el
search_text y recupera top-k chunks aplicando filtros de metadata + threshold.
Aislado del endpoint público /search."""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.foundations.config import Settings
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.repository import search_chunks
from app.generation.rag.schemas import (
    MetadataFilters,
    ReformulatedQuery,
    RetrievalResult,
    RetrievedChunk,
)

logger = structlog.get_logger(__name__)


def filters_from_reformulation(reformulated: ReformulatedQuery) -> MetadataFilters:
    """Deriva filtros deterministas del brief. Hoy: filtra por el sector detectado y
    por ambos chunk_types. Año/tech se dejan abiertos (corpus pequeño). Ajustable."""
    return MetadataFilters(
        sectors=[reformulated.sector],
        chunk_types=["budget_component", "historical_task"],
    )


class RagRetriever:
    def __init__(self, embedder: LiteLLMEmbedder, session_factory: async_sessionmaker) -> None:
        self._embedder = embedder
        self._session_factory = session_factory

    async def retrieve(
        self,
        *,
        reformulated: ReformulatedQuery,
        settings: Settings,
        filters: MetadataFilters | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        filters = filters or filters_from_reformulation(reformulated)
        query_vector = await asyncio.to_thread(self._embedder.embed_one, reformulated.search_text)

        async with self._session_factory() as session:
            rows = await search_chunks(
                session,
                query_vector=query_vector,
                k=settings.rag_top_k,
                ef_search=settings.hnsw_ef_search,
                filters=filters,
                distance_threshold=settings.rag_distance_threshold,
            )

        chunks = [
            RetrievedChunk(
                chunk_id=r._mapping["chunk_id"],
                document_id=r._mapping["document_id"],
                chunk_type=r._mapping["chunk_type"],
                content=r._mapping["content"],
                distance=round(float(r._mapping["distance"]), 4),
                metadata=r._mapping["metadata"],
            )
            for r in rows
        ]
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info("rag.retrieved", hits=len(chunks), top_k=settings.rag_top_k, search_time_ms=elapsed_ms)
        return RetrievalResult(
            reformulated=reformulated,
            filters=filters,
            top_k=settings.rag_top_k,
            distance_threshold=settings.rag_distance_threshold,
            chunks=chunks,
            search_time_ms=elapsed_ms,
        )
