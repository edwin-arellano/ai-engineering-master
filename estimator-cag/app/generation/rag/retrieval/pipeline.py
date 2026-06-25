"""RetrievalPipeline: compone las estrategias de recuperación según configuración.
- search_mode=vector  → solo rama semántica (RagRetriever, reutilizado).
- search_mode=hybrid  → rama semántica + léxica en paralelo, fusionadas con RRF.
- reranking=True       → recall-then-rerank: recall amplio (candidate_pool) → cross-encoder → top_k.
Todas las ramas respetan el contrato RetrievedChunk[] → activar/desactivar es un parámetro."""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.foundations.config import Settings
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.repository import search_chunks
from app.generation.rag.retrieval.fulltext_search import FullTextSearcher
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import get_reranker
from app.generation.rag.retrieval.retriever import filters_from_reformulation
from app.generation.rag.schemas import (
    MetadataFilters,
    ReformulatedQuery,
    RetrievalResult,
    RetrievedChunk,
)

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self, embedder: LiteLLMEmbedder, session_factory: async_sessionmaker
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._fulltext = FullTextSearcher(session_factory)

    async def _vector_search(
        self, *, search_text: str, k: int, filters: MetadataFilters, ef_search: int
    ) -> list[RetrievedChunk]:
        query_vector = await asyncio.to_thread(self._embedder.embed_one, search_text)
        async with self._session_factory() as session:
            rows = await search_chunks(
                session, query_vector=query_vector, k=k,
                ef_search=ef_search, filters=filters,
            )
        return [
            RetrievedChunk(
                chunk_id=r._mapping["chunk_id"], document_id=r._mapping["document_id"],
                chunk_type=r._mapping["chunk_type"], content=r._mapping["content"],
                distance=round(float(r._mapping["distance"]), 4),
                metadata=r._mapping["metadata"],
            )
            for r in rows
        ]

    async def retrieve(
        self,
        *,
        reformulated: ReformulatedQuery,
        settings: Settings,
        search_mode: str,
        reranking: bool,
        filters: MetadataFilters | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        filters = filters or filters_from_reformulation(reformulated)
        # Recall amplio si vamos a rerankear; si no, directamente el top_k final.
        recall_k = settings.retrieval_candidate_pool_size if reranking else settings.rag_top_k

        if search_mode == "hybrid":
            semantic, lexical = await asyncio.gather(
                self._vector_search(
                    search_text=reformulated.search_text, k=recall_k,
                    filters=filters, ef_search=settings.hnsw_ef_search,
                ),
                self._fulltext.search(
                    query_text=reformulated.search_text, k=recall_k, filters=filters
                ),
            )
            fused_ids = reciprocal_rank_fusion(
                [[c.chunk_id for c in semantic], [c.chunk_id for c in lexical]],
                k=settings.rrf_smoothing_k,
            )
            by_id = {c.chunk_id: c for c in [*semantic, *lexical]}
            candidates = [by_id[cid] for cid in fused_ids[:recall_k]]
        else:  # vector
            candidates = await self._vector_search(
                search_text=reformulated.search_text, k=recall_k,
                filters=filters, ef_search=settings.hnsw_ef_search,
            )

        if reranking:
            # Cross-encoder: cómputo local → fuera del event loop.
            candidates = await asyncio.to_thread(
                get_reranker().rerank,
                reformulated.search_text, candidates, settings.rag_top_k,
            )
        else:
            candidates = candidates[: settings.rag_top_k]

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "rag.pipeline_retrieved",
            search_mode=search_mode, reranking=reranking,
            recall_k=recall_k, final=len(candidates), search_time_ms=elapsed_ms,
        )
        return RetrievalResult(
            reformulated=reformulated, filters=filters,
            top_k=settings.rag_top_k, distance_threshold=settings.rag_distance_threshold,
            chunks=candidates, search_time_ms=elapsed_ms,
        )
