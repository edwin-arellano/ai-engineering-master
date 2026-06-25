"""Rama léxica del retrieval: full-text PostgreSQL ('spanish'). Mismo contrato que
la rama vectorial (entra texto, salen RetrievedChunk[]) para que el pipeline las
componga indistintamente."""

from __future__ import annotations

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.generation.rag.persistence.repository import search_chunks_fulltext
from app.generation.rag.schemas import MetadataFilters, RetrievedChunk

logger = structlog.get_logger(__name__)


class FullTextSearcher:
    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def search(
        self, *, query_text: str, k: int, filters: MetadataFilters | None = None
    ) -> list[RetrievedChunk]:
        async with self._session_factory() as session:
            rows = await search_chunks_fulltext(
                session, query_text=query_text, k=k, filters=filters
            )
        chunks = [
            RetrievedChunk(
                chunk_id=r._mapping["chunk_id"],
                document_id=r._mapping["document_id"],
                chunk_type=r._mapping["chunk_type"],
                content=r._mapping["content"],
                # ts_rank no es una distancia; lo guardamos negado para mantener la
                # convención "menor = mejor" del campo distance, pero NO se compara
                # con coseno: solo la posición importa (la fusión usa RRF).
                distance=-float(r._mapping["lexical_rank"]),
                metadata=r._mapping["metadata"],
            )
            for r in rows
        ]
        logger.info("rag.fulltext_search", hits=len(chunks), k=k)
        return chunks
