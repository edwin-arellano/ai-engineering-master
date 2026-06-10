"""Operaciones de persistencia y búsqueda. La ingesta de un documento y todos sus
chunks ocurre en UNA sola transacción: si el embedder o el insert fallan, no quedan
documents huérfanos sin chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.persistence.models import ChunkRow, DocumentRow
from app.generation.rag.schemas import EmbeddedChunk

# Tipo de chunk para chunking estructural: un componente de presupuesto = un chunk.
BUDGET_COMPONENT = "budget_component"


async def get_document_id_by_source_path(
    session: AsyncSession, source_path: str
) -> int | None:
    """Devuelve el id del documento con ese source_path, o None si no existe."""
    result = await session.execute(
        select(DocumentRow.id).where(DocumentRow.source_path == source_path)
    )
    return result.scalar_one_or_none()


async def ingest_document(
    session: AsyncSession,
    *,
    source_path: str,
    document_type: str,
    document_metadata: dict[str, Any],
    embedded_chunks: Sequence[EmbeddedChunk],
) -> tuple[int, int]:
    """Crea el documento y todos sus chunks en una transacción. Devuelve
    (document_id, chunks_created)."""
    document = DocumentRow(
        source_path=source_path,
        document_type=document_type,
        metadata_=document_metadata,
    )
    session.add(document)
    await session.flush()  # asigna document.id sin cerrar la transacción

    rows = [
        ChunkRow(
            document_id=document.id,
            chunk_type=BUDGET_COMPONENT,
            content=ec.text,
            embedding=ec.embedding,
            # Preservamos el chunk_id de origen ("BUD-...::AUTH-...") en el JSONB:
            # el esquema del ejercicio no tiene columna para él, pero no queremos perderlo.
            metadata_={**ec.metadata, "chunk_id": ec.chunk_id},
        )
        for ec in embedded_chunks
    ]
    session.add_all(rows)
    await session.commit()
    return document.id, len(rows)


async def search_chunks(
    session: AsyncSession,
    *,
    query_vector: list[float],
    k: int,
) -> list[Row]:
    """Top-k chunks por distancia coseno. NO proyecta la columna `embedding`
    (solo la distancia, calculada en servidor) — evita el decode del tipo vector
    en asyncpg y reduce el payload. Sequential scan: aún no hay índice (eso es el directo).
    """
    distance = ChunkRow.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            ChunkRow.id.label("chunk_id"),
            ChunkRow.document_id.label("document_id"),
            ChunkRow.chunk_type.label("chunk_type"),
            ChunkRow.content.label("content"),
            ChunkRow.metadata_.label("metadata"),
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )
    result = await session.execute(stmt)
    return list(result.all())
