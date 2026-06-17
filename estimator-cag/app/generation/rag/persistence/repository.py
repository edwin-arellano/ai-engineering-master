"""Operaciones de persistencia y búsqueda. La ingesta de un documento y todos sus
chunks ocurre en UNA sola transacción: si el embedder o el insert fallan, no quedan
documents huérfanos sin chunks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import cast, select, text
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.persistence.models import EMBEDDING_DIM, ChunkRow, DocumentRow
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


def _build_halfvec_search_stmt(query_vector: list[float], k: int):
    """Statement alineado con el índice HNSW half-vec (chunks_embedding_halfvec_idx).

    La expresión `embedding::halfvec(1536) <=> :q` debe ser IDÉNTICA a la indexada
    para que el planner use el índice: el cast a `halfvec(1536)` reproduce la
    expresión del índice y el operador `<=>` (cosine_distance) coincide con su
    operator class `halfvec_cosine_ops`. Desalinear cualquiera de los dos → Postgres
    cae a seq scan sin avisar (se verifica con EXPLAIN ANALYZE). NO proyecta `embedding`.
    """
    half_col = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIM))
    distance = half_col.cosine_distance(query_vector)
    return (
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


async def search_chunks(
    session: AsyncSession,
    *,
    query_vector: list[float],
    k: int,
    ef_search: int | None = None,
) -> list[Row]:
    """Top-k chunks por distancia coseno usando el índice HNSW half-vec.

    `ef_search` (parámetro query-time del HNSW) balancea recall/latencia; se aplica
    con SET LOCAL para acotarlo a la transacción de esta request (no contamina otras
    conexiones del pool). NO proyecta la columna `embedding` (solo la distancia,
    calculada en servidor) — evita el decode del tipo vector en asyncpg.
    """
    if ef_search is not None:
        # SET no admite bind params para el valor; interpolamos un int validado.
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
    result = await session.execute(_build_halfvec_search_stmt(query_vector, k))
    return list(result.all())


def _build_exact_search_stmt(query_vector: list[float], k: int):
    """Statement exacto (plain Vector, sin cast): top-k por fuerza bruta. Ground
    truth para medir el recall@k del índice HNSW aproximado."""
    distance = ChunkRow.embedding.cosine_distance(query_vector)
    return (
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


async def search_chunks_exact(
    session: AsyncSession, *, query_vector: list[float], k: int
) -> list[Row]:
    """Top-k exacto por fuerza bruta (seq scan). Ground truth para medir el recall
    del índice HNSW. Desactiva index/bitmap scan por si existiera un índice vectorial,
    forzando el recorrido secuencial sobre todos los vectores."""
    await session.execute(text("SET LOCAL enable_indexscan = off"))
    await session.execute(text("SET LOCAL enable_bitmapscan = off"))
    result = await session.execute(_build_exact_search_stmt(query_vector, k))
    return list(result.all())
