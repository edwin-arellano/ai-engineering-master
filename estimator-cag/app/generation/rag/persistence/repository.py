"""Operaciones de persistencia y búsqueda. La ingesta de un documento y todos sus
chunks ocurre en UNA sola transacción: si el embedder o el insert fallan, no quedan
documents huérfanos sin chunks.

S10: las búsquedas y la ingesta están parametrizadas por `model` (la clase ORM de la
colección destino: BudgetChunkRow | TranscriptChunkRow | TechnicalDocChunkRow). El
default es BudgetChunkRow por back-compat con los llamantes del camino de budgets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import cast, func, select, text
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.persistence.models import (
    EMBEDDING_DIM,
    BudgetChunkRow,
    ChunkRow,
    DocumentRow,
)
from app.generation.rag.schemas import EmbeddedChunk, MetadataFilters

# Sub-discriminadores (columna chunk_type) por colección.
# budgets:
BUDGET_COMPONENT = "budget_component"  # chunking estructural: un componente = un chunk.
HISTORICAL_TASK = "historical_task"  # chunking de tareas atómicas (S09): una tarea = un chunk.
# transcripts / technical (S10):
TRANSCRIPT_SEGMENT = "transcript_segment"  # un segmento temático de reunión = un chunk.
TECHNICAL_REFERENCE = "technical_reference"  # una sección de doc técnico = un chunk.


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
    model: type[ChunkRow] = BudgetChunkRow,
    source_path: str,
    document_type: str,
    document_metadata: dict[str, Any],
    embedded_chunks: Sequence[EmbeddedChunk],
    chunk_type: str = BUDGET_COMPONENT,
) -> tuple[int, int]:
    """Crea el documento y todos sus chunks en una transacción. Devuelve
    (document_id, chunks_created). `model` decide la tabla de colección que se puebla
    (default BudgetChunkRow por back-compat); `chunk_type` se persiste en la columna real."""
    document = DocumentRow(
        source_path=source_path,
        document_type=document_type,
        metadata_=document_metadata,
    )
    session.add(document)
    await session.flush()  # asigna document.id sin cerrar la transacción

    rows = [
        model(
            document_id=document.id,
            chunk_type=chunk_type,
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


def _halfvec_distance(model: type[ChunkRow], query_vector: list[float]):
    """Expresión de distancia coseno alineada con el índice HNSW half-vec de `model`.

    `embedding::halfvec(1536) <=> :q` debe ser IDÉNTICA a la indexada para que el
    planner use `<tabla>_embedding_halfvec_idx`: el cast a `halfvec(1536)` reproduce
    la expresión del índice y el operador `<=>` (cosine_distance) coincide con su
    operator class `halfvec_cosine_ops`. Fuente ÚNICA de esta expresión: todos los
    caminos de búsqueda la reutilizan para no desalinearse (verificar con EXPLAIN).
    """
    half_col = cast(model.embedding, HALFVEC(EMBEDDING_DIM))
    return half_col.cosine_distance(query_vector)


def _select_with_distance(model: type[ChunkRow], distance):
    """Proyección estándar de búsqueda (sin la columna `embedding`) + distancia."""
    return select(
        model.id.label("chunk_id"),
        model.document_id.label("document_id"),
        model.chunk_type.label("chunk_type"),
        model.content.label("content"),
        model.metadata_.label("metadata"),
        distance.label("distance"),
    )


def _build_halfvec_search_stmt(model: type[ChunkRow], query_vector: list[float], k: int):
    """Statement alineado con el índice HNSW half-vec de `model`.

    Reutiliza `_halfvec_distance`: la expresión de distancia es la indexada, así que
    el planner usa el índice (desalinear → seq scan silencioso; se verifica con
    EXPLAIN ANALYZE). NO proyecta `embedding`.
    """
    distance = _halfvec_distance(model, query_vector)
    return _select_with_distance(model, distance).order_by(distance).limit(k)


def _apply_metadata_filters(stmt, model: type[ChunkRow], filters: "MetadataFilters | None"):
    """Aplica filtros SQL deterministas sobre columnas y JSONB. No-op si filters es None.
    Los ejes JSONB (sector, year, tech) viven en <tabla>.metadata; chunk_type es columna.

    IMPORTANTE: solo añade cláusulas WHERE planas; NO toca la expresión de distancia
    del ORDER BY, así que el planner sigue pudiendo usar el índice HNSW.
    """
    if filters is None:
        return stmt
    md = model.metadata_
    if filters.chunk_types:
        stmt = stmt.where(model.chunk_type.in_(filters.chunk_types))
    if filters.sectors:
        # metadata->>'client_sector' IN (...)
        stmt = stmt.where(md["client_sector"].astext.in_([s for s in filters.sectors]))
    if filters.main_technology:
        stmt = stmt.where(md["main_technology"].astext == filters.main_technology)
    if filters.year_min is not None:
        stmt = stmt.where(md["year"].as_integer() >= filters.year_min)
    if filters.year_max is not None:
        stmt = stmt.where(md["year"].as_integer() <= filters.year_max)
    return stmt


async def search_chunks(
    session: AsyncSession,
    *,
    model: type[ChunkRow] = BudgetChunkRow,
    query_vector: list[float],
    k: int,
    ef_search: int | None = None,
    filters: "MetadataFilters | None" = None,
    distance_threshold: float | None = None,
) -> list[Row]:
    """Top-k chunks por distancia coseno (HNSW half-vec) de la colección `model`, con
    filtros de metadata y umbral de distancia opcionales. Sin filters ni threshold,
    idéntico a antes (back-compat con /search).

    IMPORTANTE — alineación operador/índice: el WHERE de los filtros es SQL plano
    sobre columnas/JSONB y NO altera la expresión `embedding::halfvec(1536) <=> :q`
    del ORDER BY, así que el planner sigue pudiendo usar el índice HNSW. El
    distance_threshold se aplica como WHERE sobre la MISMA expresión de distancia
    (no recalculada) para no romper esa alineación. Verifica con EXPLAIN ANALYZE.

    `ef_search` (parámetro query-time del HNSW) balancea recall/latencia; se aplica
    con SET LOCAL para acotarlo a la transacción de esta request (no contamina otras
    conexiones del pool). NO proyecta la columna `embedding` (solo la distancia).
    """
    if ef_search is not None:
        # SET no admite bind params para el valor; interpolamos un int validado.
        await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))

    # Sin filtros ni threshold: camino back-compat idéntico a /search.
    if filters is None and distance_threshold is None:
        result = await session.execute(_build_halfvec_search_stmt(model, query_vector, k))
        return list(result.all())

    distance = _halfvec_distance(model, query_vector)
    stmt = _select_with_distance(model, distance)
    stmt = _apply_metadata_filters(stmt, model, filters)
    if distance_threshold is not None:
        stmt = stmt.where(distance <= distance_threshold)
    stmt = stmt.order_by(distance).limit(k)

    result = await session.execute(stmt)
    return list(result.all())


def _build_exact_search_stmt(model: type[ChunkRow], query_vector: list[float], k: int):
    """Statement exacto (plain Vector, sin cast): top-k por fuerza bruta. Ground
    truth para medir el recall@k del índice HNSW aproximado."""
    distance = model.embedding.cosine_distance(query_vector)
    return (
        select(
            model.id.label("chunk_id"),
            model.document_id.label("document_id"),
            model.chunk_type.label("chunk_type"),
            model.content.label("content"),
            model.metadata_.label("metadata"),
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(k)
    )


async def search_chunks_exact(
    session: AsyncSession,
    *,
    model: type[ChunkRow] = BudgetChunkRow,
    query_vector: list[float],
    k: int,
) -> list[Row]:
    """Top-k exacto por fuerza bruta (seq scan) sobre la colección `model`. Ground
    truth para medir el recall del índice HNSW. Desactiva index/bitmap scan por si
    existiera un índice vectorial, forzando el recorrido secuencial sobre todos los vectores."""
    await session.execute(text("SET LOCAL enable_indexscan = off"))
    await session.execute(text("SET LOCAL enable_bitmapscan = off"))
    result = await session.execute(_build_exact_search_stmt(model, query_vector, k))
    return list(result.all())


async def search_chunks_fulltext(
    session: AsyncSession,
    *,
    model: type[ChunkRow] = BudgetChunkRow,
    query_text: str,
    k: int,
    filters: "MetadataFilters | None" = None,
) -> list[Row]:
    """Búsqueda léxica full-text (config 'spanish') sobre content_tsv de la colección
    `model`. Devuelve top-k por ts_rank. Misma proyección que search_chunks (sin
    embedding) para construir RetrievedChunk homogéneos. Reutiliza _apply_metadata_filters
    para coherencia con la rama vectorial (mismos filtros de sector/año/tech/chunk_type).

    `lexical_rank` (ts_rank) NO es comparable con la distancia coseno: por eso la
    fusión posterior usa RRF (solo posiciones), no las puntuaciones brutas.
    """
    tsquery = func.websearch_to_tsquery("spanish", query_text)
    rank = func.ts_rank(model.content_tsv, tsquery)
    stmt = (
        select(
            model.id.label("chunk_id"),
            model.document_id.label("document_id"),
            model.chunk_type.label("chunk_type"),
            model.content.label("content"),
            model.metadata_.label("metadata"),
            rank.label("lexical_rank"),
        )
        .where(model.content_tsv.op("@@")(tsquery))
    )
    stmt = _apply_metadata_filters(stmt, model, filters)
    stmt = stmt.order_by(rank.desc()).limit(k)
    result = await session.execute(stmt)
    return list(result.all())
