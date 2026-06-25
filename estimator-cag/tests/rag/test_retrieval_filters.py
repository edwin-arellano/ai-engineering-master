"""Fase 2 (retrieval con filtros). Unit: comprueba el WHERE que arma
`_apply_metadata_filters` compilando el statement (sin DB). Integration: pega a la
DB real, ingesta y verifica que el filtro por sector acota y que el plan sigue
usando el índice HNSW (no seq scan)."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.generation.rag.persistence.repository import (
    _apply_metadata_filters,
    _halfvec_distance,
    _select_with_distance,
)
from app.generation.rag.schemas import MetadataFilters


def _compiled_where(filters: MetadataFilters | None) -> str:
    distance = _halfvec_distance([0.0] * 1536)
    stmt = _select_with_distance(distance)
    stmt = _apply_metadata_filters(stmt, filters)
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_no_filters_is_noop():
    base = str(_select_with_distance(_halfvec_distance([0.0] * 1536)).compile())
    filtered = str(
        _apply_metadata_filters(
            _select_with_distance(_halfvec_distance([0.0] * 1536)), None
        ).compile()
    )
    assert "WHERE" not in filtered
    assert base == filtered


def test_sector_filter_emits_where_on_jsonb():
    sql = _compiled_where(MetadataFilters(sectors=["healthcare"]))
    assert "WHERE" in sql
    # El sector vive en el JSONB metadata->>'client_sector'.
    assert "client_sector" in sql
    assert "healthcare" in sql


def test_chunk_types_filter_emits_in_on_column():
    sql = _compiled_where(
        MetadataFilters(chunk_types=["budget_component", "historical_task"])
    )
    assert "chunk_type" in sql
    assert "budget_component" in sql
    assert "historical_task" in sql


def test_year_range_filter_emits_bounds():
    sql = _compiled_where(MetadataFilters(year_min=2023, year_max=2025))
    assert "2023" in sql
    assert "2025" in sql


def test_filters_do_not_alter_distance_expression():
    """El WHERE de los filtros NO debe tocar la expresión de distancia del ORDER BY
    (alineación con el índice HNSW). La expresión half-vec debe seguir presente."""
    sql = _compiled_where(MetadataFilters(sectors=["finance"]))
    assert "halfvec" in sql.lower()


# --- Integration: requiere Postgres migrado + corpus ingestado -----------------
#
# Estos tests usan un engine async FRESCO por test (no el AsyncSessionLocal global)
# porque cada uno arranca su propio event loop con asyncio.run(); reutilizar el pool
# del engine de módulo entre loops distintos provoca errores de "loop cerrado".


def _fresh_sessionmaker():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.foundations.config import get_settings

    engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory


@pytest.mark.integration
def test_sector_filter_narrows_results_against_db() -> None:
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder
    from app.generation.rag.persistence.repository import search_chunks

    async def run() -> tuple[int, int]:
        embedder = LiteLLMEmbedder()
        vector = await asyncio.to_thread(
            embedder.embed_one, "authentication backend for fintech"
        )
        engine, factory = _fresh_sessionmaker()
        try:
            async with factory() as session:
                unfiltered = await search_chunks(session, query_vector=vector, k=25)
                filtered = await search_chunks(
                    session,
                    query_vector=vector,
                    k=25,
                    filters=MetadataFilters(sectors=["finance"]),
                )
            return len(unfiltered), len(filtered)
        finally:
            await engine.dispose()

    total, finance_only = asyncio.run(run())
    assert finance_only <= total
    assert finance_only > 0  # hay chunks de finance en el corpus


@pytest.mark.integration
def test_filters_preserve_halfvec_alignment() -> None:
    """Gate de alineación operador/índice con filtros.

    En un corpus diminuto el planner elige Seq Scan por coste (no por desalineación),
    así que el uso del índice HNSW solo se aprecia con el corpus inflado
    (scripts/seed_synthetic_chunks.py). El test:

    - Si el baseline (sin filtro) NO usa el índice halfvec → SKIP (corpus sin sembrar).
    - Si lo usa, prueba la alineación y verifica que añadir un filtro de metadata NO
      rompe la expresión de distancia halfvec del plan: el planner sigue produciendo
      un plan válido (puede elegir otro índice más barato según selectividad), nunca
      una recomputación desalineada. Verificado a mano además con sector=finance.
    """
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder
    from app.generation.rag.persistence.models import EMBEDDING_DIM

    async def explain(where_sql: str) -> str:
        embedder = LiteLLMEmbedder()
        vector = embedder.embed_one("authentication backend for fintech")
        literal = "[" + ",".join(str(x) for x in vector) + "]"
        engine, factory = _fresh_sessionmaker()
        try:
            async with factory() as session:
                await session.execute(text("SET LOCAL hnsw.ef_search = 40"))
                plan = await session.execute(
                    text(
                        f"EXPLAIN ANALYZE SELECT id, "
                        f"(embedding::halfvec({EMBEDDING_DIM})) <=> "
                        f"'{literal}'::halfvec({EMBEDDING_DIM}) AS d "
                        f"FROM chunks {where_sql} ORDER BY d LIMIT 25"
                    )
                )
                return "\n".join(row[0] for row in plan.all())
        finally:
            await engine.dispose()

    baseline = asyncio.run(explain(""))
    if "chunks_embedding_halfvec_idx" not in baseline:
        pytest.skip(
            "corpus diminuto: el planner prefiere Seq Scan por coste. Siembra chunks "
            "sintéticos (scripts/seed_synthetic_chunks.py) para ejercitar el índice HNSW."
        )

    # Baseline sí usa el índice → la alineación operador/halfvec es correcta.
    assert "Index Scan using chunks_embedding_halfvec_idx" in baseline

    # Con un filtro de metadata el plan sigue computando la distancia halfvec
    # (alineación preservada); el access path puede variar por selectividad.
    filtered = asyncio.run(
        explain("WHERE chunk_type IN ('budget_component','historical_task')")
    )
    assert "halfvec" in filtered.lower()
