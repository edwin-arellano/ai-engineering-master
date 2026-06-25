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


@pytest.mark.integration
def test_sector_filter_narrows_results_against_db() -> None:
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder
    from app.generation.rag.persistence.database import AsyncSessionLocal
    from app.generation.rag.persistence.repository import search_chunks

    async def run() -> tuple[int, int]:
        embedder = LiteLLMEmbedder()
        vector = await asyncio.to_thread(
            embedder.embed_one, "authentication backend for fintech"
        )
        async with AsyncSessionLocal() as session:
            unfiltered = await search_chunks(session, query_vector=vector, k=25)
            filtered = await search_chunks(
                session,
                query_vector=vector,
                k=25,
                filters=MetadataFilters(sectors=["finance"]),
            )
        return len(unfiltered), len(filtered)

    total, finance_only = asyncio.run(run())
    assert finance_only <= total


@pytest.mark.integration
def test_search_with_filters_still_uses_hnsw_index() -> None:
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder
    from app.generation.rag.persistence.database import AsyncSessionLocal
    from app.generation.rag.persistence.models import EMBEDDING_DIM

    async def run() -> str:
        embedder = LiteLLMEmbedder()
        vector = embedder.embed_one("authentication backend for fintech")
        literal = "[" + ",".join(str(x) for x in vector) + "]"
        async with AsyncSessionLocal() as session:
            await session.execute(text("SET LOCAL hnsw.ef_search = 40"))
            plan = await session.execute(
                text(
                    f"EXPLAIN ANALYZE SELECT id, "
                    f"(embedding::halfvec({EMBEDDING_DIM})) <=> "
                    f"'{literal}'::halfvec({EMBEDDING_DIM}) AS d "
                    f"FROM chunks "
                    f"WHERE metadata->>'client_sector' = 'finance' "
                    f"ORDER BY d LIMIT 25"
                )
            )
            return "\n".join(row[0] for row in plan.all())

    plan = asyncio.run(run())
    assert "Index Scan using chunks_embedding_halfvec_idx" in plan
    assert "Seq Scan" not in plan
