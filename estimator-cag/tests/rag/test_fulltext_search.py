"""Búsqueda léxica full-text (integration). Requiere Postgres migrado (0003) + corpus
re-ingestado. Confirma que la léxica encuentra identificadores exactos (OAuth, PSD2)
en el presupuesto correcto, donde la semántica los diluiría."""

from __future__ import annotations

import asyncio

import pytest

from app.generation.rag.schemas import MetadataFilters


def _fresh_sessionmaker():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.foundations.config import get_settings

    engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.integration
def test_fulltext_finds_exact_identifier() -> None:
    from app.generation.rag.retrieval.fulltext_search import FullTextSearcher

    async def run() -> list[str]:
        engine, factory = _fresh_sessionmaker()
        try:
            searcher = FullTextSearcher(factory)
            chunks = await searcher.search(query_text="OAuth PSD2", k=5)
            return [str(c.metadata.get("budget_id", "")) for c in chunks]
        finally:
            await engine.dispose()

    budget_ids = asyncio.run(run())
    assert budget_ids, "la léxica no devolvió nada para 'OAuth PSD2'"
    # BUD-2024-001 es el único presupuesto con OAuth 2.0 + PSD2.
    assert "BUD-2024-001" in budget_ids


@pytest.mark.integration
def test_fulltext_respects_metadata_filters() -> None:
    from app.generation.rag.retrieval.fulltext_search import FullTextSearcher

    async def run() -> tuple[list, list]:
        engine, factory = _fresh_sessionmaker()
        try:
            searcher = FullTextSearcher(factory)
            no_filter = await searcher.search(query_text="payment", k=10)
            finance_only = await searcher.search(
                query_text="payment", k=10,
                filters=MetadataFilters(sectors=["finance"]),
            )
            return no_filter, finance_only
        finally:
            await engine.dispose()

    no_filter, finance_only = asyncio.run(run())
    assert len(finance_only) <= len(no_filter)
    # Todos los resultados filtrados deben ser de sector finance.
    for chunk in finance_only:
        assert chunk.metadata.get("client_sector") == "finance"
