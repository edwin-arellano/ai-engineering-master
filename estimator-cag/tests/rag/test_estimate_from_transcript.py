"""Flujo RAG end-to-end (integration). Pega a DB + LLM reales con una transcripción
de ejemplo. Requiere: Postgres migrado (0002), corpus re-ingestado con los dos
chunk_types (scripts/reingest_with_tasks.py) y API keys configuradas."""

from __future__ import annotations

import asyncio

import pytest

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.retrieval import estimate_from_transcript

_TRANSCRIPT = """
Hola, gracias por la llamada. Os cuento: somos una fintech española y queremos
construir una API de banca móvil. Lo crítico es la autenticación OAuth 2.0 con
gestión de sesiones por JWT, aislamiento de tokens multi-tenant y rate limiting
por cliente. Además necesitamos cumplir PSD2: Strong Customer Authentication y
gestión de consentimiento. También un ledger de transacciones con conciliación.
El stack que veníamos usando es Ruby on Rails con PostgreSQL y Redis. Operamos
en España de momento. No hay deadline duro pero queremos una estimación realista.
""".strip()


@pytest.mark.integration
def test_estimate_from_transcript_end_to_end() -> None:
    settings = get_settings()
    wrapper = LLMWrapper(settings)

    async def run():
        # Engine FRESCO por test: evita reutilizar el pool del AsyncSessionLocal
        # global entre event loops distintos (asyncio.run de varios tests).
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            return await estimate_from_transcript(
                transcript=_TRANSCRIPT,
                wrapper=wrapper,
                embedder=LiteLLMEmbedder(),
                session_factory=factory,
                settings=settings,
            )
        finally:
            await engine.dispose()

    result = asyncio.run(run())

    # Con corpus poblado, debería poder estimar (no insufficient).
    assert result.estimate.confidence.value != "insufficient"
    assert result.estimate.modules, "se esperaban módulos con corpus poblado"
    # Las citations no deben inventarse.
    assert result.invalid_citations == []
    # Trazabilidad del retrieval.
    assert result.retrieved_chunks > 0
    assert result.context_tokens > 0
