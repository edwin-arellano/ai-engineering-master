"""Test de integración del bucle agéntico (S12). Pega a la Responses API real.
Usa gpt-5-mini + transcripción compleja (barato). Marca: @pytest.mark.integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from openai import AsyncOpenAI

from app.foundations.config import get_settings
from app.generation.agentic.agent import run_agent
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

pytestmark = pytest.mark.integration


async def test_agente_transcripcion_compleja_stub():
    settings = get_settings()
    transcript = Path("examples/transcripts/sample_transcript_complex.txt").read_text("utf-8")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    pipeline = RetrievalPipeline(embedder=LiteLLMEmbedder(), session_factory=AsyncSessionLocal)
    result = await run_agent(
        transcript,
        client=client,
        pipeline=pipeline,
        settings=settings,
        model=settings.agent_debug_model,
        stub=True,
    )
    assert result.status == "done"
    # >1 componente y >1 búsqueda (criterio del ejercicio)
    searches = [s for s in result.trace if s.action == "search_budgets"]
    assert len(searches) > 1
    assert any(s.action == "calculate_estimate" for s in result.trace)
    assert result.estimate is not None and len(result.estimate.components) > 1
    if settings.agent_validate_enabled:
        assert any(s.action == "validate_estimate" for s in result.trace)
