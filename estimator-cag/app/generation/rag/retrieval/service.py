"""Orquesta las 5 fases del flujo RAG end-to-end: reformulación → retrieval →
augmentation → generación → verificación. Punto de entrada del endpoint de producto."""

from __future__ import annotations

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.rag_estimation import RagEstimate
from app.foundations.config import Settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.retrieval.augmentation import assemble_context
from app.generation.rag.retrieval.generation import generate_rag_estimate
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.retrieval.retriever import RagRetriever
from app.generation.rag.retrieval.verification import (
    enforce_confidence_coherence,
    verify_citations,
)

logger = structlog.get_logger(__name__)


class EstimateFromTranscriptResult(BaseModel):
    """Respuesta del endpoint: la estimación + diagnóstico del retrieval para trazabilidad."""

    estimate: RagEstimate
    retrieved_chunks: int
    context_tokens: int
    invalid_citations: list[str]
    search_time_ms: int


async def estimate_from_transcript(
    *,
    transcript: str,
    wrapper: LLMWrapper,
    embedder: LiteLLMEmbedder,
    session_factory: async_sessionmaker,
    settings: Settings,
) -> EstimateFromTranscriptResult:
    # 1. Reformulación (LLM barato; bloqueante → thread no necesario, Instructor es sync
    #    pero la llamada es corta; si se quiere, envolver en asyncio.to_thread).
    reformulated = reformulate_transcript(transcript=transcript, wrapper=wrapper, settings=settings)

    # 2. Retrieval (filtros + threshold).
    retriever = RagRetriever(embedder=embedder, session_factory=session_factory)
    retrieval = await retriever.retrieve(reformulated=reformulated, settings=settings)

    # 3. Augmentation (token budget).
    context = assemble_context(retrieval, max_tokens=settings.rag_max_context_tokens)

    # 4. Generación RAG-grounded.
    estimate = generate_rag_estimate(
        reformulated=reformulated, context=context, wrapper=wrapper, settings=settings
    )

    # 5. Verificación.
    invalid = verify_citations(estimate, context)
    enforce_confidence_coherence(estimate)

    return EstimateFromTranscriptResult(
        estimate=estimate,
        retrieved_chunks=len(retrieval.chunks),
        context_tokens=context.token_count,
        invalid_citations=invalid,
        search_time_ms=retrieval.search_time_ms,
    )
