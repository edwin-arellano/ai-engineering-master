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
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.retrieval.verification import (
    enforce_confidence_coherence,
    verify_citations,
)

logger = structlog.get_logger(__name__)


def dedup_budget_ids(chunks) -> list[str]:
    """budget_id de los chunks recuperados, deduplicados preservando orden (un budget
    puede aportar varios chunks; la métrica del golden set es por presupuesto)."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk in chunks:
        bid = str(chunk.metadata.get("budget_id", ""))
        if bid and bid not in seen:
            seen.add(bid)
            out.append(bid)
    return out


class EstimateFromTranscriptResult(BaseModel):
    """Respuesta del endpoint: la estimación + diagnóstico del retrieval para trazabilidad."""

    estimate: RagEstimate
    retrieved_chunks: int
    retrieved_budget_ids: list[str]
    context_tokens: int
    invalid_citations: list[str]
    search_time_ms: int
    # Config efectiva que corrió (trazabilidad de las 4 configuraciones A/B/C/D).
    search_mode: str
    reranking: bool


async def estimate_from_transcript(
    *,
    transcript: str,
    wrapper: LLMWrapper,
    embedder: LiteLLMEmbedder,
    session_factory: async_sessionmaker,
    settings: Settings,
    search_mode: str | None = None,
    reranking: bool | None = None,
) -> EstimateFromTranscriptResult:
    # Resolución de defaults desde Settings (None → valor de producción configurado).
    search_mode = search_mode or settings.rag_search_mode
    reranking = settings.reranking_enabled if reranking is None else reranking

    # 1. Reformulación (LLM barato; bloqueante → thread no necesario, Instructor es sync
    #    pero la llamada es corta; si se quiere, envolver en asyncio.to_thread).
    reformulated = reformulate_transcript(transcript=transcript, wrapper=wrapper, settings=settings)

    # 2. Retrieval (vector|hybrid + rerank opcional, según config).
    pipeline = RetrievalPipeline(embedder=embedder, session_factory=session_factory)
    retrieval = await pipeline.retrieve(
        reformulated=reformulated, settings=settings,
        search_mode=search_mode, reranking=reranking,
    )

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
        retrieved_budget_ids=dedup_budget_ids(retrieval.chunks),
        context_tokens=context.token_count,
        invalid_citations=invalid,
        search_time_ms=retrieval.search_time_ms,
        search_mode=search_mode,
        reranking=reranking,
    )
