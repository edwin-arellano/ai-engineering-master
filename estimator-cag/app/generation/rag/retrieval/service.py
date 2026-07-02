"""Orquesta las 5 fases del flujo RAG end-to-end: reformulación → retrieval →
augmentation → generación → verificación. Punto de entrada del endpoint de producto."""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.rag_estimation import RagEstimate
from app.domain.structured_estimation import (
    Coverage,
    EstimatedModule,
    StructuredEstimate,
    TaskEstimate,
)
from app.foundations.config import Settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.retrieval.augmentation import assemble_context
from app.generation.rag.retrieval.generation import generate_rag_estimate
from app.generation.rag.retrieval.per_task import estimate_task_hours
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.quality import (
    DegradationReport,
    anchor_line,
    apply_gate,
    gate_line,
    judge_lines,
    synthesize_range,
)
from app.generation.rag.retrieval.structure import generate_skeleton
from app.generation.rag.retrieval.verification import (
    CitationReport,
    CitationVerificationError,
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
    citation_report: CitationReport  # informe estructural de citaciones (S11)
    invalid_citations: list[str]  # back-compat = citation_report.dangling
    degradation_report: DegradationReport  # informe del gate de alucinaciones (S11)
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

    # 3. Augmentation (token budget + capa de calidad S11 opt-in: compresión/reorden).
    context = assemble_context(
        retrieval, max_tokens=settings.rag_max_context_tokens, settings=settings
    )

    # 4. Generación RAG-grounded.
    estimate = generate_rag_estimate(
        reformulated=reformulated, context=context, wrapper=wrapper, settings=settings
    )

    # 5. Verificación estructural por línea (grounded/dangling/insufficient).
    report = verify_citations(estimate, context)
    enforce_confidence_coherence(estimate)
    # Política configurable: por defecto detectar+reportar; con REJECT_ON_DANGLING, bloquear.
    if settings.reject_on_dangling and report.dangling:
        raise CitationVerificationError(
            f"citas colgantes (no estuvieron en el contexto): {report.dangling}"
        )

    # 6. Gate de alucinaciones (ancla numérica + juez) → degradar líneas a cero.
    degradation = DegradationReport(
        total_lines=0, degraded_lines=0, verified_lines=0, gates=[]
    )
    if settings.hallucination_gate_enabled:
        lines: list[dict] = []
        anchors: dict[int, object] = {}
        assumptions: dict[int, bool] = {}
        idx = 0
        for module in estimate.modules:
            for task in module.tasks:
                evidence = task.sources[0].evidence if task.sources else ""
                anchors[idx] = anchor_line(
                    line_engineer_days=task.engineer_days,
                    evidence=evidence,
                    hours_per_day=settings.hours_per_engineer_day,
                    tolerance=settings.numeric_deviation_tolerance,
                )
                assumptions[idx] = task.is_assumption
                lines.append(
                    {
                        "index": idx,
                        "title": task.title,
                        "engineer_days": task.engineer_days,
                        "evidence": evidence,
                    }
                )
                idx += 1
        verdicts = (
            await judge_lines(lines=lines, wrapper=wrapper, settings=settings)
            if settings.judge_enabled
            else {}
        )
        gates = {
            i: gate_line(
                index=i,
                is_assumption=assumptions[i],
                anchor=anchors[i],
                verdict=verdicts.get(i),
            )
            for i in anchors
        }
        estimate, degradation = apply_gate(estimate, gates)

    # 7. Síntesis de rangos por línea (single-pass): rango desde las fuentes citadas.
    if settings.synthesis_enabled:
        by_ref = {c.chunk_ref: c for c in retrieval.chunks}
        for module in estimate.modules:
            for task in module.tasks:
                hours = [
                    by_ref[s.source_id].metadata.get("estimated_hours")
                    for s in task.sources
                    if s.source_id in by_ref
                ]
                rng = synthesize_range(
                    [float(h) for h in hours if h is not None],
                    wrapper=wrapper,
                    settings=settings,
                    context=task.title,
                )
                if rng is not None:
                    task.hour_range = rng

    return EstimateFromTranscriptResult(
        estimate=estimate,
        retrieved_chunks=len(retrieval.chunks),
        retrieved_budget_ids=dedup_budget_ids(retrieval.chunks),
        context_tokens=context.token_count,
        citation_report=report,
        invalid_citations=report.dangling,
        degradation_report=degradation,
        search_time_ms=retrieval.search_time_ms,
        search_mode=search_mode,
        reranking=reranking,
    )


async def estimate_structured_from_transcript(
    *,
    transcript: str,
    wrapper: LLMWrapper,
    embedder: LiteLLMEmbedder,
    session_factory: async_sessionmaker,
    settings: Settings,
    search_mode: str | None = None,
    reranking: bool | None = None,
) -> StructuredEstimate:
    """Flujo invertido (S10): esqueleto (CAG, sin horas) → horas por-tarea (RAG
    determinista, consenso de vecinos + fiabilidad) → ensamblado con cobertura.

    Las horas NO las infiere el modelo: salen de la mediana de `estimated_hours` de los
    vecinos históricos. Las tareas sin match quedan con needs_human_input=True."""
    # search_mode/reranking del flujo por-tarea: default desde settings.per_task_*.
    if search_mode is not None:
        settings = settings.model_copy(update={"per_task_search_mode": search_mode})
    if reranking is not None:
        settings = settings.model_copy(update={"per_task_reranking": reranking})

    # Fase 1 — esqueleto (CAG, sin store, sin horas).
    skeleton = generate_skeleton(transcript=transcript, wrapper=wrapper, settings=settings)

    # Fase 2 — horas por-tarea (RAG determinista, en paralelo por tarea).
    pipeline = RetrievalPipeline(embedder=embedder, session_factory=session_factory)
    tasks_flat = [(m.name, t.title) for m in skeleton.modules for t in m.tasks]
    estimates = await asyncio.gather(
        *[
            estimate_task_hours(title, pipeline=pipeline, settings=settings)
            for _, title in tasks_flat
        ]
    )

    # Fase 3 (contrato) — re-agrupa por módulo, calcula cobertura y total.
    by_module: dict[str, list[TaskEstimate]] = {}
    iterator = iter(estimates)
    for module in skeleton.modules:
        by_module.setdefault(module.name, [])
        for _ in module.tasks:
            by_module[module.name].append(next(iterator))
    modules = [EstimatedModule(name=name, tasks=ts) for name, ts in by_module.items()]
    with_hist = sum(1 for e in estimates if not e.needs_human_input)
    total = sum(e.suggested_hours or 0.0 for e in estimates)
    coverage = Coverage(
        with_history=with_hist,
        without_history=len(estimates) - with_hist,
        total=len(estimates),
    )
    logger.info("structured.assembled", **coverage.model_dump(), total_hours=total)
    return StructuredEstimate(
        modules=modules, coverage=coverage, total_suggested_hours=total
    )
