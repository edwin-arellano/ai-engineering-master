"""Orquestación del flujo híbrido (S12): el determinista de S10 con el agente en dos fases.

La arquitectura es quirúrgica a propósito. El agente NO sustituye el flujo:

    fase 1 (agente)      propose_structure → EstimateSkeleton + traza
    ---- puerta humana ----  el cliente valida/edita la estructura
    fase 2a (determinista)   estimate_task_hours por tarea  (S10, sin tocar)
    fase 2b (determinista)   flag_task_estimates            (sin LLM)
    fase 2c (agente)         run_task_hours_recovery_agent  (solo las flaggeadas)

La puerta humana vive ENTRE los dos endpoints, no dentro de una función: por eso la fase 1 y
la 2 no se componen aquí en una sola llamada. Este módulo solo orquesta la fase 2 completa;
la 1 la expone `structure_agent`.
"""

from __future__ import annotations

import asyncio

import structlog
from openai import AsyncOpenAI

from app.domain.agent_trace import AgentTrace
from app.domain.structured_estimation import (
    Coverage,
    EstimatedModule,
    EstimateSkeleton,
    StructuredEstimate,
    TaskEstimate,
)
from app.foundations.config import Settings
from app.generation.agentic.flagging import flag_task_estimates
from app.generation.agentic.recovery_agent import run_task_hours_recovery_agent
from app.generation.rag.retrieval.per_task import estimate_task_hours
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)


def _group_by_module(
    skeleton: EstimateSkeleton, estimates: list[TaskEstimate]
) -> list[EstimatedModule]:
    """Re-agrupa las estimaciones planas por módulo, en el orden del esqueleto.

    Replica el ensamblado de `service.estimate_structured_from_transcript` (incluida la
    fusión de módulos homónimos) porque allí vive inline dentro de esa función y ese flujo
    no se toca en S12.
    """
    by_module: dict[str, list[TaskEstimate]] = {}
    iterator = iter(estimates)
    for module in skeleton.modules:
        by_module.setdefault(module.name, [])
        for _ in module.tasks:
            by_module[module.name].append(next(iterator))
    return [EstimatedModule(name=name, tasks=tasks) for name, tasks in by_module.items()]


def _assemble(modules: list[EstimatedModule], trace: AgentTrace | None) -> StructuredEstimate:
    """Cobertura y total desde el estado FINAL de las tareas (tras la recuperación).

    `with_history` cuenta las tareas con horas, no las que no necesitan revisión humana.
    En el flujo determinista (S10) da lo mismo, porque allí `needs_human_input` es
    exactamente "no hay horas". Tras la recuperación deja de serlo: una tarea recuperada con
    fiabilidad baja tiene horas Y necesita revisión, y contarla como "sin histórico" haría
    reportar `with_history=0` junto a un total de cientos de horas.
    """
    tasks = [task for module in modules for task in module.tasks]
    with_history = sum(1 for t in tasks if t.suggested_hours is not None)
    total = sum(t.suggested_hours or 0.0 for t in tasks)
    coverage = Coverage(
        with_history=with_history,
        without_history=len(tasks) - with_history,
        total=len(tasks),
    )
    logger.info("agent.structured.assembled", **coverage.model_dump(), total_hours=total)
    return StructuredEstimate(
        modules=modules,
        coverage=coverage,
        total_suggested_hours=total,
        agent_trace=trace,
    )


async def estimate_task_hours_agentic(
    skeleton: EstimateSkeleton,
    *,
    client: AsyncOpenAI,
    pipeline: RetrievalPipeline | None,
    settings: Settings,
    model: str | None = None,
    stub: bool = False,
) -> StructuredEstimate:
    """Fase 2 sobre un esqueleto ya validado por el humano: horas + flags + recuperación.

    Las horas de las tareas fiables las sigue derivando el retrieval determinista de S10
    (mediana de vecinos históricos), invocado tal cual. El agente solo toca las flaggeadas.
    """
    tasks_flat = [(m.name, t.title) for m in skeleton.modules for t in m.tasks]
    if not tasks_flat:
        return _assemble([], None)

    # Fase 2a — retrieval determinista por tarea (S10, en paralelo). Sin LLM, sin cambios.
    estimates = await asyncio.gather(
        *[
            estimate_task_hours(title, pipeline=pipeline, settings=settings)
            for _, title in tasks_flat
        ]
    )
    modules = _group_by_module(skeleton, list(estimates))

    # Fase 2b — flagging determinista: dónde el determinista no llegó.
    flag_task_estimates([t for m in modules for t in m.tasks], settings)

    # Fase 2c — recuperación agéntica, solo sobre las flaggeadas.
    trace: AgentTrace | None = None
    if settings.agent_recovery_enabled:
        modules, trace = await run_task_hours_recovery_agent(
            modules,
            client=client,
            pipeline=pipeline,
            settings=settings,
            model=model,
            stub=stub,
        )

    return _assemble(modules, trace)
