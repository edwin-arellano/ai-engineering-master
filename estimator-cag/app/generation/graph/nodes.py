"""Los cinco nodos del grafo de estimación (S13). Cada uno envuelve la lógica real
S9–S12 y devuelve una actualización parcial del estado. Nombres de nodo = enunciado;
semántica = dominio del repo. Un span Logfire por nodo."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import logfire
import structlog

from app.domain.agent_trace import AgentTrace
from app.domain.structured_estimation import (
    Coverage,
    EstimatedModule,
    StructuredEstimate,
    TaskEstimate,
)
from app.generation.agentic.flagging import flag_task_estimates
from app.generation.agentic.recovery_agent import run_task_hours_recovery_agent
from app.generation.graph.deps import GraphDeps
from app.generation.graph.state import EstimationState
from app.generation.rag.retrieval.per_task import estimate_task_hours
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.retrieval.structure import generate_skeleton

logger = structlog.get_logger(__name__)

Node = Callable[[EstimationState], Awaitable[dict]]


def _coverage(tasks: list[TaskEstimate]) -> Coverage:
    with_hist = sum(1 for t in tasks if not t.needs_human_input)
    return Coverage(
        with_history=with_hist,
        without_history=len(tasks) - with_hist,
        total=len(tasks),
    )


def _consolidate(
    modules: list[EstimatedModule], trace: AgentTrace | None = None
) -> StructuredEstimate:
    flat = [t for m in modules for t in m.tasks]
    return StructuredEstimate(
        modules=modules,
        coverage=_coverage(flat),
        total_suggested_hours=sum(t.suggested_hours or 0.0 for t in flat),
        agent_trace=trace,
    )


def _needs_review(estimate: StructuredEstimate) -> bool:
    """Determinista, sin LLM: hay revisión pendiente si tras la recuperación queda
    alguna tarea sin horas o marcada."""
    return any(
        t.needs_human_input or t.flag_reason is not None
        for m in estimate.modules
        for t in m.tasks
    )


def route_after_validation(state: EstimationState) -> str:
    """Router del Nivel 3: enruta por el status ya fijado en validate_and_consolidate."""
    return state.get("status") or "validated"


def make_nodes(deps: GraphDeps) -> dict[str, Node]:
    """Devuelve los 5 nodos como closures sobre `deps` (inyección sin tocar el estado)."""

    async def extract_requirements(state: EstimationState) -> dict:
        # Reformulación S9: transcript → ReformulatedQuery (sector/techs/search_text).
        with logfire.span("node: extract_requirements"):
            reformulated = reformulate_transcript(
                transcript=state["transcript"],
                wrapper=deps.wrapper,
                settings=deps.settings,
            )
            logger.info("graph.extract_requirements", sector=reformulated.sector)
            return {"reformulated": reformulated}

    async def classify_components(state: EstimationState) -> dict:
        # CAG S10: transcript → esqueleto de módulos/tareas (sin horas). Parte del
        # transcript, no del reformulated: el camino determinista por-tarea reconstruye
        # su propia query desde el título de cada tarea (diseño S10).
        with logfire.span("node: classify_components"):
            skeleton = generate_skeleton(
                transcript=state["transcript"],
                wrapper=deps.wrapper,
                settings=deps.settings,
            )
            logger.info("graph.classify_components", modules=len(skeleton.modules))
            return {"skeleton": skeleton}

    async def search_budgets(state: EstimationState) -> dict:
        # Retrieval determinista por-tarea (S10). SECUENCIAL a propósito (el directo lo
        # paraleliza con Send API). Puebla el reducer acumulador `task_estimates`.
        with logfire.span("node: search_budgets"):
            estimates: list[TaskEstimate] = []
            for module in state["skeleton"].modules:
                for task in module.tasks:
                    est = await estimate_task_hours(
                        task.title, pipeline=deps.pipeline, settings=deps.settings
                    )
                    estimates.append(est)
            logger.info("graph.search_budgets", tasks=len(estimates))
            return {"task_estimates": estimates}

    async def generate_estimate(state: EstimationState) -> dict:
        # Consolida: reagrupa por módulo (orden plano alineado con el skeleton),
        # cobertura y total. Sin recuperación todavía.
        with logfire.span("node: generate_estimate"):
            skeleton = state["skeleton"]
            it = iter(state["task_estimates"])
            modules = [
                EstimatedModule(name=m.name, tasks=[next(it) for _ in m.tasks])
                for m in skeleton.modules
            ]
            estimate = _consolidate(modules)
            # total_hours, no total: `Coverage` ya trae un campo `total` y colisionaría.
            logger.info(
                "graph.generate_estimate",
                **estimate.coverage.model_dump(),
                total_hours=estimate.total_suggested_hours,
            )
            return {"estimate": estimate}

    async def validate_and_consolidate(state: EstimationState) -> dict:
        # Flujo S12 completo dentro del grafo: flagging determinista → recuperación
        # agéntica (solo flaggeadas, gated) → re-consolidación → status.
        with logfire.span("node: validate_and_consolidate"):
            estimate = state["estimate"]
            flag_task_estimates(
                [t for m in estimate.modules for t in m.tasks], deps.settings
            )
            trace = None
            if deps.settings.agent_recovery_enabled:
                modules, trace = await run_task_hours_recovery_agent(
                    estimate.modules,
                    client=deps.client,
                    pipeline=deps.pipeline,
                    settings=deps.settings,
                )
                # El recovery muta las horas in-place → cobertura y total se recomputan.
                estimate = _consolidate(modules, trace=trace)
            status = "needs_review" if _needs_review(estimate) else "validated"
            logger.info("graph.validate_and_consolidate", status=status)
            return {"estimate": estimate, "agent_trace": trace, "status": status}

    return {
        "extract_requirements": extract_requirements,
        "classify_components": classify_components,
        "search_budgets": search_budgets,
        "generate_estimate": generate_estimate,
        "validate_and_consolidate": validate_and_consolidate,
    }
