"""Fase 2b del flujo híbrido (S12): recuperación agéntica de las tareas flaggeadas.

El agente NO sustituye la búsqueda vectorial: el retrieval determinista (S10) ya resolvió las
tareas con buen histórico, y es barato y reproducible. Aquí el agente entra SOLO donde aquel
se quedó corto — las tareas que `flagging` marcó — y por cada una corre un bucle
reason→act→observe con `search_budgets`: puede reformular la consulta, razonar el consenso
cuando las fuentes discrepan, y reportar honestamente que no hay dato en vez de inventarlo.

Las tareas no flaggeadas ni las mira: pasan intactas al resultado final.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.domain.agent_trace import AgentStep, AgentTrace
from app.domain.structured_estimation import EstimatedModule, Reliability, TaskEstimate
from app.foundations.config import Settings
from app.foundations.prompts.loader import render_recovery_agent_prompt
from app.generation.agentic.agent_loop import run_loop
from app.generation.agentic.agent_tools import build_tool_registry, build_tools
from app.generation.rag.quality.synthesis import coefficient_of_variation
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)


class RecoveredHours(BaseModel):
    """`text_format` del bucle de recuperación: lo que el agente devuelve por tarea.

    `suggested_hours=None` es una respuesta VÁLIDA y esperada: significa "no hay evidencia
    histórica", que es justo lo que queremos en vez de una cifra plausible inventada.
    """

    model_config = ConfigDict(extra="forbid")

    suggested_hours: float | None = Field(default=None, ge=0)
    reliability: Literal["none", "low", "medium", "high"]
    reference_budget_ids: list[str] = Field(default_factory=list)
    note: str


def _recovery_tools(settings: Settings) -> list[dict[str, Any]]:
    """Solo `search_budgets`, reutilizando su schema del one-shot.

    `calculate_estimate` y `validate_estimate` razonan sobre un proyecto entero (mediana +
    contingencia, cuadre del total); aquí el agente mira UNA tarea, así que dárselas solo
    invita a usarlas mal. La lista de tools también es prompt.
    """
    return [t for t in build_tools(settings) if t["name"] == "search_budgets"]


def _log_consensus_check(task_title: str, steps: list[AgentStep], recovered: float | None) -> None:
    """Contrasta lo recuperado con la dispersión real de lo observado. Solo log.

    Deliberadamente NO fuerza el número: el consenso lo razona el agente (puede ponderar
    scope o antigüedad como no lo hace una mediana), y lo que auditamos es precisamente ese
    razonamiento. Esto deja el rastro para detectar si se aleja de sus propias fuentes.
    """
    observed: list[float] = []
    for step in steps:
        for item in step.observation.get("items", []) or []:
            hours = item.get("estimated_hours")
            if hours is not None:
                observed.append(float(hours))
    if len(observed) < 2:
        return
    logger.info(
        "agent.recovery.consensus_check",
        task=task_title,
        sources=len(observed),
        min_hours=min(observed),
        max_hours=max(observed),
        dispersion=round(coefficient_of_variation(observed), 3),
        recovered_hours=recovered,
    )


async def _recover_task(
    module_name: str,
    task: TaskEstimate,
    *,
    client: AsyncOpenAI,
    model: str,
    instructions: str,
    tools: list[dict[str, Any]],
    registry: dict[str, Any],
    settings: Settings,
) -> list[AgentStep]:
    """Corre el bucle sobre UNA tarea flaggeada y aplica el resultado in-place."""
    user_msg = (
        f"Module: {module_name}\n"
        f"Task: {task.title}\n"
        f"Why deterministic retrieval failed: {task.flag_reason}\n"
        f"Recover the effort for this task in engineer-hours."
    )
    parsed, steps, status = await run_loop(
        client=client,
        model=model,
        instructions=instructions,
        initial_input=[{"role": "user", "content": user_msg}],
        tools=tools,
        registry=registry,
        reasoning_cfg={
            "effort": settings.agent_reasoning_effort,
            "summary": settings.agent_reasoning_summary,
        },
        text_format=RecoveredHours,
        max_steps=settings.agent_recovery_max_steps,
        phase="recovery",
    )

    if status == "done" and parsed is not None:
        _log_consensus_check(task.title, steps, parsed.suggested_hours)
        task.suggested_hours = parsed.suggested_hours
        if parsed.suggested_hours is None:
            # Sin dato: NO se inventa. Queda para el humano y conserva su flag.
            task.reliability = Reliability.NONE
            task.needs_human_input = True
        else:
            task.reliability = Reliability(parsed.reliability)
            task.needs_human_input = task.reliability in {Reliability.NONE, Reliability.LOW}
            task.flag_reason = None  # recuperada: deja de estar pendiente
    else:
        # Bucle agotado: la tarea se queda como la dejó el determinista, con su flag.
        task.needs_human_input = True

    logger.info(
        "agent.recovery.task",
        task=task.title,
        status=status,
        steps=len(steps),
        recovered=task.flag_reason is None,
    )
    return steps


async def run_task_hours_recovery_agent(
    modules: list[EstimatedModule],
    *,
    client: AsyncOpenAI,
    pipeline: RetrievalPipeline | None,
    settings: Settings,
    model: str | None = None,
    stub: bool = False,
) -> tuple[list[EstimatedModule], AgentTrace]:
    """Recupera las tareas flaggeadas de `modules` (in-place) y devuelve la traza conjunta.

    Recibe los módulos y no una lista plana de tareas para poder darle al agente el módulo al
    que pertenece cada una (contexto que cambia lo que busca) sin depender de que los títulos
    sean únicos entre módulos.
    """
    model = model or settings.agent_model
    flagged = [
        (module.name, task)
        for module in modules
        for task in module.tasks
        if task.flag_reason is not None
    ]
    if not flagged:
        logger.info("agent.recovery.skipped", reason="no flagged tasks")
        return modules, AgentTrace(agent=settings.agent_profile_name, phase="recovery")

    tools = _recovery_tools(settings)
    registry = build_tool_registry(pipeline=pipeline, settings=settings, stub=stub)
    instructions = render_recovery_agent_prompt(
        settings.recovery_agent_prompt_version, agent=settings.agent_profile_name
    )
    logger.info("agent.recovery.start", flagged=len(flagged), model=model)

    # Una tarea flaggeada no depende de otra: bucles en paralelo (mismo criterio que el
    # retrieval por-tarea de S10).
    per_task_steps = await asyncio.gather(
        *(
            _recover_task(
                module_name,
                task,
                client=client,
                model=model,
                instructions=instructions,
                tools=tools,
                registry=registry,
                settings=settings,
            )
            for module_name, task in flagged
        )
    )

    # `run_loop` numera desde 1 en cada tarea: renumeramos para que la traza conjunta se lea
    # como una secuencia única.
    steps = [
        step.model_copy(update={"step": i})
        for i, step in enumerate(
            (step for task_steps in per_task_steps for step in task_steps), start=1
        )
    ]
    recovered = sum(1 for _, task in flagged if task.flag_reason is None)
    logger.info(
        "agent.recovery.done", flagged=len(flagged), recovered=recovered, steps=len(steps)
    )
    return modules, AgentTrace(
        agent=settings.agent_profile_name, phase="recovery", steps=steps
    )
