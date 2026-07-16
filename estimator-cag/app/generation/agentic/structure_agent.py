"""Fase 1 del flujo híbrido (S12): proponer la estructura módulos/tareas con el agente Neo.

Sin tools y un solo step: aquí no hay nada que buscar, solo criterio de arquitectura sobre
lo que dice la transcripción. El valor que aporta el agente frente al `generate_skeleton`
determinista (S10) es el modelo de razonamiento y, sobre todo, la traza: el reasoning summary
del turno queda como único `AgentStep`, y con él se audita POR QUÉ propuso esos módulos.

Reutiliza `EstimateSkeleton` como formato de salida — el agente se adapta al contrato de la
app, no al revés. Detrás de esta fase va la puerta humana: el cliente revisa y edita la
estructura antes de pedir horas (por eso fase 1 y fase 2 son endpoints distintos).
"""

from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from app.domain.agent_trace import AgentStep, AgentTrace
from app.domain.structured_estimation import EstimateSkeleton, StructureProposal
from app.foundations.config import Settings
from app.foundations.prompts.loader import render_structure_agent_prompt
from app.generation.agentic.agent import partition_output

logger = structlog.get_logger(__name__)


async def propose_structure(
    transcript: str,
    *,
    client: AsyncOpenAI,
    settings: Settings,
    model: str | None = None,
) -> StructureProposal:
    """Transcripción → esqueleto de módulos/tareas SIN horas, con la traza del agente.

    Recibe la transcripción cruda, igual que el `generate_skeleton` determinista al que
    reemplaza en este flujo: la reformulación (S09) produce un brief tipado para buscar en
    el store, no un texto que estructurar, y aquí no se busca nada.
    """
    model = model or settings.agent_model
    system_prompt = render_structure_agent_prompt(
        settings.structure_agent_prompt_version, agent=settings.agent_profile_name
    )
    response = await client.responses.parse(
        model=model,
        reasoning={
            "effort": settings.agent_reasoning_effort,
            "summary": settings.agent_reasoning_summary,
        },
        instructions=system_prompt,
        input=[{"role": "user", "content": transcript}],
        text_format=EstimateSkeleton,  # mismo contrato que el flujo determinista
    )

    skeleton: EstimateSkeleton = response.output_parsed
    turn_reasoning, _ = partition_output(response)  # sin tools ⇒ nunca hay function_call
    n_tasks = sum(len(m.tasks) for m in skeleton.modules)
    trace = AgentTrace(
        agent=settings.agent_profile_name,
        phase="structure",
        steps=[
            AgentStep(
                step=1,
                reasoning=turn_reasoning,
                action="propose_structure",
                args={},
                observation={"modules": len(skeleton.modules), "tasks": n_tasks},
            )
        ],
    )
    logger.info(
        "agent.structure.done", modules=len(skeleton.modules), tasks=n_tasks, model=model
    )
    return StructureProposal(skeleton=skeleton, agent_trace=trace)
