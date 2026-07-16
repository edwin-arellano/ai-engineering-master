"""Agente de estimación (S12): bucle manual sobre la Responses API de OpenAI.

Sin framework. Cuatro piezas: tools (agent_tools), el modelo (gpt-5, orquestador), el
bucle (este módulo) y el estado/traza (AgentStep acumulados). El razonamiento vive dentro
del modelo (reasoning); lo exponemos vía reasoning summaries para la traza. El control de
flujo es nuestro: nº de pasos, qué hacer si una tool falla, qué devolver si no converge.

Contrato de salida DETERMINISTA (AgentEstimate) aunque el camino del bucle varíe. El
encadenado es con previous_response_id (requiere store=True, el default). Para orgs ZDR
(store=False) habría que reenviar la lista de items en cada vuelta (ver notas).
"""

from __future__ import annotations

import asyncio
import json

import structlog
from openai import AsyncOpenAI

from app.foundations.config import Settings
from app.generation.agentic.agent_schemas import AgentEstimate, AgentResult, AgentStep
from app.generation.agentic.agent_tools import build_tool_registry, build_tools, execute_tool
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)


def build_system_prompt(settings: Settings) -> str:
    """System prompt del agente (EN INGLÉS). Rol + método + tools disponibles."""
    validate_step = (
        "\n4. Call validate_estimate as the LAST step before answering, passing the "
        "candidate components (name, estimated_hours, reference_budget_ids) and "
        "total_hours. If it reports issues, fix them (search again for unbudgeted "
        "components) and re-validate."
        if settings.agent_validate_enabled
        else ""
    )
    return (
        "You are a software estimation agent. You receive a raw meeting transcript and "
        "must produce a structured, per-component effort estimate in engineer-hours.\n\n"
        "Method:\n"
        "1. Read the transcript and DECOMPOSE the project into distinct components. "
        "Independent pieces (for example a business backend, an ERP integration, a mobile "
        "app, an analytics dashboard) are separate components.\n"
        "2. For EACH component, call search_budgets with a focused, self-contained query "
        "— one component per call. Never merge unrelated components into a single search. "
        "If a search returns NO matches, reformulate it ONCE with different terms. Do not "
        "reformulate a search that already returned matches, and never repeat a query you "
        "already ran: search_budgets returns AT MOST two rounds per component.\n"
        "The `confidence` field is informational, not a gate. The historical corpus rarely "
        "matches a new project exactly, so `low` confidence is normal and expected: take "
        "the closest items you found, note the weak match in your notes, and MOVE ON. "
        "Never keep searching in the hope of a better score.\n"
        "3. When you have searched every component (at most twice each), call "
        "calculate_estimate once, passing every component with its reference_amounts (the "
        "estimated_hours of the items search_budgets returned). It is deterministic; it "
        "does not guess."
        f"{validate_step}\n"
        "5. Produce the final structured estimate: one entry per component with its hours "
        "and the budget_ids that back it, the overall total, and brief notes. Mark as "
        "unbudgeted any component with no historical reference — never invent numbers."
    )


def _partition_output(response) -> tuple[str, list]:
    """Extrae (reasoning_summary_del_turno, [function_call...]) de response.output.

    La salida de la Responses API es una lista de items tipados; hay que recorrerla e
    inspeccionar por `type`, no asumir posiciones. Los items `reasoning` traen `summary`.
    """
    reasoning_texts: list[str] = []
    calls: list = []
    for item in response.output:
        if item.type == "reasoning":
            reasoning_texts.extend(s.text for s in (item.summary or []) if s.text)
        elif item.type == "function_call":
            calls.append(item)
    return " ".join(reasoning_texts).strip(), calls


async def run_agent(
    transcript: str,
    *,
    client: AsyncOpenAI,
    pipeline: RetrievalPipeline | None,
    settings: Settings,
    model: str | None = None,
    stub: bool = False,
) -> AgentResult:
    """Ejecuta el bucle agéntico sobre una transcripción y devuelve estimación + traza."""
    model = model or settings.agent_model
    tools = build_tools(settings)
    registry = build_tool_registry(pipeline=pipeline, settings=settings, stub=stub)
    system_prompt = build_system_prompt(settings)
    reasoning_cfg = {
        "effort": settings.agent_reasoning_effort,
        "summary": settings.agent_reasoning_summary,
    }
    trace: list[AgentStep] = []

    # Primera llamada: la transcripción como input. reasoning + primeras function_call.
    response = await client.responses.parse(
        model=model,
        reasoning=reasoning_cfg,
        instructions=system_prompt,
        input=[{"role": "user", "content": transcript}],
        tools=tools,
        text_format=AgentEstimate,
    )

    for _ in range(settings.agent_max_steps):
        turn_reasoning, calls = _partition_output(response)
        if not calls:
            # Sin más tools: el modelo ha producido la respuesta final estructurada.
            logger.info("agent.final", reasoning=turn_reasoning, steps=len(trace))
            return AgentResult(
                status="done",
                estimate=response.output_parsed,
                trace=trace,
                steps=len(trace),
            )

        # Ejecuta TODAS las llamadas del turno en paralelo (una vuelta puede traer varias).
        parsed = [(c, json.loads(c.arguments)) for c in calls]
        results = await asyncio.gather(
            *(execute_tool(registry, c.name, a) for c, a in parsed)
        )

        tool_outputs: list[dict] = []
        for (call, args), result in zip(parsed, results, strict=True):
            trace.append(
                AgentStep(
                    step=len(trace) + 1,
                    reasoning=turn_reasoning,
                    action=call.name,
                    args=args,
                    observation=result,
                )
            )
            logger.info(
                "agent.step",
                step=len(trace),
                action=call.name,
                observation=list(result.keys()),  # correlado por request_id vía contextvars
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        # Continúa el bucle: mismo call_id por resultado, encadenado con previous_response_id.
        # previous_response_id NO arrastra el system prompt → reenviamos instructions y tools.
        response = await client.responses.parse(
            model=model,
            previous_response_id=response.id,
            reasoning=reasoning_cfg,
            instructions=system_prompt,
            input=tool_outputs,
            tools=tools,
            text_format=AgentEstimate,
        )
    else:
        logger.warning("agent.max_steps_exceeded", steps=len(trace))
        return AgentResult(
            status="max_steps_exceeded", estimate=None, trace=trace, steps=len(trace)
        )
