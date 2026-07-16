"""Mecánica reutilizable del bucle agéntico sobre la Responses API (extraída de `agent.py`).

reason→act→observe: llamada al modelo → recorrer `response.output` por `type` → ejecutar las
tools del turno en paralelo → devolver `function_call_output` con el MISMO `call_id` →
encadenar con `previous_response_id` → repetir mientras haya `function_call` → salir al
obtener la salida final estructurada o al agotar `max_steps`. Cada vuelta produce los
`AgentStep` de la traza.

Este módulo es SOLO la mecánica: no sabe qué tools existen ni qué se está estimando. Quién lo
usa decide instrucciones, tools, formato de salida y presupuesto de pasos. `agent.py` (el
one-shot, baseline de comparación) conserva su propia copia del bucle intacta a propósito.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.domain.agent_trace import AgentStep
from app.generation.agentic.agent import partition_output
from app.generation.agentic.agent_tools import ToolFn, execute_tool

logger = structlog.get_logger(__name__)


async def run_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    instructions: str,
    initial_input: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    registry: dict[str, ToolFn],
    reasoning_cfg: dict[str, str],
    text_format: type,
    max_steps: int,
    phase: str,
) -> tuple[Any | None, list[AgentStep], str]:
    """Ejecuta el bucle hasta la salida final o el agotamiento de pasos.

    Devuelve `(output_parsed | None, steps, status)` con `status` en {"done",
    "max_steps_exceeded"}. El control de flujo es NUESTRO, no del modelo: quién decide
    cuántas vueltas caben y qué pasa si no converge es este bucle. Un fallo de tool no lo
    revienta: `execute_tool` lo convierte en observación de error y el modelo puede
    reaccionar.
    """
    response = await client.responses.parse(
        model=model,
        reasoning=reasoning_cfg,
        instructions=instructions,
        input=initial_input,
        tools=tools,
        text_format=text_format,
    )

    steps: list[AgentStep] = []
    for _ in range(max_steps):
        turn_reasoning, calls = partition_output(response)
        if not calls:
            # Sin más tools: el modelo ha producido la respuesta final estructurada.
            logger.info("agent.final", phase=phase, steps=len(steps))
            return response.output_parsed, steps, "done"

        # Una vuelta puede traer varias llamadas: se ejecutan en paralelo.
        parsed = [(c, json.loads(c.arguments)) for c in calls]
        results = await asyncio.gather(
            *(execute_tool(registry, c.name, a) for c, a in parsed)
        )

        tool_outputs: list[dict[str, Any]] = []
        for (call, args), result in zip(parsed, results, strict=True):
            steps.append(
                AgentStep(
                    step=len(steps) + 1,
                    reasoning=turn_reasoning,
                    action=call.name,
                    args=args,
                    observation=result,
                )
            )
            logger.info(
                "agent.step",
                phase=phase,
                step=len(steps),
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

        # previous_response_id NO arrastra el system prompt → reenviamos instructions y tools.
        response = await client.responses.parse(
            model=model,
            previous_response_id=response.id,
            reasoning=reasoning_cfg,
            instructions=instructions,
            input=tool_outputs,
            tools=tools,
            text_format=text_format,
        )

    logger.warning("agent.max_steps_exceeded", phase=phase, steps=len(steps))
    return None, steps, "max_steps_exceeded"
