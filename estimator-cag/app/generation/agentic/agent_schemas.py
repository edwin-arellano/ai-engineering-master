"""Contrato de salida del agente (S12) y estructuras de traza.

`AgentEstimate` es el `text_format` de las llamadas a la Responses API: la forma de la
salida es SIEMPRE la misma aunque el camino del bucle varíe (no-determinación dentro del
bucle, contrato de salida determinista). Es un contrato PROPIO del agente, distinto de
`RagEstimate` (S09) y `StructuredEstimate` (S10): otro flujo, otro endpoint.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# AgentStep se movió a domain/ (S12): lo comparten el one-shot y las dos fases del flujo
# híbrido, y `domain` no puede depender de `generation`. Se reexporta para que los
# importadores de este módulo sigan funcionando sin cambios.
from app.domain.agent_trace import AgentStep


class ComponentEstimate(BaseModel):
    """Un componente estimado con las horas y los presupuestos históricos que lo respaldan."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    estimated_hours: float = Field(..., ge=0)
    reference_budget_ids: list[str]  # vacío ⇒ unbudgeted
    unbudgeted: bool


class AgentEstimate(BaseModel):
    """Salida estructurada final del agente. Sin números inventados: un componente sin
    referencia histórica va a 0h con unbudgeted=True."""

    model_config = ConfigDict(extra="forbid")

    components: list[ComponentEstimate]
    total_hours: float = Field(..., ge=0)
    notes: str


class AgentResult(BaseModel):
    """Resultado del bucle. `status` separa éxito de agotamiento; el backend de negocio
    enruta por él (done → guardar; max_steps_exceeded → revisión manual)."""

    status: Literal["done", "max_steps_exceeded"]
    estimate: AgentEstimate | None
    trace: list[AgentStep]
    steps: int


__all__ = ["AgentEstimate", "AgentResult", "AgentStep", "ComponentEstimate"]
