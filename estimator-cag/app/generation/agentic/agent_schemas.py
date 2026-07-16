"""Contrato de salida del agente (S12) y estructuras de traza.

`AgentEstimate` es el `text_format` de las llamadas a la Responses API: la forma de la
salida es SIEMPRE la misma aunque el camino del bucle varíe (no-determinación dentro del
bucle, contrato de salida determinista). Es un contrato PROPIO del agente, distinto de
`RagEstimate` (S09) y `StructuredEstimate` (S10): otro flujo, otro endpoint.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class AgentStep(BaseModel):
    """Una vuelta ejecutada del bucle: razonamiento (reasoning summary del turno) +
    acción (tool) + observación (resultado). Es la ventana de depuración del agente."""

    step: int
    reasoning: str  # reasoning summary del turno (puede ir vacío si el modelo no lo emite)
    action: str  # nombre de la tool
    args: dict[str, Any]
    observation: dict[str, Any]


class AgentResult(BaseModel):
    """Resultado del bucle. `status` separa éxito de agotamiento; el backend de negocio
    enruta por él (done → guardar; max_steps_exceeded → revisión manual)."""

    status: Literal["done", "max_steps_exceeded"]
    estimate: AgentEstimate | None
    trace: list[AgentStep]
    steps: int
