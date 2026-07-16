"""Traza agéntica compartida por las dos fases del flujo híbrido (S12).

Una traza NO es un log: es la secuencia de decisiones del agente (razonamiento + acción +
observación) que permite auditar cómo llegó a un número y, sobre todo, mejorarlo. Se
ADJUNTA a los modelos de dominio existentes (`StructureProposal` envuelve `EstimateSkeleton`;
`StructuredEstimate` gana un campo opcional), nunca los reemplaza: el agente se adapta a los
contratos de la app, no al revés.

`AgentStep` vive aquí (y no en `generation/agentic/agent_schemas.py`, de donde se movió)
porque lo consumen tanto el paquete agéntico como el dominio estructurado: `domain` es la
capa base y no puede depender de `generation`. `agent_schemas` lo reexporta, así que los
importadores del one-shot (S12 pre-directo) siguen funcionando sin cambios.

En producción la traza iría a un bucket por volumen (S3); aquí se devuelve en la respuesta y
se loguea correlada por `request_id` (structlog).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    """Una vuelta ejecutada del bucle: razonamiento (reasoning summary del turno) +
    acción (tool) + observación (resultado). Es la ventana de depuración del agente."""

    step: int
    reasoning: str  # reasoning summary del turno (puede ir vacío si el modelo no lo emite)
    action: str  # nombre de la tool
    args: dict[str, Any]
    observation: dict[str, Any]


class AgentTrace(BaseModel):
    """Secuencia ordenada de pasos ejecutados por un agente en UNA fase del flujo.

    `agent` es el perfil que la produjo (S12: siempre "neo"); `phase` distingue las dos
    entradas del agente en el flujo, que producen trazas de tamaño muy distinto:
    "structure" → 1 step (sin tools), "recovery" → un bucle por tarea flaggeada.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    phase: str  # "structure" | "recovery"
    steps: list[AgentStep] = Field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)
