"""Schemas del flujo invertido (S10): (1) esqueleto de módulos/tareas SIN horas
(generado por CAG), (2) estimación por-tarea con horas DERIVADAS de vecinos históricos
(consenso determinista, nunca inferencia del modelo) + fiabilidad.

S12 los ENVUELVE sin romperlos: los campos agénticos (`flag_reason`, `agent_trace`) son
opcionales y su default `None` deja la salida del flujo determinista byte a byte idéntica."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.agent_trace import AgentTrace


class SkeletonTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)


class SkeletonModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    tasks: list[SkeletonTask] = Field(default_factory=list)


class EstimateSkeleton(BaseModel):
    """Esqueleto sin horas (salida de la fase CAG, revisable por el humano)."""

    model_config = ConfigDict(extra="forbid")

    modules: list[SkeletonModule] = Field(default_factory=list)


class StructureProposal(BaseModel):
    """Salida de la fase 1 agéntica (S12): el esqueleto existente + la traza que lo produjo.

    Envoltorio, no sustituto: `EstimateSkeleton` sigue siendo el contrato que viaja a la
    fase 2, y es lo que el humano revisa en la puerta entre ambas."""

    model_config = ConfigDict(extra="forbid")

    skeleton: EstimateSkeleton
    agent_trace: AgentTrace | None = None


class Reliability(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class TaskNeighbor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_id: str
    chunk_ref: str
    estimated_hours: float
    distance: float


class TaskEstimate(BaseModel):
    """Horas por-tarea derivadas de vecinos. needs_human_input=True cuando no hay
    evidencia histórica suficiente (el humano fija las horas en la revisión)."""

    model_config = ConfigDict(extra="forbid")

    title: str
    suggested_hours: float | None = Field(default=None, ge=0)
    reliability: Reliability = Reliability.NONE
    neighbors: list[TaskNeighbor] = Field(default_factory=list)
    needs_human_input: bool = True
    # S12: por qué el retrieval determinista no supo cerrar esta tarea. None ⇒ no flaggeada
    # (el determinista la resolvió bien y el agente de recuperación ni la mira). Lo puebla
    # `agentic.flagging`, sin LLM.
    flag_reason: str | None = None


class EstimatedModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    tasks: list[TaskEstimate] = Field(default_factory=list)


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    with_history: int
    without_history: int
    total: int


class StructuredEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modules: list[EstimatedModule] = Field(default_factory=list)
    coverage: Coverage
    total_suggested_hours: float = Field(ge=0)  # suma de suggested_hours conocidas
    # S12: traza de la fase de recuperación. None ⇒ la produjo el flujo determinista puro.
    agent_trace: AgentTrace | None = None
