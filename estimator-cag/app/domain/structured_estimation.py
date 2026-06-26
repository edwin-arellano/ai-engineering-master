"""Schemas del flujo invertido (S10): (1) esqueleto de módulos/tareas SIN horas
(generado por CAG), (2) estimación por-tarea con horas DERIVADAS de vecinos históricos
(consenso determinista, nunca inferencia del modelo) + fiabilidad."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
