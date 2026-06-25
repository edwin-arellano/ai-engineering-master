"""Schema de salida de la estimación RAG-grounded (S09). Separado de
`EstimationResult` (CAG) a propósito: distinto contrato, distinto endpoint.

Reglas de negocio que el modelo debe cumplir (se hacen cumplir con model_validator,
política 'fix con retry' de Instructor):
- Cada tarea con evidencia cita >=1 source_id; las asunciones van con sources vacío.
- total_engineer_days == suma de engineer_days de todas las tareas.
- Si confidence == insufficient: modules vacío y totales a None (no inventar números).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class Citation(BaseModel):
    """Referencia a un chunk del que deriva una tarea. `source_id` es el chunk_ref
    (BUD-...::AUTH-...), verificable contra los chunks recuperados."""

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(..., min_length=1)


class RagTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)
    engineer_days: float = Field(..., ge=0)
    # Tareas basadas en evidencia: >=1 source. Asunciones: lista vacía + is_assumption=True.
    sources: list[Citation] = Field(default_factory=list)
    is_assumption: bool = False

    @model_validator(mode="after")
    def evidence_or_assumption(self) -> "RagTask":
        if not self.is_assumption and not self.sources:
            raise ValueError(
                "una tarea basada en evidencia debe citar al menos un source_id; "
                "si no hay fuente, márcala con is_assumption=True"
            )
        return self


class RagModule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    tasks: list[RagTask] = Field(default_factory=list)


class RagEstimate(BaseModel):
    """Estimación generada a partir del contexto RAG. `reasoning` explica la
    derivación (de dónde sale cada bloque); útil para el revisor humano, no para UI."""

    model_config = ConfigDict(extra="forbid")

    confidence: Confidence
    reasoning: str = Field(..., min_length=1, max_length=4000)
    modules: list[RagModule] = Field(default_factory=list)
    total_engineer_days: float | None = Field(default=None, ge=0)
    total_duration_weeks: int | None = Field(default=None, ge=0, le=520)

    @model_validator(mode="after")
    def insufficient_means_empty(self) -> "RagEstimate":
        """confidence=insufficient ⇒ no se devuelven módulos ni totales (serían inventados)."""
        if self.confidence == Confidence.INSUFFICIENT:
            if self.modules or self.total_engineer_days is not None or self.total_duration_weeks is not None:
                raise ValueError(
                    "confidence=insufficient requiere modules=[] y totales=None "
                    "(no inventar duración/totales sin contexto suficiente)"
                )
        return self

    @model_validator(mode="after")
    def totals_match_sum_of_tasks(self) -> "RagEstimate":
        """total_engineer_days == suma de engineer_days de todas las tareas (±0.5 días)."""
        if self.confidence == Confidence.INSUFFICIENT or not self.modules:
            return self
        task_sum = sum(t.engineer_days for m in self.modules for t in m.tasks)
        if self.total_engineer_days is None:
            raise ValueError("con módulos presentes, total_engineer_days no puede ser None")
        if abs(task_sum - self.total_engineer_days) > 0.5:
            raise ValueError(
                f"total_engineer_days ({self.total_engineer_days}) no cuadra con la suma "
                f"de tareas ({task_sum}) — tolerancia ±0.5 días"
            )
        return self
