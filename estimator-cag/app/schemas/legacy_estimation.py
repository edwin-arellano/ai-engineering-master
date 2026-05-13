"""Schemas legacy de session-02/03 que siguen vivos en pre-session-04.

Estos schemas y enums los sigue consumiendo:
- El endpoint /api/v1/estimate/stream (que no se ha migrado al nuevo schema).
- evaluation_service.py (en standby, pero debe seguir compilando).

A partir de session-04, todo este archivo es candidato a eliminación si se
decide migrar el stream al nuevo schema o sustituirlo por algo distinto.

NOTA: los nombres de campos de StructureCheck y la firma de TokenUsage se
preservan idénticos a session-03 para que evaluation_service.py siga
compilando sin tocar su cuerpo.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LegacyExampleFormat(str, Enum):
    """Formato de los ejemplos few-shot inyectados en el system prompt legacy."""

    MARKDOWN = "markdown"


class LegacyOutputFormat(str, Enum):
    """Formato del output esperado en el endpoint legacy /estimate/stream."""

    MARKDOWN = "markdown"
    JSON = "json"


class LegacyPreprocessingType(str, Enum):
    """Estrategia de preprocesado de la transcripción en el flujo legacy."""

    NONE = "none"
    INLINE_CLEANING = "inline_cleaning"
    TWO_PHASE = "two_phase"


class StreamEstimationRequest(BaseModel):
    """Request del endpoint /api/v1/estimate/stream (sin cambios desde session-03).

    Deliberadamente minimal: sin preprocessing, sin evaluation, sin
    thinking_budget. La selección de ejemplos es determinista para que
    la cache exact-match funcione.
    """

    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión.",
    )
    num_examples: int = Field(
        default=3,
        ge=0,
        le=5,
        description=(
            "Número de ejemplos CAG a inyectar. La selección es DETERMINISTA "
            "(primeros N en orden) para que la cache exact-match haga hits."
        ),
    )


class TokenUsage(BaseModel):
    """Desglose de uso de tokens del flujo legacy."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    preprocessing_input_tokens: int = Field(default=0, ge=0)
    preprocessing_output_tokens: int = Field(default=0, ge=0)


class StructureCheck(BaseModel):
    """Resultado de la evaluación estructural automática (nivel 1, determinista).

    Preservado tal cual desde session-03: evaluation_service.py instancia este
    modelo con los nombres exactos de los campos, así que no podemos simplificar
    el schema sin romper el evaluator.
    """

    has_title: bool
    has_breakdown_table: bool
    has_total_sections: bool
    has_team_sections: bool
    has_duration_sections: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool
    finish_reason_ok: bool
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Promedio simple de los 7 booleanos del check.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Lista de problemas detectados, en lenguaje natural.",
    )
