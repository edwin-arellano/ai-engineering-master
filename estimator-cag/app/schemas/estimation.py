"""Contratos del endpoint de estimaciones."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# === Enums ===

class PreprocessingType(str, Enum):
    """Estrategia de preprocesado de la transcripción antes de generar la estimación."""
    NONE = "none"
    INLINE_CLEANING = "inline_cleaning"
    TWO_PHASE = "two_phase"


class ExampleFormat(str, Enum):
    """Formato en el que se inyectan los ejemplos en el system prompt."""
    MARKDOWN = "markdown"


class OutputFormat(str, Enum):
    """Formato esperado de la respuesta del LLM."""
    MARKDOWN = "markdown"
    JSON = "json"


# === Request ===

class EstimationRequest(BaseModel):
    """Petición de estimación con todas las opciones de generación.

    Sólo `transcription` es obligatorio. El resto son overrides opcionales
    de los defaults configurados en Settings.
    """

    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión con el cliente.",
    )

    # Contexto CAG
    num_examples: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Número de ejemplos a inyectar (0 = sin CAG, máximo = total disponibles).",
    )
    example_format: ExampleFormat = Field(
        default=ExampleFormat.MARKDOWN,
        description="Formato en que se serializan los ejemplos en el prompt.",
    )

    # Preprocesado
    preprocessing: PreprocessingType = Field(
        default=PreprocessingType.NONE,
        description="Estrategia de preprocesado de la transcripción.",
    )

    # Configuración del LLM (overrides opcionales de los defaults en Settings)
    model: str | None = Field(
        default=None,
        description="Override del modelo. Si es None, se usa el configurado en Settings.",
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        le=16000,
        description="Máximo de tokens de salida.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperatura de muestreo. Ignorada cuando thinking_budget > 0 en Anthropic.",
    )
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=8000,
        description=(
            "Budget de tokens de razonamiento. Sólo aplica a modelos Anthropic "
            "compatibles con extended thinking (Claude 4.x). Ignorado en otros casos."
        ),
    )

    # Formato de salida
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN,
        description="Formato que se le exige al LLM en la respuesta.",
    )

    # Respuesta enriquecida (opt-in)
    usage: bool = Field(
        default=True,
        description="Incluir el desglose de tokens en la respuesta.",
    )
    evaluation: bool = Field(
        default=True,
        description="Ejecutar la evaluación estructural nivel 1.",
    )


# === Response ===

class TokenUsage(BaseModel):
    """Desglose de uso de tokens, incluyendo la fase de preprocesado si aplica."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    preprocessing_input_tokens: int = Field(default=0, ge=0)
    preprocessing_output_tokens: int = Field(default=0, ge=0)


class StructureCheck(BaseModel):
    """Resultado de la evaluación estructural automática (nivel 1, determinista)."""

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
        description="Promedio simple de los 7 booleanos del check (has_title, has_breakdown_table, has_total_sections, has_team_sections, has_duration_sections, hours_match, finish_reason_ok).",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Lista de problemas detectados, en lenguaje natural.",
    )


class EstimationResponse(BaseModel):
    """Respuesta completa del endpoint de estimación."""

    estimation: str = Field(
        ...,
        description="Estimación generada por el LLM. El formato depende de output_format.",
    )
    model: str
    provider: str
    finish_reason: str = Field(
        ...,
        description="Razón de finalización tal como la reporta el SDK (e.g. 'stop', 'end_turn', 'max_tokens').",
    )
    preprocessing_type: PreprocessingType
    output_format: OutputFormat
    latency_ms: int = Field(
        ge=0,
        description="Latencia total de la petición en milisegundos (incluye preprocesado y evaluación).",
    )
    token_usage: TokenUsage | None = None
    extracted_requirements: str | None = Field(
        default=None,
        description="Salida de la fase de extracción cuando preprocessing=two_phase.",
    )
    evaluation: StructureCheck | None = None
