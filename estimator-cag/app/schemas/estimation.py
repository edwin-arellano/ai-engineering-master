"""Schemas Pydantic v2 del dominio de estimación.

Contiene:
- Enums de la sesión pre-04 (ProjectType, DetailLevel, OutputFormat).
- EstimationRequest (input del usuario).
- Phase y EstimationResult: nuevos en S04, representan la salida tipada del LLM.
- EstimationResponse: envuelve el resultado más metadata de cache y versión de prompt.

Los `model_validator` de `EstimationResult` actúan como guardrail semántico ligero:
- `total_must_match_sum_of_phases`: rechaza estimaciones incoherentes.
- `low_confidence_must_be_explicit`: obliga al modelo a marcar las respuestas
  de baja confianza como out-of-scope explícitamente.

Cuando Instructor recibe un fallo de estos validators, reintenta automáticamente
mostrando el error al modelo (política de fallo "fix con retry").
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums del input
# ---------------------------------------------------------------------------


class ProjectType(StrEnum):
    """Tipo de proyecto a estimar."""

    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    INTEGRATION = "integration"
    OTHER = "other"


class DetailLevel(StrEnum):
    """Nivel de detalle pedido al modelo."""

    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(StrEnum):
    """Formato narrativo del summary; las fases siempre vienen estructuradas."""

    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class EstimationRequest(BaseModel):
    """Petición de estimación tal como llega del formulario."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Descripción libre del proyecto a estimar.",
    )
    project_type: ProjectType = Field(default=ProjectType.OTHER)
    detail_level: DetailLevel = Field(default=DetailLevel.MEDIUM)
    output_format: OutputFormat = Field(default=OutputFormat.PHASES_TABLE)


# ---------------------------------------------------------------------------
# Output estructurado (nuevo en S04)
# ---------------------------------------------------------------------------


class Phase(BaseModel):
    """Una fase del plan de trabajo dentro de una estimación."""

    name: str = Field(..., min_length=1, max_length=120)
    duration_weeks: int = Field(
        ...,
        ge=0,
        le=104,
        description="Duración en semanas. 0 solo cuando la fase no se puede dimensionar.",
    )
    cost_eur: int = Field(..., ge=0, description="Coste estimado en euros, sin decimales.")
    confidence_pct: int = Field(..., ge=0, le=100)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Asunciones que sostienen la estimación de la fase.",
    )


class EstimationResult(BaseModel):
    """Salida estructurada del modelo después de Instructor."""

    summary: str = Field(..., min_length=1, max_length=4000)
    total_duration_weeks: int = Field(..., ge=0, le=520)
    total_cost_eur: int = Field(..., ge=0)
    confidence_pct: int = Field(..., ge=0, le=100)
    phases: list[Phase] = Field(default_factory=list)

    # ----- Validators custom (capa 4 de guardrails) -----

    @model_validator(mode="after")
    def total_must_match_sum_of_phases(self) -> "EstimationResult":
        """La suma de las fases debe cuadrar con los totales (±tolerancias).

        - Duración: tolerancia de ±1 semana respecto al total.
        - Coste: tolerancia de ±5% respecto al total.

        Si `phases` está vacío (caso out-of-scope), no se valida la coherencia.
        """
        if not self.phases:
            return self

        sum_weeks = sum(p.duration_weeks for p in self.phases)
        sum_cost = sum(p.cost_eur for p in self.phases)

        if abs(sum_weeks - self.total_duration_weeks) > 1:
            raise ValueError(
                "total_duration_weeks ({total}) no cuadra con la suma de fases "
                "({sum_weeks}) — tolerancia permitida ±1 semana".format(
                    total=self.total_duration_weeks, sum_weeks=sum_weeks
                )
            )

        if self.total_cost_eur > 0:
            relative_diff = abs(sum_cost - self.total_cost_eur) / self.total_cost_eur
            if relative_diff > 0.05:
                raise ValueError(
                    "total_cost_eur ({total}) no cuadra con la suma de fases "
                    "({sum_cost}) — tolerancia permitida ±5%".format(
                        total=self.total_cost_eur, sum_cost=sum_cost
                    )
                )

        return self

    @model_validator(mode="after")
    def low_confidence_must_be_explicit(self) -> "EstimationResult":
        """Si la confianza es inferior al umbral, el summary debe marcarlo explícitamente.

        Forzamos al modelo a etiquetar las respuestas dudosas como out-of-scope
        en lugar de devolver una estimación "tibia" sin contexto.
        """
        if self.confidence_pct < 30 and not self.summary.startswith("Out of scope:"):
            raise ValueError(
                "confidence_pct < 30 requiere que summary empiece con 'Out of scope:'"
            )
        return self


# ---------------------------------------------------------------------------
# Envoltorio de respuesta de la API
# ---------------------------------------------------------------------------


class EstimationResponse(BaseModel):
    """Lo que el endpoint `POST /api/v1/estimate` devuelve al cliente.

    `cached` y `cache_level` permiten al frontend mostrar si la respuesta vino
    de cache y de qué capa (exact-match o semántico), útil para observabilidad
    y para que el usuario interprete la latencia que está viendo.
    """

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    cache_level: str | None = Field(
        default=None,
        description="Origen de la cache: None | 'exact_match' | 'semantic'.",
    )
