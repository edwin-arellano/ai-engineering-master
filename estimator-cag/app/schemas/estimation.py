"""Schemas Pydantic v2 del dominio de estimación.

Contiene:
- Enums (ProjectType, DetailLevel, OutputFormat) que sobreviven al refactor de
  pre-S05 porque siguen describiendo el input del formulario.
- Phase y EstimationResult: salida estructurada del LLM, con los dos
  ``model_validator`` que actúan como guardrail semántico ligero.
- EstimationResponse: envuelve el resultado más metadata de prompt/cache.

Pre-S05: ``EstimationRequest`` se elimina porque el endpoint single-shot
desaparece. Los enums se siguen usando como campos ``Form()`` en el endpoint
multipart de sesiones. El shim para el código del cache (que aún tipa con un
request-like) vive en ``app/schemas/estimation_compat.py``.

Cuando Instructor recibe un fallo de los validators de ``EstimationResult``,
reintenta automáticamente mostrando el error al modelo (política
"fix con retry").
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.schemas.actor_critic_boss import BossIteration


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
    """Respuesta del endpoint de sesiones.

    En modo actor, los campos `acb_*` son None. En modo actor_critic_boss,
    informan de la convergencia y las iteraciones para que el cliente las
    muestre.

    ``cached`` y ``cache_level`` quedan inertes en S05 (siempre ``False`` y
    ``None``) porque el flujo conversacional no invoca el cache. Se conservan
    en el shape para no romper consumidores ni cerrar la puerta a reactivar el
    cache en sesiones futuras.
    """

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    cache_level: str | None = Field(
        default=None,
        description="Origen de la cache: None | 'exact_match' | 'semantic'.",
    )
    tier: str | None = None
    estimation_mode: str | None = None
    acb_converged: bool | None = None
    acb_total_iterations: int | None = None
    acb_iterations: list["BossIteration"] | None = None
