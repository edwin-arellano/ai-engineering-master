"""Síntesis: cuando las fuentes citadas de una línea discrepan sin contradecirse,
devuelve un rango honesto (min/max) en vez de un número ciego. Detección de contradicción
por dispersión (coeficiente de variación); si es demasiado alta, se DESCARTA (datos que no
encajan). El número es determinista; un mini LLM barato SOLO explica el rango al humano."""

from __future__ import annotations

from statistics import mean, pstdev

import structlog
from pydantic import BaseModel

from app.domain.rag_estimation import HourRange
from app.foundations.config import Settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_synthesis_reason_prompt

logger = structlog.get_logger(__name__)


class _Reason(BaseModel):
    reason: str


def coefficient_of_variation(values: list[float]) -> float:
    """Dispersión relativa (σ/μ) de un conjunto de horas. Determinista, sin LLM.

    Público desde S12: el flagging del flujo híbrido mide con la MISMA vara la discrepancia
    entre vecinos históricos, aunque decida otra cosa con ella (marcar la tarea para el
    agente en vez de descartar la síntesis).
    """
    m = mean(values)
    return pstdev(values) / m if m > 0 else 0.0


def synthesize_range(
    hours: list[float],
    *,
    wrapper: LLMWrapper,
    settings: Settings,
    context: str = "",
) -> HourRange | None:
    """Devuelve un HourRange o None (una sola fuente, sin datos, o CONTRADICCIÓN)."""
    values = [h for h in hours if h is not None and h >= 0]
    if len(values) < 2:
        return None
    cv = coefficient_of_variation(values)
    if cv > settings.contradiction_cv_threshold:
        logger.info("synthesis.contradiction_discarded", cv=round(cv, 3), n=len(values))
        return None  # contradicción: no sintetizar
    lo, hi = min(values), max(values)
    reason = f"{len(values)} fuentes análogas entre {lo:g} y {hi:g} horas."
    if settings.synthesis_reason_enabled:
        try:
            reason = wrapper.complete_structured(  # mini LLM SOLO para la explicación
                system_prompt=render_synthesis_reason_prompt(
                    settings.synthesis_reason_prompt_version
                ),
                user_message=f"valores={values} contexto='{context}'",
                response_model=_Reason,
                alias=REFORMULATOR_ALIAS,
                temperature=0.2,
                max_tokens=200,
            ).reason
        except Exception:  # noqa: BLE001 — fallback a la razón determinista
            pass
    return HourRange(min=lo, max=hi, reason=reason, dispersion=round(cv, 3))
