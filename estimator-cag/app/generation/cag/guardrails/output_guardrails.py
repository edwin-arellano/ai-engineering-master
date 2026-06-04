"""Capa 5 de guardrails: filtro de salida.

A diferencia de las capas anteriores, este módulo NO lanza excepciones cuando
dispara. La política de fallo es **filter**: el resultado se devuelve al
cliente tal cual (para que la UI pueda mostrar un mensaje útil), pero la
función `should_cache_result` evita persistirlo en cache.

Razón: cachear respuestas de baja confianza o explícitamente out-of-scope
propagaría una respuesta dudosa a futuras requests semánticamente similares.
El cache propaga aciertos tan rápido como propaga errores; mejor pagar la
llamada al LLM otra vez que servir basura cacheada.
"""

from __future__ import annotations

import structlog

from app.foundations.config import Settings
from app.domain.estimation import EstimationResult

logger = structlog.get_logger(__name__)


def is_out_of_scope(result: EstimationResult) -> bool:
    """True si el summary del modelo marca el resultado como out-of-scope."""
    return result.summary.startswith("Out of scope:")


def is_low_confidence(result: EstimationResult, settings: Settings) -> bool:
    """True si la confianza está por debajo del umbral configurado."""
    return result.confidence_pct < settings.min_confidence_pct


def should_cache_result(result: EstimationResult, settings: Settings) -> bool:
    """Decide si el resultado se persiste en cache.

    - Out-of-scope → no cachear.
    - Confianza inferior al umbral → no cachear.
    - Cualquier otro caso → cachear.

    Loguea la razón del rechazo con `output_guardrail_skip_cache` para que la
    decisión sea auditable en producción.
    """
    if is_out_of_scope(result):
        logger.info("output_guardrail_skip_cache", reason="out_of_scope")
        return False
    if is_low_confidence(result, settings):
        logger.info(
            "output_guardrail_skip_cache",
            reason="low_confidence",
            confidence_pct=result.confidence_pct,
            min_confidence_pct=settings.min_confidence_pct,
        )
        return False
    return True
