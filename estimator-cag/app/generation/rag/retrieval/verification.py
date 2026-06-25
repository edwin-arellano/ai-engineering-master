"""Fase 5 del RAG: verificación automática del output. (a) Las citations deben
referirse a chunks realmente recuperados (no inventados). (b) Coherencia de
confianza: si insufficient, no debe haber totales (los validators de RagEstimate
ya lo garantizan; aquí lo reforzamos a nivel de servicio y reportamos)."""

from __future__ import annotations

import structlog

from app.domain.rag_estimation import Confidence, RagEstimate
from app.generation.rag.schemas import AugmentedContext

logger = structlog.get_logger(__name__)


class CitationVerificationError(ValueError):
    """Una tarea cita un source_id que no está entre los chunks recuperados."""


def verify_citations(estimate: RagEstimate, context: AugmentedContext) -> list[str]:
    """Devuelve la lista de source_id citados que NO están en included_refs.
    Vacía = todas las citations son válidas."""
    allowed = set(context.included_refs)
    invalid: list[str] = []
    for module in estimate.modules:
        for task in module.tasks:
            for citation in task.sources:
                if citation.source_id not in allowed:
                    invalid.append(citation.source_id)
    if invalid:
        logger.warning("rag.invalid_citations", count=len(invalid), refs=invalid[:10])
    return invalid


def enforce_confidence_coherence(estimate: RagEstimate) -> None:
    """Refuerzo a nivel servicio: insufficient ⇒ sin totales. (Redundante con el
    validator, pero deja el invariante explícito en el pipeline.)"""
    if estimate.confidence == Confidence.INSUFFICIENT:
        assert estimate.total_engineer_days is None and not estimate.modules
