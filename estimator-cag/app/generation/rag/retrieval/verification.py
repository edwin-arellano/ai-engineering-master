"""Fase 5 del RAG: verificación automática del output. (a) Las citations deben
referirse a chunks realmente recuperados (no inventados). (b) Coherencia de
confianza: si insufficient, no debe haber totales (los validators de RagEstimate
ya lo garantizan; aquí lo reforzamos a nivel de servicio y reportamos)."""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from app.domain.rag_estimation import Confidence, RagEstimate
from app.generation.rag.schemas import AugmentedContext

logger = structlog.get_logger(__name__)


class CitationVerificationError(ValueError):
    """Una tarea cita un source_id que no está entre los chunks recuperados."""


class CitationReport(BaseModel):
    """Informe estructural de citaciones (no semántico: confirma que la fuente
    estuvo en el contexto, no que diga lo que se afirma — eso es el directo S11)."""

    total_lines: int
    grounded_lines: int  # tareas con evidencia y TODAS sus citas resueltas
    insufficient_lines: int  # tareas marcadas is_assumption (sin datos suficientes)
    dangling: list[str]  # source_id citados que NO estuvieron en el contexto


def verify_citations(estimate: RagEstimate, context: AugmentedContext) -> CitationReport:
    """Verificación estructural por línea: clasifica cada tarea en grounded
    (evidencia con todas sus citas resueltas), insufficient (asunción) o con citas
    colgantes (source_id que nunca estuvo en included_refs). Las colgantes se reportan
    y loguean (request_id se inyecta vía structlog.contextvars en el middleware)."""
    allowed = set(context.included_refs)
    dangling: list[str] = []
    grounded = insufficient = total = 0
    for module in estimate.modules:
        for task in module.tasks:
            total += 1
            if task.is_assumption:
                insufficient += 1
                continue
            line_dangling = [c.source_id for c in task.sources if c.source_id not in allowed]
            dangling.extend(line_dangling)
            if not line_dangling:
                grounded += 1
    if dangling:
        logger.warning("rag.dangling_citations", count=len(dangling), refs=dangling[:10])
    return CitationReport(
        total_lines=total,
        grounded_lines=grounded,
        insufficient_lines=insufficient,
        dangling=dangling,
    )


def enforce_confidence_coherence(estimate: RagEstimate) -> None:
    """Refuerzo a nivel servicio: insufficient ⇒ sin totales. (Redundante con el
    validator, pero deja el invariante explícito en el pipeline.)"""
    if estimate.confidence == Confidence.INSUFFICIENT:
        assert estimate.total_engineer_days is None and not estimate.modules
