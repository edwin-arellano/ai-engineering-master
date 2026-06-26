"""Ponderación blanda: reordena SOLO los finalistas como último ajuste (no filtra,
no expulsa candidatos). Decaimiento temporal por AÑO (nuestros chunks llevan `year`,
no fecha completa): peso = 0.5 ** (edad_años / semivida_años). Boosts contextuales
opcionales y conservadores (tech/sector) cuando la consulta los menciona. Multiplica
el score de relevancia previo.

Es la ÚLTIMA etapa del pipeline: 'lo barato y excluyente al principio; lo caro y fino
al final; lo blando al cierre'. Un multiplicador puede invertir el orden del
cross-encoder a propósito (un histórico reciente equivalente debe ganar a uno viejo).

La base de relevancia es POSITIVA y monótona: base = 1/(1+distance) en (0,1] (menor
distancia → mayor relevancia). Así un peso <1 SIEMPRE degrada y uno >1 mejora, sin la
ambigüedad de signo de usar -distance directamente.
"""

from __future__ import annotations

from datetime import date

import structlog

from app.foundations.config import Settings
from app.generation.rag.schemas import ReformulatedQuery, RetrievedChunk

logger = structlog.get_logger(__name__)


def temporal_weight(
    year: int | None, *, half_life_years: float, today_year: int | None = None
) -> float:
    """Peso temporal por año: 1.0 para el año actual, 0.5 a una semivida de distancia.
    `year=None` → 1.0 (sin penalización: no sabemos la edad)."""
    if year is None:
        return 1.0
    today_year = today_year or date.today().year
    age = max(today_year - int(year), 0)
    return 0.5 ** (age / half_life_years)


def apply_soft_weighting(
    chunks: list[RetrievedChunk],
    *,
    reformulated: ReformulatedQuery,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Reordena `chunks` (ya finalistas, p.ej. tras rerank) por un score blando.

    Base de relevancia = -distance (menor distancia = mejor). Multiplicadores: temporal
    (si activo) y, opcionalmente, match de tech/sector mencionados en el brief. No
    expulsa candidatos: solo reordena.
    """
    if not chunks:
        return chunks
    techs = {t.lower() for t in reformulated.technologies}

    def score(chunk: RetrievedChunk) -> float:
        # Relevancia POSITIVA y monótona (menor distancia → mayor relevancia), en (0,1].
        # Se clampa la distancia a >=0 para robustez (la rama léxica guarda -ts_rank).
        # Con esta base, un peso <1 SIEMPRE degrada y uno >1 mejora, sin ambigüedad de
        # signo: weighted = base * weight; se ordena descendente.
        base = 1.0 / (1.0 + max(chunk.distance, 0.0))
        weight = 1.0
        if settings.temporal_decay_enabled:
            weight *= temporal_weight(
                chunk.metadata.get("year"),
                half_life_years=settings.temporal_half_life_years,
            )
        if settings.contextual_weighting_enabled:
            tech = str(chunk.metadata.get("main_technology", "")).lower()
            if tech and tech in techs:
                weight *= settings.contextual_tech_boost
            if chunk.metadata.get("client_sector") == reformulated.sector:
                weight *= settings.contextual_sector_boost
        return base * weight

    ranked = sorted(chunks, key=score, reverse=True)
    logger.info(
        "rag.soft_weighting",
        count=len(ranked),
        temporal=settings.temporal_decay_enabled,
        contextual=settings.contextual_weighting_enabled,
    )
    return ranked
