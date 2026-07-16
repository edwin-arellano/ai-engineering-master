"""Ancla numérica DETERMINISTA para detectar alucinaciones de cifra: parsea la cantidad
literal de la evidencia citada, la lleva a días-ingeniero y la compara con la línea.
Para cálculo NUNCA se usa un modelo.

Nivelación con el corpus real: la `evidence` copiada de los chunks viene en formato
"Estimated hours: 90" / "Hours: 120" (unidad ANTES del número), no solo "120 horas"
(número antes). El parser cubre ambos órdenes, en español e inglés."""

from __future__ import annotations

import re

from pydantic import BaseModel

# Se cubren los dos órdenes: "120 horas"/"120h" y "Hours: 120"/"horas 120".
_HOURS = [
    re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:horas?|hours?|hrs?|h)\b", re.I),
    re.compile(r"\b(?:horas?|hours?|hrs?)\s*[:=]?\s*(\d+(?:[.,]\d+)?)", re.I),
]
_DAYS = [
    re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:d[ií]as?|days?|d)\b", re.I),
    re.compile(r"\b(?:d[ií]as?|days?)\s*[:=]?\s*(\d+(?:[.,]\d+)?)", re.I),
]


class AnchorResult(BaseModel):
    evidence_days: float | None  # cifra de la evidencia en días-ingeniero (None si no se parsea)
    line_days: float
    deviation: float | None  # |line - evidence| / max(evidence, 1)
    numeric_fail: bool  # deviation > tolerance


def _first_match(patterns: list[re.Pattern[str]], text: str) -> float | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def anchor_line(
    *,
    line_engineer_days: float,
    evidence: str,
    hours_per_day: float,
    tolerance: float,
) -> AnchorResult:
    """Convierte la cifra de la evidencia a días-ingeniero y mide la desviación relativa.
    Prioriza horas sobre días (el corpus mide en horas). Sin cifra parseable → no bloquea."""
    hours = _first_match(_HOURS, evidence)
    evidence_days: float | None
    if hours is not None:
        evidence_days = hours / hours_per_day
    else:
        evidence_days = _first_match(_DAYS, evidence)
    if evidence_days is None:
        return AnchorResult(
            evidence_days=None,
            line_days=line_engineer_days,
            deviation=None,
            numeric_fail=False,  # sin cifra parseable → no bloquea
        )
    deviation = abs(line_engineer_days - evidence_days) / max(evidence_days, 1.0)
    return AnchorResult(
        evidence_days=evidence_days,
        line_days=line_engineer_days,
        deviation=deviation,
        numeric_fail=deviation > tolerance,
    )
