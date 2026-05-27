"""Framework mínimo de métricas para evals. Patrón MetricResult reutilizable.

Tres métricas base que formalizan lo que `_check_case` hacía con asserts:
- SchemaAdherenceMetric: la salida respeta el contrato (out-of-scope esperado o no).
- CostBoundsMetric: el coste estimado cae en el rango esperado (tolerancia generosa).
- ContentRecallMetric: nombre/tecnologías esperadas aparecen en la salida.

`run_all_metrics` ejecuta una lista de métricas sobre una observación y agrega.
El stress test (Bloque 4) añade métricas nuevas reutilizando este mismo patrón.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MetricResult:
    """Resultado de evaluar una métrica sobre una observación."""

    name: str
    score: float  # 0.0–1.0
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Metric(Protocol):
    """Contrato de una métrica: recibe una observación, devuelve un MetricResult."""

    def evaluate(self, observation: Any) -> MetricResult: ...


def run_all_metrics(metrics: list[Metric], observation: Any) -> list[MetricResult]:
    """Ejecuta todas las métricas sobre una observación."""
    return [m.evaluate(observation) for m in metrics]


# ---------------------------------------------------------------------------
# Métricas base (formalizan _check_case)
# ---------------------------------------------------------------------------


@dataclass
class SchemaAdherenceMetric:
    """1.0 si el flag out_of_scope observado coincide con el esperado."""

    expected_out_of_scope: bool

    def evaluate(self, observation: dict) -> MetricResult:
        summary = observation.get("summary", "")
        is_out = summary.startswith("Out of scope:")
        passed = is_out == self.expected_out_of_scope
        return MetricResult(
            name="schema_adherence",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "observed_out_of_scope": is_out,
                "expected": self.expected_out_of_scope,
            },
        )


@dataclass
class CostBoundsMetric:
    """1.0 si total_cost_eur cae en [low*0.5, high*1.5] (tolerancia de primera pasada)."""

    low: int
    high: int

    def evaluate(self, observation: dict) -> MetricResult:
        cost = observation.get("total_cost_eur", 0)
        lo, hi = self.low * 0.5, self.high * 1.5
        passed = lo <= cost <= hi
        return MetricResult(
            name="cost_bounds",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"cost_eur": cost, "bounds": [self.low, self.high]},
        )


@dataclass
class PhaseCountMetric:
    """1.0 si el número de phases cae en [low, high] (rango cerrado, sin tolerancia).

    Formaliza la comprobación de `phase_count_range` que hacía `_check_case`.
    A diferencia de `CostBoundsMetric`, el rango es exacto: el dataset ya define
    bandas anchas, así que no se aplica tolerancia extra.
    """

    low: int
    high: int

    def evaluate(self, observation: dict) -> MetricResult:
        count = len(observation.get("phases", []))
        passed = self.low <= count <= self.high
        return MetricResult(
            name="phase_count",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"phase_count": count, "bounds": [self.low, self.high]},
        )


@dataclass
class ContentRecallMetric:
    """Mide presencia de términos esperados en summary o phases (case-insensitive).

    `require_all=True` (default) exige que TODOS los términos aparezcan: es la
    definición canónica de recall del framework, apropiada para un término
    obligatorio único como `project_name_contains`.

    `require_all=False` exige al menos uno: el modo correcto para el campo
    `technologies_any_of` del golden dataset, donde los términos son alternativas
    (p. ej. "Postgres"/"PostgreSQL", o "React Native"/"Swift"/"Kotlin") y exigir
    todas sería imposible por construcción. `score` es siempre el recall fraccional
    (términos hallados / esperados) independientemente del modo.
    """

    expected_terms: list[str]
    require_all: bool = True

    def evaluate(self, observation: dict) -> MetricResult:
        haystack = observation.get("summary", "").lower()
        for phase in observation.get("phases", []):
            haystack += " " + " ".join(
                str(v).lower() for v in phase.values() if isinstance(v, str)
            )
        found = [t for t in self.expected_terms if t.lower() in haystack]
        if not self.expected_terms:
            passed = True
        elif self.require_all:
            passed = len(found) == len(self.expected_terms)
        else:
            passed = len(found) >= 1
        score = len(found) / len(self.expected_terms) if self.expected_terms else 1.0
        return MetricResult(
            name="content_recall",
            score=score,
            passed=passed,
            details={
                "found": found,
                "expected": self.expected_terms,
                "require_all": self.require_all,
            },
        )
