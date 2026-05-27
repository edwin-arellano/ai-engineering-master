"""Métricas del stress test. Reutilizan MetricResult del framework de evals.

Operan sobre el `turn_observed` (latencia, coste) o sobre el snapshot de sesión
(memory drift). Sin embeddings, sin LLM-as-judge: match exacto case-insensitive.
"""

from __future__ import annotations

from dataclasses import dataclass

from evals.metrics import MetricResult


@dataclass
class LatencyBudgetMetric:
    """1.0 si latency_ms ≤ budget_ms; 0.0 si no."""

    budget_ms: int

    def evaluate(self, observation: dict) -> MetricResult:
        latency = observation.get("latency_ms", 0.0)
        passed = latency <= self.budget_ms
        return MetricResult(
            name="latency_budget",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"latency_ms": latency, "budget_ms": self.budget_ms},
        )


@dataclass
class CostBudgetMetric:
    """1.0 si cost_usd ≤ budget_usd; 0.0 si no."""

    budget_usd: float

    def evaluate(self, observation: dict) -> MetricResult:
        cost = observation.get("cost_usd", 0.0)
        passed = cost <= self.budget_usd
        return MetricResult(
            name="cost_budget",
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"cost_usd": cost, "budget_usd": self.budget_usd},
        )


@dataclass
class MemoryDriftMetric:
    """1.0 si el fact declarado aparece en la memoria persistente del snapshot.

    La "memoria" del CAG es lo que sobrevive entre turnos y se reinyecta al
    contexto del siguiente: el `running_summary` de la compresión (expuesto como
    `last_summary`), los `anchored_facts` y el `project_metadata`. NO se mira el
    `summary` del EstimationResult del turno: ese es output, no memoria, y
    estaría contaminado por los mensajes recientes aún sin comprimir, lo que
    inflaría el recall artificialmente.

    Match exacto, case-insensitive, sin normalización semántica (determinismo >
    sofisticación). `where` selecciona en qué partes buscar; las etiquetas
    lógicas ("summary"/"anchors"/"metadata") mapean a las claves reales del
    `SessionDebugResponse` (`last_summary`/`anchored_facts`/`project_metadata`),
    de modo que el snapshot del endpoint debug se evalúa sin transformación.
    """

    fact: str
    where: tuple[str, ...] = ("summary", "anchors", "metadata")

    def evaluate(self, session_snapshot: dict) -> MetricResult:
        needle = self.fact.lower()
        haystacks: list[str] = []

        if "summary" in self.where:
            haystacks.append(str(session_snapshot.get("last_summary") or "").lower())
        if "anchors" in self.where:
            facts = session_snapshot.get("anchored_facts") or []
            haystacks.append(" ".join(facts).lower())
        if "metadata" in self.where:
            metadata = session_snapshot.get("project_metadata") or {}
            haystacks.append(str(metadata).lower())

        found = any(needle in haystack for haystack in haystacks)
        return MetricResult(
            name="memory_drift",
            score=1.0 if found else 0.0,
            passed=found,
            details={"fact": self.fact, "found": found, "where": list(self.where)},
        )
