"""Marco de decisión CAG/RAG/híbrido. Las cuatro restricciones del CAG operan a la
vez: basta con que una falle para que la arquitectura no sea viable. El evaluador
consume los números REALES del baseline pre-S06 (evals/stress/results.csv).
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Architecture(Enum):
    PURE_CAG = "pure_cag"
    HYBRID_CAG_RAG = "hybrid_cag_rag"
    PURE_RAG = "pure_rag"


@dataclass
class CAGViability:
    fits_in_context_window: bool
    cost_per_query_acceptable: bool
    latency_acceptable: bool
    quality_holds_with_load: bool

    def is_viable(self) -> bool:
        return all(
            [
                self.fits_in_context_window,
                self.cost_per_query_acceptable,
                self.latency_acceptable,
                self.quality_holds_with_load,
            ]
        )


@dataclass
class CorpusProfile:
    total_tokens: int
    update_frequency_days: int
    requires_source_attribution: bool
    requires_per_user_access_control: bool


@dataclass
class ModelProfile:
    context_window: int
    cost_per_million_input_tokens: float


def recommend_architecture(
    corpus: CorpusProfile, model: ModelProfile, *, usable_ratio: float = 0.7
) -> Architecture:
    context_usage = corpus.total_tokens / (model.context_window * usable_ratio)
    if corpus.requires_source_attribution:
        return Architecture.PURE_RAG
    if corpus.requires_per_user_access_control:
        return Architecture.PURE_RAG
    if context_usage > 1.0:
        return Architecture.PURE_RAG
    if corpus.update_frequency_days < 7:
        return Architecture.PURE_RAG
    if corpus.update_frequency_days > 90 and context_usage < 0.3:
        return Architecture.PURE_CAG
    return Architecture.HYBRID_CAG_RAG


@dataclass
class BaselineSummary:
    latency_p50: float
    latency_p95: float
    cost_per_turn_mean: float
    turns: int


def summarize_baseline(results_csv: Path) -> BaselineSummary:
    """Lee evals/stress/results.csv y resume latencia y coste por turno.

    El runner pre-S06 vuelca la latencia en MILISEGUNDOS (columna ``latency_ms``)
    y el coste por turno en USD (columna ``cost_usd``). Aquí la latencia se
    convierte a segundos para poder compararla contra el SLA, que está expresado
    en segundos (``cag_latency_sla_seconds``).
    """
    latencies_seconds: list[float] = []
    costs: list[float] = []
    with open(results_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lat_ms = row.get("latency_ms")
            cost = row.get("cost_usd")
            if lat_ms:
                latencies_seconds.append(float(lat_ms) / 1000.0)
            if cost:
                costs.append(float(cost))
    latencies_seconds.sort()
    p95 = (
        latencies_seconds[int(len(latencies_seconds) * 0.95)]
        if latencies_seconds
        else 0.0
    )
    return BaselineSummary(
        latency_p50=statistics.median(latencies_seconds) if latencies_seconds else 0.0,
        latency_p95=p95,
        cost_per_turn_mean=statistics.mean(costs) if costs else 0.0,
        turns=len(latencies_seconds),
    )


@dataclass
class IngestionArchitecture:
    """Combina el perfil del corpus/modelo con el baseline empírico para emitir
    una recomendación defendible ante un stakeholder con números concretos.
    """

    corpus: CorpusProfile
    model: ModelProfile
    baseline: BaselineSummary
    latency_sla_seconds: float
    cost_per_turn_budget_usd: float
    usable_ratio: float = 0.7

    def viability(self) -> CAGViability:
        return CAGViability(
            fits_in_context_window=self.corpus.total_tokens
            <= self.model.context_window * self.usable_ratio,
            cost_per_query_acceptable=self.baseline.cost_per_turn_mean
            <= self.cost_per_turn_budget_usd,
            latency_acceptable=self.baseline.latency_p95 <= self.latency_sla_seconds,
            # lost-in-the-middle / degradación bajo carga: el baseline mostró fallos
            # bajo saturación, así que la calidad NO se sostiene con el corpus completo.
            quality_holds_with_load=False,
        )

    def recommend(self) -> Architecture:
        return recommend_architecture(
            self.corpus, self.model, usable_ratio=self.usable_ratio
        )
