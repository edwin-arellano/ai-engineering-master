"""Acumuladores de métricas de observabilidad para una llamada y un turno.

``CallMetrics`` describe una única llamada al LLM. ``TurnMetrics`` agrega todas
las llamadas de un turno (en modo actor son 2: actor + extractor; en ACB son
muchas más). El servicio crea un ``TurnMetrics`` por turno y lo pasa como sink
opcional al wrapper; el wrapper le añade una ``CallMetrics`` por cada llamada.

Diseño deliberado: el sink se pasa explícitamente por parámetro (no contextvar)
para que el flujo de datos sea trivial de razonar y seguro con el threadpool
de FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CallMetrics:
    """Métricas de una única llamada al LLM."""

    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


@dataclass
class TurnMetrics:
    """Acumulador de todas las llamadas LLM de un turno."""

    calls: list[CallMetrics] = field(default_factory=list)

    def add(self, call: CallMetrics) -> None:
        self.calls.append(call)

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def llm_latency_ms(self) -> float:
        """Suma de latencias de las llamadas LLM (no es el wall-clock del turno)."""
        return sum(c.latency_ms for c in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)
