"""Comparador de estrategias de chunking. Por estrategia reporta nº de chunks,
huérfanos, percentiles de tokens, latencia y coste; y si hay query, el coseno
top contra los chunks (señal de calidad de la separación).
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field

from app.generation.rag.chunking.registry import build_chunker
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.schemas import Budget


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _p(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[min(len(s) - 1, int(len(s) * q))])


@dataclass
class StrategyReport:
    name: str
    num_chunks: int
    orphan_count: int
    min_tokens: int
    p50_tokens: float
    p95_tokens: float
    max_tokens: int
    latency_ms: float
    top_scores: list[float] = field(default_factory=list)


def compare_strategies(
    budgets: list[Budget],
    strategy_names: list[str],
    *,
    query: str | None = None,
    embedder: LiteLLMEmbedder | None = None,
    wrapper=None,
    top_k: int = 5,
) -> list[StrategyReport]:
    embedder = embedder or LiteLLMEmbedder()
    reports: list[StrategyReport] = []
    query_vec = embedder.embed_one(query) if query else None
    for name in strategy_names:
        chunker = build_chunker(name, embedder=embedder, wrapper=wrapper)
        started = time.perf_counter()
        chunks = chunker.chunk(budgets)
        latency_ms = (time.perf_counter() - started) * 1000
        sizes = [c.token_count for c in chunks]
        scores: list[float] = []
        if query_vec is not None and chunks:
            non_orphans = [c for c in chunks if not c.is_orphan]
            vecs = embedder.embed_many(non_orphans) if non_orphans else []
            scores = sorted(
                (_cosine(query_vec, c.embedding) for c in vecs), reverse=True
            )[:top_k]
        reports.append(
            StrategyReport(
                name=name,
                num_chunks=len(chunks),
                orphan_count=sum(1 for c in chunks if c.is_orphan),
                min_tokens=min(sizes) if sizes else 0,
                p50_tokens=statistics.median(sizes) if sizes else 0.0,
                p95_tokens=_p(sizes, 0.95),
                max_tokens=max(sizes) if sizes else 0,
                latency_ms=round(latency_ms, 1),
                top_scores=[round(s, 4) for s in scores],
            )
        )
    return reports
