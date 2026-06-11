"""Barre hnsw.ef_search midiendo recall@k (vs ground truth exacto) y latencia, para
encontrar el punto dulce: el ef más bajo con recall ~1.0 sin penalizar latencia.

El recall@k es el solape de ids entre el top-k del índice (search_chunks) y el top-k
exacto por fuerza bruta (search_chunks_exact). El punto dulce sube con el volumen del
corpus: a 1M vectores, 40 puede quedarse corto.

    uv run python -m scripts.tune_ef_search
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.persistence.repository import search_chunks, search_chunks_exact

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "secure backend service with token-based access control for banking applications",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]
EF_VALUES = [10, 20, 40, 80, 120, 200]


def _recall(hnsw_ids: list[int], exact_ids: list[int]) -> float:
    if not exact_ids:
        return 1.0
    return len(set(hnsw_ids) & set(exact_ids)) / len(exact_ids)


async def main(k: int, runs: int) -> None:
    embedder = LiteLLMEmbedder()
    vectors = [embedder.embed_one(q) for q in QUERIES]

    # Ground truth exacto por query (una vez).
    exact: list[list[int]] = []
    async with AsyncSessionLocal() as session:
        for vector in vectors:
            rows = await search_chunks_exact(session, query_vector=vector, k=k)
            exact.append([r._mapping["chunk_id"] for r in rows])

    print(f"k={k}  runs={runs}\n" + "=" * 64)
    print(f"{'ef_search':>10} | {'recall@k':>9} | {'p50_ms':>8}")
    print("-" * 64)
    for ef in EF_VALUES:
        recalls, latencies = [], []
        for vector, gt in zip(vectors, exact):
            for _ in range(runs):
                async with AsyncSessionLocal() as session:
                    started = time.perf_counter()
                    rows = await search_chunks(session, query_vector=vector, k=k, ef_search=ef)
                    latencies.append((time.perf_counter() - started) * 1000)
                ids = [r._mapping["chunk_id"] for r in rows]
                recalls.append(_recall(ids, gt))
        print(f"{ef:>10} | {statistics.mean(recalls):>9.3f} | {statistics.median(latencies):>8.2f}")
    print("=" * 64)
    print("Punto dulce: el ef más bajo con recall ~1.0 sin salto de latencia.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.k, args.runs))
