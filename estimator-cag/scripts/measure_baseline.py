"""Mide la latencia de la parte SQL de la búsqueda, aislada del embedder. La latencia
de /search está dominada por la llamada de embedding de la query (red), que enmascara
el tiempo de BD; para ver el efecto del índice hay que embeber UNA vez y cronometrar
solo la función de repositorio.

    --mode exact    → search_chunks_exact (seq scan forzado, ground truth)
    --mode indexed  → search_chunks (half-vec + HNSW, usa HNSW_EF_SEARCH de settings)

    uv run python -m scripts.measure_baseline --mode exact
    uv run python -m scripts.measure_baseline --mode indexed
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from app.foundations.config import get_settings
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.persistence.repository import search_chunks, search_chunks_exact

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "secure backend service with token-based access control for banking applications",
    "mobile application for restaurant reservations",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]


async def main(mode: str, runs: int, k: int) -> None:
    embedder = LiteLLMEmbedder()
    vectors = [embedder.embed_one(q) for q in QUERIES]  # embeber una sola vez
    settings = get_settings()

    print(f"mode={mode}  runs={runs}  k={k}\n" + "=" * 72)
    all_latencies: list[float] = []
    for query, vector in zip(QUERIES, vectors):
        latencies: list[float] = []
        for _ in range(runs):
            async with AsyncSessionLocal() as session:
                started = time.perf_counter()
                if mode == "exact":
                    await search_chunks_exact(session, query_vector=vector, k=k)
                else:
                    await search_chunks(
                        session,
                        query_vector=vector,
                        k=k,
                        ef_search=settings.hnsw_ef_search,
                    )
                latencies.append((time.perf_counter() - started) * 1000)
        med = statistics.median(latencies)
        all_latencies.extend(latencies)
        print(f"  {med:7.2f} ms (median)  {query[:60]}")
    print("=" * 72)
    print(
        f"  GLOBAL median: {statistics.median(all_latencies):.2f} ms "
        f"over {len(all_latencies)} runs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "indexed"], default="exact")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.runs, args.k))
