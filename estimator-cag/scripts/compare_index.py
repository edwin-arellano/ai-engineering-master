"""Compara el índice HNSW float32 (vector) vs half-vec sobre el corpus actual:
tamaño en disco, latencia y recall (vs exacto). Crea/dropea el índice float32 ad-hoc
(el half-vec lo gestiona la migración 0002), así demuestra el 235→117 MB sin merma de
recall sin dejar dos índices vectoriales en el schema.

    uv run python -m scripts.compare_index
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import cast, select, text

from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal, engine
from app.generation.rag.persistence.models import EMBEDDING_DIM, ChunkRow
from app.generation.rag.persistence.repository import search_chunks_exact

QUERIES = [
    "REST API development with JWT authentication for financial sector",
    "integration with external system",
    "migration from monolith to microservices architecture using Kubernetes",
]
FLOAT32_IDX = "chunks_embedding_float32_idx"
HALFVEC_IDX = "chunks_embedding_halfvec_idx"


async def _index_size(conn, name: str) -> str:
    res = await conn.execute(
        text("SELECT pg_size_pretty(pg_relation_size(:n))"), {"n": name}
    )
    return res.scalar_one()


def _recall(got: list[int], gt: list[int]) -> float:
    return 1.0 if not gt else len(set(got) & set(gt)) / len(gt)


async def main(k: int, runs: int) -> None:
    embedder = LiteLLMEmbedder()
    vectors = [embedder.embed_one(q) for q in QUERIES]

    async with AsyncSessionLocal() as session:
        gts = [
            [r._mapping["chunk_id"] for r in await search_chunks_exact(session, query_vector=v, k=k)]
            for v in vectors
        ]

    # Crear el índice float32 ad-hoc (el half-vec ya existe por la migración 0002).
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP INDEX IF EXISTS {FLOAT32_IDX}"))
        await conn.execute(
            text(
                f"CREATE INDEX {FLOAT32_IDX} ON chunks "
                f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128)"
            )
        )
        size_f32 = await _index_size(conn, FLOAT32_IDX)
        size_half = await _index_size(conn, HALFVEC_IDX)

    def f32_stmt(v):
        d = ChunkRow.embedding.cosine_distance(v)
        return select(ChunkRow.id.label("chunk_id"), d.label("distance")).order_by(d).limit(k)

    def half_stmt(v):
        d = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIM)).cosine_distance(v)
        return select(ChunkRow.id.label("chunk_id"), d.label("distance")).order_by(d).limit(k)

    async def measure(stmt_fn):
        recalls, lats = [], []
        for v, gt in zip(vectors, gts):
            for _ in range(runs):
                async with AsyncSessionLocal() as s:
                    t0 = time.perf_counter()
                    rows = (await s.execute(stmt_fn(v))).all()
                    lats.append((time.perf_counter() - t0) * 1000)
                recalls.append(_recall([r._mapping["chunk_id"] for r in rows], gt))
        return statistics.mean(recalls), statistics.median(lats)

    r_f32, l_f32 = await measure(f32_stmt)
    r_half, l_half = await measure(half_stmt)

    print("=" * 64)
    print(f"{'índice':<10} | {'tamaño':>10} | {'recall@k':>9} | {'p50_ms':>8}")
    print("-" * 64)
    print(f"{'float32':<10} | {size_f32:>10} | {r_f32:>9.3f} | {l_f32:>8.2f}")
    print(f"{'half-vec':<10} | {size_half:>10} | {r_half:>9.3f} | {l_half:>8.2f}")
    print("=" * 64)

    # Limpieza: dejamos solo el half-vec adoptado.
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP INDEX IF EXISTS {FLOAT32_IDX}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.k, args.runs))
