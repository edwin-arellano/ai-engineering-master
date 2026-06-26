"""EXPLAIN ANALYZE de la búsqueda half-vec debe usar el índice HNSW, no seq scan.

Es la prueba del antipatrón del operador: si el plan cae a Seq Scan, la realineación
de search_chunks está mal (expresión/operador desalineados con halfvec_cosine_ops).
La query del EXPLAIN reproduce exactamente la expresión que emite
repository._build_halfvec_search_stmt (`embedding::halfvec(1536) <=> ...`).

Requiere: Postgres migrado (0002), corpus ingestado y, idealmente, el seed sintético
para que el planner prefiera el índice sobre el seq scan en un corpus diminuto.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.persistence.models import EMBEDDING_DIM


@pytest.mark.integration
def test_search_uses_hnsw_index() -> None:
    async def run() -> str:
        embedder = LiteLLMEmbedder()
        vector = embedder.embed_one("authentication backend for fintech")
        literal = "[" + ",".join(str(x) for x in vector) + "]"
        async with AsyncSessionLocal() as session:
            await session.execute(text("SET LOCAL hnsw.ef_search = 40"))
            plan = await session.execute(
                text(
                    f"EXPLAIN ANALYZE SELECT id, "
                    f"(embedding::halfvec({EMBEDDING_DIM})) <=> "
                    f"'{literal}'::halfvec({EMBEDDING_DIM}) AS d "
                    f"FROM budget_chunks ORDER BY d LIMIT 5"
                )
            )
            return "\n".join(row[0] for row in plan.all())

    try:
        plan_text = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DB/índice no disponible: {exc}")

    assert "budget_chunks_embedding_halfvec_idx" in plan_text, plan_text
    assert "Seq Scan" not in plan_text, plan_text
