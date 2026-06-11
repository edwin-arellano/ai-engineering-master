"""Infla la tabla chunks con N chunks sintéticos (vectores aleatorios normalizados)
para que las diferencias de indexación se aprecien: el corpus real son decenas de
chunks y con eso el seq scan ya es instantáneo, no se ve nada. Los chunks reales
siguen ahí y son los que importan para el recall; los sintéticos son ruido (vectores
gaussianos normalizados, ~ortogonales a las queries reales en 1536 dims).

DB-directo (usa AsyncSessionLocal). Inserta por lotes. Idempotente por source_path del
documento sintético: rellena hasta --total (no duplica al re-ejecutar). Usa --reset
para borrar los sintéticos y regenerarlos desde cero.

    uv run python -m scripts.seed_synthetic_chunks --total 30000
    uv run python -m scripts.seed_synthetic_chunks --total 30000 --reset
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import structlog
from sqlalchemy import delete, func, insert, select, text

from app.generation.rag.persistence.database import AsyncSessionLocal, engine
from app.generation.rag.persistence.models import EMBEDDING_DIM, ChunkRow, DocumentRow

logger = structlog.get_logger(__name__)

SYNTHETIC_SOURCE_PATH = "synthetic/stress-corpus"
BATCH_SIZE = 1000


def _normalized_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """n vectores gaussianos normalizados a norma 1 (como los de OpenAI)."""
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


async def _ensure_synthetic_document(session) -> int:
    existing = await session.execute(
        select(DocumentRow.id).where(DocumentRow.source_path == SYNTHETIC_SOURCE_PATH)
    )
    doc_id = existing.scalar_one_or_none()
    if doc_id is not None:
        return doc_id
    doc = DocumentRow(
        source_path=SYNTHETIC_SOURCE_PATH,
        document_type="synthetic",
        metadata_={"synthetic": True},
    )
    session.add(doc)
    await session.flush()
    await session.commit()
    return doc.id


async def _synthetic_chunk_count(session, doc_id: int) -> int:
    result = await session.execute(
        select(func.count()).select_from(ChunkRow).where(ChunkRow.document_id == doc_id)
    )
    return int(result.scalar_one())


async def main(total: int, reset: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    async with AsyncSessionLocal() as session:
        if reset:
            doc = await session.execute(
                select(DocumentRow.id).where(DocumentRow.source_path == SYNTHETIC_SOURCE_PATH)
            )
            doc_id = doc.scalar_one_or_none()
            if doc_id is not None:
                # ON DELETE CASCADE limpia los chunks sintéticos.
                await session.execute(delete(DocumentRow).where(DocumentRow.id == doc_id))
                await session.commit()
                logger.info("seed.reset", document_id=doc_id)

        doc_id = await _ensure_synthetic_document(session)

        # Idempotencia: solo insertamos lo que falta para llegar a `total`.
        existing = await _synthetic_chunk_count(session, doc_id)
        if existing >= total:
            logger.info("seed.noop", document_id=doc_id, existing=existing, total=total)
            return
        to_insert = total - existing

        inserted = 0
        while inserted < to_insert:
            n = min(BATCH_SIZE, to_insert - inserted)
            vectors = _normalized_vectors(n, EMBEDDING_DIM, rng)
            rows = [
                {
                    "document_id": doc_id,
                    "chunk_type": "synthetic",
                    "content": f"synthetic chunk {existing + inserted + i}",
                    "embedding": vectors[i].tolist(),
                    "metadata_": {"synthetic": True},
                }
                for i in range(n)
            ]
            await session.execute(insert(ChunkRow), rows)
            await session.commit()
            inserted += n
            if inserted % 5000 == 0 or inserted == to_insert:
                logger.info("seed.progress", inserted=inserted, target=to_insert)

    # Estadísticas frescas para el planner tras una carga grande.
    async with engine.begin() as conn:
        await conn.execute(text("ANALYZE chunks"))
    logger.info("seed.done", inserted=inserted, total=total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.total, args.reset, args.seed))
