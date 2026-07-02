"""Health-check del índice vectorial (S11-06): diagnostica lo que RAGAS NO ve. RAGAS
mide contra el golden set (limpio); un corpus que se degrada en silencio (vectores malos,
duplicados, dimensiones mal) no lo detecta. Este arnés SOLO diagnostica (no corrige).

Por colección reporta: nº de chunks, huérfanos (embedding NULL), duplicados exactos de
content, embeddings con dimensión ≠ 1536, y la distribución de estimated_hours/year
(outliers que podrían falsear la búsqueda).

Uso: PYTHONPATH=. uv run python scripts/check_index_health.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.persistence.models import EMBEDDING_DIM

TABLES = ["budget_chunks", "transcript_chunks", "technical_doc_chunks"]


async def _scalar(session, sql: str) -> int:
    result = await session.execute(text(sql))
    return result.scalar() or 0


async def _check_table(session, table: str) -> dict:
    total = await _scalar(session, f"SELECT count(*) FROM {table}")
    orphans = await _scalar(session, f"SELECT count(*) FROM {table} WHERE embedding IS NULL")
    bad_dim = await _scalar(
        session,
        f"SELECT count(*) FROM {table} "
        f"WHERE embedding IS NOT NULL AND vector_dims(embedding) <> {EMBEDDING_DIM}",
    )
    dup_groups = await _scalar(
        session,
        f"SELECT count(*) FROM (SELECT content FROM {table} "
        f"GROUP BY content HAVING count(*) > 1) d",
    )
    return {
        "total": total,
        "orphans": orphans,
        "bad_dimension": bad_dim,
        "duplicate_content_groups": dup_groups,
    }


async def _hours_distribution(session) -> dict | None:
    """Distribución de estimated_hours/year en budget_chunks (outliers = ruido potencial)."""
    row = (
        await session.execute(
            text(
                "SELECT "
                "min((metadata->>'estimated_hours')::float), "
                "max((metadata->>'estimated_hours')::float), "
                "avg((metadata->>'estimated_hours')::float), "
                "min((metadata->>'year')::int), "
                "max((metadata->>'year')::int) "
                "FROM budget_chunks WHERE metadata ? 'estimated_hours'"
            )
        )
    ).first()
    if not row or row[0] is None:
        return None
    return {
        "hours_min": row[0],
        "hours_max": row[1],
        "hours_avg": round(row[2], 1),
        "year_min": row[3],
        "year_max": row[4],
    }


async def main() -> None:
    async with AsyncSessionLocal() as session:
        print("=== Salud del índice vectorial (diagnóstico; no corrige) ===\n")
        issues = 0
        for table in TABLES:
            try:
                stats = await _check_table(session, table)
            except Exception as exc:  # noqa: BLE001 — tabla ausente/no migrada
                print(f"[{table}] no disponible: {str(exc)[:80]}")
                continue
            flags = []
            if stats["orphans"]:
                flags.append(f"{stats['orphans']} huérfanos")
            if stats["bad_dimension"]:
                flags.append(f"{stats['bad_dimension']} dim≠{EMBEDDING_DIM}")
            if stats["duplicate_content_groups"]:
                flags.append(f"{stats['duplicate_content_groups']} grupos duplicados")
            issues += len(flags)
            status = "⚠ " + ", ".join(flags) if flags else "OK"
            print(f"[{table}] {stats['total']} chunks → {status}")

        dist = await _hours_distribution(session)
        if dist:
            print(
                f"\nbudget_chunks · estimated_hours: min={dist['hours_min']:g} "
                f"max={dist['hours_max']:g} avg={dist['hours_avg']} | "
                f"year: {dist['year_min']}–{dist['year_max']}"
            )
        print(f"\nResumen: {issues} señal(es) de salud a revisar.")


if __name__ == "__main__":
    asyncio.run(main())
