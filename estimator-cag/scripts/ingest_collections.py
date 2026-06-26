"""Orquesta la ingesta de las colecciones no-budget (S10): siembra technical_docs si
falta el directorio, e ingesta transcripts y technical_docs a sus tablas vía funciones
directas (AsyncSessionLocal). Imprime conteos por colección. Idempotente.

Budgets se ingestan aparte con scripts/reingest_with_tasks.py (camino HTTP existente).

    uv run python -m scripts.ingest_collections
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.ingest.collections.technical_docs import (
    TECHNICAL_DOCS_DIR,
    ingest_technical_docs,
)
from app.ingest.collections.transcripts import ingest_transcripts
from scripts.seed_technical_docs import seed_technical_docs


async def _run() -> None:
    embedder = LiteLLMEmbedder()

    # 1. Siembra el corpus técnico si aún no existe.
    tech_dir = Path(TECHNICAL_DOCS_DIR)
    if not tech_dir.exists() or not any(tech_dir.glob("*.md")):
        print("Sembrando data/technical_docs/ desde budgets_sample.json...")
        seed_technical_docs()

    # 2. Ingesta transcripts.
    print("\n== transcripts ==")
    transcripts = await ingest_transcripts(
        embedder=embedder, session_factory=AsyncSessionLocal
    )
    print(f"  -> {transcripts}")

    # 3. Ingesta technical_docs.
    print("\n== technical_docs ==")
    technical = await ingest_technical_docs(
        embedder=embedder, session_factory=AsyncSessionLocal
    )
    print(f"  -> {technical}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
