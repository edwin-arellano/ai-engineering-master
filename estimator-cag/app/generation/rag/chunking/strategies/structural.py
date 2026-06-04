"""Chunking estructural de presupuestos JSON. Granularidad: un componente = un
chunk. El texto combina los detalles del componente con CONTEXTUAL CHUNK HEADERS
del presupuesto padre (sector, año, tecnología, summary) — la palanca de mayor
ROI conocida en RAG (versión estática y barata de Contextual Retrieval, sin LLM).
La metadata filtrable va FUERA del texto embebido.
"""

from __future__ import annotations

import structlog

from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk

logger = structlog.get_logger(__name__)

# Umbral de aviso: un chunk por encima de esto es candidato a discutir en directo.
# NO se parte (el ejercicio pide no hacer splitting de descripciones largas).
LONG_CHUNK_WARN_TOKENS = 512


class StructuralChunker(Chunker):
    """Un componente del presupuesto = un chunk (con contextual headers)."""

    name = "structural"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                text = common.render_component_text(c, parent)
                tokens = common.count_tokens(text)
                if tokens > LONG_CHUNK_WARN_TOKENS:
                    logger.warning(
                        "chunk.unusually_large",
                        chunk_id=f"{budget.budget_id}::{c.component_id}",
                        token_count=tokens,
                    )
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{c.component_id}",
                        text=text,
                        metadata=common.build_metadata(c, budget, strategy=self.name),
                        token_count=tokens,
                        is_orphan=common.is_orphan(tokens),
                    )
                )
        return chunks
