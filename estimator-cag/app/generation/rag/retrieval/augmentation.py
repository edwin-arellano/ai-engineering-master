"""Fase 3 del RAG: ensamblar los chunks recuperados en un context_block acotado
por token budget. NO enriquece con fuentes externas (fuera de scope S09). Preserva
los chunk_ref para que la generación pueda citar (trazabilidad)."""

from __future__ import annotations

import structlog

from app.generation.rag.chunking.common import count_tokens
from app.generation.rag.schemas import AugmentedContext, RetrievalResult

logger = structlog.get_logger(__name__)


def assemble_context(retrieval: RetrievalResult, *, max_tokens: int) -> AugmentedContext:
    """Concatena chunks (ordenados por distancia) hasta agotar el presupuesto de tokens.
    Cada bloque va etiquetado con su source_id (verificable contra included_refs) y su
    document_id (budget_id) para que la generación pueda atribuir y copiar evidencia."""
    parts: list[str] = []
    included_refs: list[str] = []
    total = 0
    dropped = 0

    for chunk in retrieval.chunks:
        block = (
            f"[source_id: {chunk.chunk_ref} | document_id: {chunk.metadata.get('budget_id', '')} "
            f"| type: {chunk.chunk_type} | distance: {chunk.distance}]\n{chunk.content}"
        )
        block_tokens = count_tokens(block)
        if total + block_tokens > max_tokens:
            dropped += 1
            continue
        parts.append(block)
        included_refs.append(chunk.chunk_ref)
        total += block_tokens

    context_block = "\n\n---\n\n".join(parts)
    logger.info("rag.augmented", included=len(included_refs), dropped=dropped, tokens=total)
    return AugmentedContext(
        context_block=context_block,
        token_count=total,
        included_refs=included_refs,
        dropped=dropped,
    )
