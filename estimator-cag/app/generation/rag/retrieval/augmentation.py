"""Fase 3 del RAG: ensamblar los chunks recuperados en un context_block acotado
por token budget. NO enriquece con fuentes externas (fuera de scope S09). Preserva
los chunk_ref para que la generación pueda citar (trazabilidad).

S11: capa de calidad opt-in (todo off = comportamiento pre-S11):
- `extract_keypoints`: compresión EXTRACTIVA determinista (limpia el vector, no llama al LLM).
- `reorder_by_edges`: reordena por extremos contra el 'lost-in-the-middle'.
"""

from __future__ import annotations

import re

import structlog

from app.foundations.config import Settings
from app.generation.rag.chunking.common import count_tokens
from app.generation.rag.schemas import AugmentedContext, RetrievalResult

logger = structlog.get_logger(__name__)

# Señal a conservar en la compresión extractiva: cifras + vocabulario de dominio.
_SIGNAL = re.compile(
    r"\d|\b(hora|d[ií]a|semana|m[oó]dulo|integr|api|auth|pago|panel|sensor|stripe|sap)",
    re.I,
)


def extract_keypoints(content: str, *, max_chars: int) -> str:
    """Compresión EXTRACTIVA determinista: reduce ruido conservando líneas con señal
    (cifras, tecnologías, verbos de acción). No llama al LLM. La compresión no busca
    tanto ahorrar tokens como limpiar el vector: quita artículos/relleno que ensucian
    la señal. Devuelve el contenido si ya es corto."""
    if len(content) <= max_chars:
        return content
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    kept = [ln for ln in lines if _SIGNAL.search(ln)] or lines
    out = " · ".join(kept)
    return out[:max_chars]


def reorder_by_edges(chunks: list) -> list:
    """Contra 'lost-in-the-middle': el más fuerte al principio, el 2º más fuerte al final,
    los débiles al centro. Entra ordenado por relevancia (menor distance = mejor)."""
    ordered = sorted(chunks, key=lambda c: c.distance)  # mejor→peor
    head, tail = [], []
    for i, chunk in enumerate(ordered):
        (head if i % 2 == 0 else tail).append(chunk)  # 0→head, 1→tail, 2→head...
    return head + tail[::-1]  # [1º,3º,5º,...,6º,4º,2º]


def assemble_context(
    retrieval: RetrievalResult,
    *,
    max_tokens: int,
    settings: Settings | None = None,
) -> AugmentedContext:
    """Concatena chunks (ordenados por distancia) hasta agotar el presupuesto de tokens.
    Cada bloque va etiquetado con su source_id (verificable contra included_refs) y su
    document_id (budget_id) para que la generación pueda atribuir y copiar evidencia.

    Con `settings`, aplica la capa de calidad S11 (opt-in): reorden por extremos y/o
    compresión extractiva. Sin `settings` (o toggles a False) → comportamiento pre-S11."""
    compress = bool(settings and settings.context_compression_enabled)
    reorder = bool(settings and settings.reorder_by_edges_enabled)
    keypoint_max_chars = settings.keypoint_max_chars if settings else 600

    chunks = reorder_by_edges(retrieval.chunks) if reorder else retrieval.chunks

    parts: list[str] = []
    included_refs: list[str] = []
    total = 0
    dropped = 0

    for chunk in chunks:
        content = extract_keypoints(chunk.content, max_chars=keypoint_max_chars) if compress else chunk.content
        block = (
            f"[source_id: {chunk.chunk_ref} | document_id: {chunk.metadata.get('budget_id', '')} "
            f"| type: {chunk.chunk_type} | distance: {chunk.distance}]\n{content}"
        )
        block_tokens = count_tokens(block)
        if total + block_tokens > max_tokens:
            dropped += 1
            continue
        parts.append(block)
        included_refs.append(chunk.chunk_ref)
        total += block_tokens

    context_block = "\n\n---\n\n".join(parts)
    logger.info(
        "rag.augmented",
        included=len(included_refs),
        dropped=dropped,
        tokens=total,
        compressed=compress,
        reordered=reorder,
    )
    return AugmentedContext(
        context_block=context_block,
        token_count=total,
        included_refs=included_refs,
        dropped=dropped,
    )
