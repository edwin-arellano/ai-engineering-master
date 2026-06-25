"""Reranker cross-encoder (integration). Descarga/carga BAAI/bge-reranker-v2-m3 la
1ª vez (pesos de HuggingFace; luego cachea). Verifica que reordena (el candidato más
relevante sube) y respeta top_k."""

from __future__ import annotations

import pytest

from app.generation.rag.schemas import RetrievedChunk


def _chunk(cid: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=cid, chunk_type="budget_component",
        content=content, distance=0.4, metadata={},
    )


@pytest.mark.integration
def test_reranker_loads_and_reorders() -> None:
    from app.generation.rag.retrieval.reranker import get_reranker

    reranker = get_reranker()
    # El candidato 3 es el más relevante para la query; entra en última posición.
    candidates = [
        _chunk(1, "App de pagos móviles con pasarela y antifraude."),
        _chunk(2, "Dashboard de telemetría industrial y OEE."),
        _chunk(3, "Tienda online con catálogo de productos, carrito y checkout."),
    ]
    out = reranker.rerank(
        "plataforma de e-commerce con catálogo y carrito de compra", candidates, top_k=2
    )
    assert len(out) == 2  # respeta top_k
    assert out[0].chunk_id == 3  # el más relevante sube al primer puesto


@pytest.mark.integration
def test_reranker_empty_candidates() -> None:
    from app.generation.rag.retrieval.reranker import get_reranker

    assert get_reranker().rerank("cualquier query", [], top_k=5) == []


@pytest.mark.integration
def test_reranker_is_singleton() -> None:
    from app.generation.rag.retrieval.reranker import get_reranker

    assert get_reranker() is get_reranker()
