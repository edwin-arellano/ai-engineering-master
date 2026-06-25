"""Verifica que el modelo de reranking descarga y carga correctamente. Ejecutar
tras levantar el entorno: python -m app.generation.rag.retrieval.verify_reranker"""

from __future__ import annotations

from app.generation.rag.retrieval.reranker import get_reranker
from app.generation.rag.schemas import RetrievedChunk


def main() -> None:
    reranker = get_reranker()
    sample = [
        RetrievedChunk(chunk_id=1, document_id=1, chunk_type="budget_component",
                       content="Plataforma de e-commerce con catálogo y carrito.",
                       distance=0.4, metadata={}),
        RetrievedChunk(chunk_id=2, document_id=2, chunk_type="budget_component",
                       content="App de pagos móviles con pasarela.", distance=0.41, metadata={}),
    ]
    out = reranker.rerank("tienda online con catálogo de productos", sample, top_k=2)
    print(f"OK reranker cargó y reordenó {len(out)} candidatos.")
    print("Orden:", [c.chunk_id for c in out])


if __name__ == "__main__":
    main()
