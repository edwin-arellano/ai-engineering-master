"""Test de integración del embedder (pega a la API real de embeddings).

Aquí se fija el comportamiento real de la versión instalada de LiteLLM: acceso a
``response.data`` (vía ``_extract_embedding``), presencia de ``usage`` y la clase
de ``RateLimitError`` (importable como ``litellm.RateLimitError``).
"""

from __future__ import annotations

import pytest

from app.embedding_pipeline.embedder import LiteLLMEmbedder
from app.embedding_pipeline.schemas import Chunk


@pytest.mark.integration
def test_embed_one_returns_1536_floats():
    vector = LiteLLMEmbedder().embed_one("OAuth 2.0 authentication backend")
    assert len(vector) == 1536
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.integration
def test_embed_many_uses_multiple_batches_and_counts_tokens():
    # batch_size pequeño para forzar más de un batch sin generar 100+ chunks reales.
    embedder = LiteLLMEmbedder(batch_size=2)
    chunks = [
        Chunk(chunk_id=f"c{i}", text=f"componente de prueba número {i}", metadata={}, token_count=0)
        for i in range(5)
    ]
    embedded = embedder.embed_many(chunks)
    assert len(embedded) == 5
    assert all(len(e.embedding) == 1536 for e in embedded)
    assert embedder.last_run_total_tokens > 0
    assert embedder.last_run_cost_usd > 0.0
