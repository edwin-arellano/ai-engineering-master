"""RetrievalPipeline — composición de los 4 modos con búsquedas mockeadas (sin DB
ni modelo). Verifica: vector/hybrid llaman a las ramas correctas, hybrid fusiona con
RRF, y reranking recorta a rag_top_k usando el recall amplio."""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.generation.rag.retrieval import pipeline as pipeline_module
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.schemas import ReformulatedQuery, RetrievedChunk


def _chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=cid, chunk_type="budget_component",
        content=f"chunk {cid}", distance=0.1, metadata={"budget_id": f"BUD-{cid}"},
    )


def _reformulated() -> ReformulatedQuery:
    return ReformulatedQuery(
        project_function="x", sector="finance", scale="medium", search_text="query densa"
    )


@pytest.fixture
def settings():
    # rag_top_k=3, candidate_pool=6 para poder observar recorte y recall amplio.
    return get_settings().model_copy(
        update={"rag_top_k": 3, "retrieval_candidate_pool_size": 6}
    )


def _build_pipeline(monkeypatch, *, vector_ids, lexical_ids):
    pipe = RetrievalPipeline(embedder=object(), session_factory=object())
    calls = {"vector_k": None, "lexical_k": None, "rerank_top_k": None}

    async def fake_vector(*, model=None, search_text, k, filters, ef_search):
        calls["vector_k"] = k
        return [_chunk(c) for c in vector_ids]

    async def fake_lexical(*, model=None, query_text, k, filters=None):
        calls["lexical_k"] = k
        return [_chunk(c) for c in lexical_ids]

    monkeypatch.setattr(pipe, "_vector_search", fake_vector)
    monkeypatch.setattr(pipe._fulltext, "search", fake_lexical)

    class _FakeReranker:
        def rerank(self, query, candidates, top_k):
            calls["rerank_top_k"] = top_k
            # Reordena al revés para detectar que corrió; respeta top_k.
            return list(reversed(candidates))[:top_k]

    monkeypatch.setattr(pipeline_module, "get_reranker", lambda: _FakeReranker())
    return pipe, calls


async def test_vector_no_rerank(monkeypatch, settings):
    pipe, calls = _build_pipeline(monkeypatch, vector_ids=[1, 2, 3, 4, 5], lexical_ids=[9])
    result = await pipe.retrieve(
        reformulated=_reformulated(), settings=settings,
        search_mode="vector", reranking=False,
    )
    assert calls["lexical_k"] is None  # la léxica NO se llama
    assert calls["rerank_top_k"] is None  # no rerank
    assert calls["vector_k"] == settings.rag_top_k  # sin rerank, recall = top_k
    assert [c.chunk_id for c in result.chunks] == [1, 2, 3]  # truncado a rag_top_k


async def test_hybrid_no_rerank_fuses_both(monkeypatch, settings):
    pipe, calls = _build_pipeline(monkeypatch, vector_ids=[1, 2, 3], lexical_ids=[4, 2, 5])
    result = await pipe.retrieve(
        reformulated=_reformulated(), settings=settings,
        search_mode="hybrid", reranking=False,
    )
    assert calls["vector_k"] == settings.rag_top_k
    assert calls["lexical_k"] == settings.rag_top_k
    assert calls["rerank_top_k"] is None
    # id=2 aparece en ambas ramas (consenso RRF) → primero.
    assert result.chunks[0].chunk_id == 2
    assert len(result.chunks) <= settings.rag_top_k


async def test_vector_with_rerank_uses_candidate_pool(monkeypatch, settings):
    pipe, calls = _build_pipeline(
        monkeypatch, vector_ids=[1, 2, 3, 4, 5, 6], lexical_ids=[9]
    )
    result = await pipe.retrieve(
        reformulated=_reformulated(), settings=settings,
        search_mode="vector", reranking=True,
    )
    assert calls["vector_k"] == settings.retrieval_candidate_pool_size  # recall amplio
    assert calls["rerank_top_k"] == settings.rag_top_k
    # El fake reranker invierte y recorta a top_k.
    assert [c.chunk_id for c in result.chunks] == [6, 5, 4]


async def test_hybrid_with_rerank(monkeypatch, settings):
    pipe, calls = _build_pipeline(
        monkeypatch, vector_ids=[1, 2, 3], lexical_ids=[4, 5, 6]
    )
    result = await pipe.retrieve(
        reformulated=_reformulated(), settings=settings,
        search_mode="hybrid", reranking=True,
    )
    assert calls["vector_k"] == settings.retrieval_candidate_pool_size
    assert calls["lexical_k"] == settings.retrieval_candidate_pool_size
    assert calls["rerank_top_k"] == settings.rag_top_k
    assert len(result.chunks) <= settings.rag_top_k
