"""Fase 3 (augmentation). Determinista, sin LLM. Verifica el token budget: con un
max_tokens pequeño hay chunks descartados (dropped > 0) y los included_refs son un
subconjunto de los chunk_ref de entrada."""

from __future__ import annotations

from app.generation.rag.retrieval.augmentation import assemble_context
from app.generation.rag.schemas import (
    MetadataFilters,
    ReformulatedQuery,
    RetrievalResult,
    RetrievedChunk,
)


def _chunk(i: int, *, ref: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        document_id=1,
        chunk_type="budget_component",
        content=content,
        distance=0.1 * i,
        metadata={"chunk_id": ref},
    )


def _retrieval(chunks: list[RetrievedChunk]) -> RetrievalResult:
    reformulated = ReformulatedQuery(
        project_function="x",
        sector="finance",
        scale="medium",
        search_text="x",
    )
    return RetrievalResult(
        reformulated=reformulated,
        filters=MetadataFilters(),
        top_k=25,
        distance_threshold=0.6,
        chunks=chunks,
        search_time_ms=1,
    )


def test_all_chunks_fit_under_large_budget():
    chunks = [_chunk(i, ref=f"BUD::C{i}", content="word " * 20) for i in range(3)]
    ctx = assemble_context(_retrieval(chunks), max_tokens=100_000)
    assert ctx.dropped == 0
    assert ctx.included_refs == ["BUD::C0", "BUD::C1", "BUD::C2"]
    assert ctx.token_count > 0
    assert "source_id: BUD::C0" in ctx.context_block


def test_small_budget_drops_chunks_and_keeps_subset():
    input_refs = [f"BUD::C{i}" for i in range(5)]
    chunks = [_chunk(i, ref=input_refs[i], content="word " * 50) for i in range(5)]
    # Presupuesto pequeño: solo entran los primeros bloques.
    ctx = assemble_context(_retrieval(chunks), max_tokens=80)

    assert ctx.dropped > 0
    assert ctx.token_count <= 80
    assert set(ctx.included_refs).issubset(set(input_refs))
    assert len(ctx.included_refs) + ctx.dropped == len(chunks)


def test_empty_retrieval_yields_empty_context():
    ctx = assemble_context(_retrieval([]), max_tokens=1000)
    assert ctx.context_block == ""
    assert ctx.included_refs == []
    assert ctx.dropped == 0
    assert ctx.token_count == 0
