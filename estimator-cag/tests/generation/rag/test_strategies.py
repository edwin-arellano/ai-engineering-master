"""Estrategias mecánicas de chunking (sin API)."""

from __future__ import annotations

from app.foundations.config import get_settings
from app.generation.rag.chunking.strategies.fixed_size import FixedSizeChunker
from app.generation.rag.chunking.strategies.hierarchical import HierarchicalChunker
from app.generation.rag.chunking.strategies.recursive import RecursiveChunker
from app.generation.rag.chunking.strategies.sentence_window import SentenceWindowChunker
from app.generation.rag.chunking.strategies.structural import StructuralChunker
from app.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata


def _budget(*, multi_sentence: bool = False) -> Budget:
    desc = (
        "Handles login. Validates tokens. Refreshes sessions. Logs events. "
        "Rotates keys. Audits access."
        if multi_sentence
        else "OAuth 2.0 flows with JWT session management and rate limiting."
    )
    return Budget(
        budget_id="BUD-2024-001",
        client_metadata=ClientMetadata(name="Acme", sector="finance", country="ES"),
        project_summary="Mobile banking API",
        main_technology="ruby_on_rails",
        year=2024,
        total_estimated_hours=210,
        components=[
            BudgetComponent(
                component_id="AUTH-001",
                name="OAuth backend",
                description=desc,
                tech_stack=["ruby_on_rails", "redis"],
                estimated_hours=120,
                complexity="high",
            ),
            BudgetComponent(
                component_id="TXN-002",
                name="Ledger",
                description="Double-entry ledger with reconciliation.",
                tech_stack=["postgresql"],
                estimated_hours=90,
                complexity="medium",
            ),
        ],
    )


def test_structural_one_chunk_per_component_with_chunk_id():
    chunks = StructuralChunker().chunk([_budget()])
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "BUD-2024-001::AUTH-001"
    assert chunks[1].chunk_id == "BUD-2024-001::TXN-002"
    assert "[Client sector: finance" in chunks[0].text


def test_recursive_and_fixed_respect_max_tokens():
    max_tokens = get_settings().chunk_max_tokens
    for chunker in (RecursiveChunker(), FixedSizeChunker()):
        chunks = chunker.chunk([_budget()])
        assert chunks, chunker.name
        assert all(c.token_count <= max_tokens for c in chunks), chunker.name


def test_sentence_window_more_chunks_and_orphans_than_structural():
    budget = _budget(multi_sentence=True)
    structural = StructuralChunker().chunk([budget])
    sentence = SentenceWindowChunker().chunk([budget])
    assert len(sentence) > len(structural)
    structural_orphans = sum(c.is_orphan for c in structural)
    sentence_orphans = sum(c.is_orphan for c in sentence)
    assert sentence_orphans > structural_orphans


def test_hierarchical_one_parent_per_budget_and_child_per_component():
    chunks = HierarchicalChunker().chunk([_budget()])
    parents = [c for c in chunks if c.metadata.get("level") == "parent"]
    children = [c for c in chunks if c.metadata.get("level") == "child"]
    assert len(parents) == 1
    assert len(children) == 2
    assert all(c.metadata["parent_chunk_id"] == parents[0].chunk_id for c in children)


def test_is_orphan_matches_threshold():
    threshold = get_settings().chunk_orphan_min_tokens
    chunks = SentenceWindowChunker().chunk([_budget(multi_sentence=True)])
    for c in chunks:
        assert c.is_orphan == (c.token_count < threshold)
