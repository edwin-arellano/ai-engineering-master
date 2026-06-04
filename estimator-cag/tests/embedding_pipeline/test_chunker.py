"""JSONStructuralChunker: un componente = un chunk, headers contextuales, metadata."""

from __future__ import annotations

from app.generation.rag.chunking.strategies.structural import JSONStructuralChunker
from app.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata


def _budget() -> Budget:
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
                description="OAuth 2.0 flows with JWT.",
                tech_stack=["ruby_on_rails", "redis"],
                estimated_hours=120,
                complexity="high",
            ),
            BudgetComponent(
                component_id="TXN-002",
                name="Ledger",
                description="Double-entry ledger.",
                tech_stack=["postgresql"],
                estimated_hours=90,
                complexity="medium",
            ),
        ],
    )


def test_one_chunk_per_component():
    chunks = JSONStructuralChunker().chunk([_budget()])
    assert len(chunks) == 2


def test_chunk_id_format():
    chunks = JSONStructuralChunker().chunk([_budget()])
    assert chunks[0].chunk_id == "BUD-2024-001::AUTH-001"
    assert chunks[1].chunk_id == "BUD-2024-001::TXN-002"


def test_text_contains_contextual_headers():
    chunk = JSONStructuralChunker().chunk([_budget()])[0]
    assert "[Client sector: finance" in chunk.text
    assert "Main tech: ruby_on_rails" in chunk.text
    assert "Component: OAuth backend" in chunk.text


def test_metadata_has_seven_filterable_keys_and_not_in_text():
    chunk = JSONStructuralChunker().chunk([_budget()])[0]
    assert set(chunk.metadata.keys()) == {
        "budget_id",
        "component_id",
        "client_sector",
        "main_technology",
        "year",
        "complexity",
        "estimated_hours",
    }
    # la metadata filtrable no se embebe (no aparece como claves en el texto)
    assert "client_sector" not in chunk.text
    assert "component_id" not in chunk.text


def test_token_count_positive_and_deterministic():
    chunks_a = JSONStructuralChunker().chunk([_budget()])
    chunks_b = JSONStructuralChunker().chunk([_budget()])
    assert chunks_a[0].token_count > 0
    assert chunks_a[0].token_count == chunks_b[0].token_count
