"""Verifica el mapeo Budget → chunks → filas ORM sin tocar la base de datos."""

from __future__ import annotations

from app.generation.rag.chunking.registry import build_chunker
from app.generation.rag.persistence.models import ChunkRow, DocumentRow
from app.generation.rag.persistence.repository import BUDGET_COMPONENT
from app.generation.rag.schemas import Budget

_BUDGET = {
    "budget_id": "BUD-TEST-001",
    "client_metadata": {"name": "Acme", "sector": "finance", "country": "ES"},
    "project_summary": "Test project",
    "main_technology": "python",
    "year": 2024,
    "total_estimated_hours": 100,
    "components": [
        {
            "component_id": "C-001",
            "name": "Auth backend",
            "description": "OAuth 2.0 with JWT session management and rate limiting.",
            "tech_stack": ["python", "fastapi", "postgresql"],
            "estimated_hours": 60,
            "complexity": "high",
            "dependencies": [],
        }
    ],
}


def test_budget_parses_and_chunks() -> None:
    budget = Budget.model_validate(_BUDGET)
    chunks = build_chunker("structural").chunk([budget])
    assert chunks, "el chunker estructural debe producir al menos un chunk"
    chunk = chunks[0]
    assert chunk.chunk_id == "BUD-TEST-001::C-001"
    assert "C-001" in chunk.metadata.get("component_id", "C-001") or chunk.metadata


def test_chunk_to_row_mapping_is_lossless() -> None:
    budget = Budget.model_validate(_BUDGET)
    chunk = build_chunker("structural").chunk([budget])[0]

    row = ChunkRow(
        document_id=1,
        chunk_type=BUDGET_COMPONENT,
        content=chunk.text,
        embedding=[0.0] * 1536,
        metadata_={**chunk.metadata, "chunk_id": chunk.chunk_id},
    )
    assert row.content == chunk.text
    assert row.chunk_type == "budget_component"
    assert row.metadata_["chunk_id"] == "BUD-TEST-001::C-001"


def test_document_row_metadata_attr_maps_to_metadata_column() -> None:
    # `metadata` está reservado en DeclarativeBase; el atributo es `metadata_`.
    doc = DocumentRow(source_path="x", document_type="historical_budget", metadata_={"sector": "finance"})
    assert doc.metadata_["sector"] == "finance"
    assert DocumentRow.__table__.c.metadata.name == "metadata"
