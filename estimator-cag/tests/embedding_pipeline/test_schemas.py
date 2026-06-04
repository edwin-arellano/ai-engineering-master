"""Schemas del pipeline de embeddings: parseo del sample y validaciones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.embedding_pipeline.schemas import Budget, Chunk, EmbeddedChunk

ROOT = Path(__file__).resolve().parents[2]


def test_sample_parses_with_budget_schema():
    data = json.loads((ROOT / "data" / "budgets_sample.json").read_text("utf-8"))
    budgets = [Budget(**b) for b in data]
    assert len(budgets) == 15
    assert all(len(b.components) >= 1 for b in budgets)


def test_sector_literal_rejects_unknown_value():
    data = json.loads((ROOT / "data" / "budgets_sample.json").read_text("utf-8"))
    bad = dict(data[0])
    bad["client_metadata"] = dict(bad["client_metadata"], sector="aerospace")
    with pytest.raises(ValidationError):
        Budget(**bad)


def test_embedded_chunk_requires_embedding():
    chunk = Chunk(chunk_id="x", text="t", metadata={}, token_count=1)
    with pytest.raises(ValidationError):
        EmbeddedChunk(**chunk.model_dump())  # falta embedding
    embedded = EmbeddedChunk(**chunk.model_dump(), embedding=[0.1, 0.2])
    assert embedded.embedding == [0.1, 0.2]
