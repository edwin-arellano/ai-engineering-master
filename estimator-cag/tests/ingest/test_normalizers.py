"""Normalizers: producen Document válidos y propagan metadatos del catálogo."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.ingest.documents import Document
from app.ingest.normalizers import canonical

from tests.ingest.conftest import make_source

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_normalize_budgets_propagates_pii_and_content():
    source = make_source(contains_pii=True)
    df = pd.DataFrame(
        [
            {
                "budget_id": "BUDGET-2024-0001",
                "client_name": "Acme",
                "total_amount": 80000.0,
                "currency": "EUR",
                "status": "signed",
                "signed_at": pd.Timestamp("2024-03-15", tz="UTC"),
            }
        ]
    )
    docs = canonical.normalize_budgets(df, source, _NOW)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "BUDGET-2024-0001" in docs[0].content
    assert docs[0].metadata.contains_pii is True
    assert docs[0].metadata.source_name == source.name
    assert docs[0].metadata.document_id == "BUDGET-2024-0001"


def test_normalize_transcript_turns_puts_speaker_in_extra():
    source = make_source(name="transcripts", fmt="txt")
    records = [
        {"timestamp": "00:00:01", "speaker": "Antonio", "text": "hola"},
        {"timestamp": None, "speaker": None, "text": "legacy"},
    ]
    docs = canonical.normalize_transcript_turns(records, source, _NOW, "meeting.txt")
    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta.document_id == "meeting.txt"
    assert meta.extra["has_speaker_tags"] is True
    assert meta.extra["turn_count"] == 2


def test_normalize_tabular_generic_renders_markdown():
    source = make_source(name="rates", fmt="xlsx")
    df = pd.DataFrame([{"role": "backend", "rate": 75}])
    docs = canonical.normalize_tabular_generic(df, source, _NOW, "rate.xlsx")
    assert len(docs) == 1
    assert "| role | rate |" in docs[0].content
