"""Modelos del catálogo: calidad RAG-ready e included_sources()."""

from __future__ import annotations

from app.ingest.catalog.models import DataCatalog, IngestionDecision, Quality

from tests.ingest.conftest import make_source


def test_quality_rag_ready_requires_all_dimensions():
    assert Quality(
        completeness=3, consistency=3, actuality=3, reliability=3
    ).is_rag_ready
    # una sola dimensión por debajo de 3 lo invalida (no se compensan)
    assert not Quality(
        completeness=5, consistency=5, actuality=2, reliability=5
    ).is_rag_ready


def test_included_sources_filters_by_decision():
    catalog = DataCatalog(
        version=1,
        last_audited="2024-06-01",
        sources=[
            make_source(name="budgets", decision=IngestionDecision.INCLUDE),
            make_source(name="transcripts", decision=IngestionDecision.REVIEW),
            make_source(name="rates", decision=IngestionDecision.EXCLUDE),
        ],
    )
    assert [s.name for s in catalog.included_sources()] == ["budgets"]
