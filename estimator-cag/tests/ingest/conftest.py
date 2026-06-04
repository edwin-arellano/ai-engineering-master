"""Fixtures compartidas para los tests de ingesta."""

from __future__ import annotations

import pytest

from app.ingest.catalog.models import (
    CatalogSource,
    IngestionDecision,
    Lineage,
    Quality,
    Refresh,
    Sensitivity,
    Volume,
)


def make_source(
    *,
    name: str = "budgets",
    location: str = "data/seed/budgets",
    fmt: str = "json",
    decision: IngestionDecision = IngestionDecision.INCLUDE,
    contains_pii: bool = True,
) -> CatalogSource:
    """Construye un CatalogSource mínimo y válido para los tests."""
    return CatalogSource(
        name=name,
        description="fuente de prueba",
        location=location,
        owner_technical="tech@test",
        owner_business="business",
        format=fmt,
        volume=Volume(records=1, size_mb=0.01),
        refresh=Refresh(
            declared="on_close",
            observed_last_update="2024-06-01",
            observed_lag_days=1,
        ),
        quality=Quality(completeness=4, consistency=4, actuality=4, reliability=4),
        sensitivity=Sensitivity(
            contains_pii=contains_pii,
            pii_types=["client_name"] if contains_pii else [],
            access_restrictions="interno" if contains_pii else None,
        ),
        lineage=Lineage(upstream="CRM", transformations=["export"]),
        decision=decision,
    )


@pytest.fixture
def source_factory():
    return make_source
