"""Orquestador: respeta la decisión del catálogo y produce Document[] en memoria."""

from __future__ import annotations

import json

from app.ingest.catalog.models import DataCatalog, IngestionDecision
from app.ingest.documents import Document
from app.ingest.orchestrator import run_ingestion

from tests.ingest.conftest import make_source


def _write_budget(path, budget_id, client_name, total, status="signed", signed="2024-03-15"):
    path.write_text(
        json.dumps(
            {
                "budget_id": budget_id,
                "client_name": client_name,
                "total_amount": total,
                "currency": "EUR",
                "signed_at": signed,
                "status": status,
            }
        ),
        encoding="utf-8",
    )


def test_run_ingestion_respects_catalog_decision(tmp_path):
    budgets_dir = tmp_path / "budgets"
    budgets_dir.mkdir()
    _write_budget(budgets_dir / "b1.json", "BUDGET-2024-0001", "Acme", 80000)
    _write_budget(budgets_dir / "b2.json", "BUDGET-2024-0004", "Initech", -50000)

    catalog = DataCatalog(
        version=1,
        last_audited="2024-06-01",
        sources=[
            make_source(
                name="budgets",
                location="budgets",
                fmt="json",
                decision=IngestionDecision.INCLUDE,
            ),
            make_source(
                name="transcripts",
                location="transcripts",
                fmt="txt",
                decision=IngestionDecision.REVIEW,
            ),
            make_source(
                name="rates",
                location="rates",
                fmt="xlsx",
                decision=IngestionDecision.EXCLUDE,
            ),
        ],
    )

    result = run_ingestion(catalog, project_root=tmp_path)

    # review/exclude nunca se ingestan
    assert set(result.rejected_sources) == {"transcripts", "rates"}
    # el presupuesto válido produce un Document; el negativo se descarta
    assert all(isinstance(d, Document) for d in result.documents)
    assert len(result.documents) == 1
    assert result.documents[0].metadata.document_id == "BUDGET-2024-0001"
    assert result.discarded_count == 1
    assert result.validation_reports["budgets"]["total"] == 2
