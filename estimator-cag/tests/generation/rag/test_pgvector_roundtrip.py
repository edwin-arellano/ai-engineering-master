"""Round-trip end-to-end contra el servicio HTTP real: ingesta un documento y
comprueba que /search lo recupera. Marcado integration: necesita Postgres (migrado),
una API key de embeddings y el servidor levantado en localhost:8000.
"""

from __future__ import annotations

import httpx
import pytest

BACKEND = "http://localhost:8000"

_BUDGET = {
    "budget_id": "BUD-IT-001",
    "client_metadata": {"name": "Roundtrip", "sector": "finance", "country": "ES"},
    "project_summary": "OAuth 2.0 authentication backend for fintech",
    "main_technology": "python",
    "year": 2024,
    "total_estimated_hours": 120,
    "components": [
        {
            "component_id": "AUTH-001",
            "name": "OAuth 2.0 authentication backend",
            "description": "OAuth 2.0 flows with JWT session management for fintech.",
            "tech_stack": ["python", "fastapi"],
            "estimated_hours": 120,
            "complexity": "high",
            "dependencies": [],
        }
    ],
}


@pytest.mark.integration
def test_ingest_then_search_roundtrip() -> None:
    try:
        with httpx.Client(base_url=BACKEND, timeout=120.0) as client:
            ingest = client.post(
                "/embeddings/ingest",
                json={
                    "source_path": "it/BUD-IT-001",
                    "document_type": "historical_budget",
                    "content": _BUDGET,
                },
            )
            assert ingest.status_code in (200, 409)

            # idempotencia: segunda ingesta del mismo source_path → 409
            dup = client.post(
                "/embeddings/ingest",
                json={
                    "source_path": "it/BUD-IT-001",
                    "document_type": "historical_budget",
                    "content": _BUDGET,
                },
            )
            assert dup.status_code == 409
            assert "document_id" in dup.json()

            search = client.post(
                "/search",
                json={"query": "OAuth authentication for fintech", "k": 5},
            )
            assert search.status_code == 200
            body = search.json()
            assert body["results"], "la búsqueda debe devolver al menos un chunk"
            # resultados ordenados por distancia ascendente
            distances = [r["distance"] for r in body["results"]]
            assert distances == sorted(distances)
    except httpx.ConnectError:
        pytest.skip("servidor no disponible en localhost:8000")
