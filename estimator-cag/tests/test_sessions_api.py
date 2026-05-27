"""Tests de la API de sesiones.

Tres tests del ejercicio. Los que requieren llamadas reales al LLM están
marcados como ``@pytest.mark.integration`` y se excluyen con
``pytest -m "not integration"`` para no pagar tokens en CI ni en local sin
intención.

El test ``test_history_never_exceeds_max_turns`` NO está marcado como
integration porque inyecta turnos sintéticos directamente en el store, sin
llamar al LLM.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document as DocxDocument
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.sessions import get_session_store


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    document = DocxDocument()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture()
def asgi_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.integration
async def test_two_turns_update_project_metadata(asgi_client: AsyncClient) -> None:
    """Dos turnos enlazados muestran coherencia conversacional."""
    async with asgi_client as client:
        session_response = await client.post("/api/v1/sessions")
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        first = await client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={
                "transcript": (
                    "Mobile app named BookFlow for booking medical appointments "
                    "on iOS and Android, with login and push notifications."
                ),
                "project_type": "mobile_app",
                "detail_level": "medium",
                "output_format": "phases_table",
            },
        )
        assert first.status_code == 200
        assert first.json()["result"]["phases"]

        second = await client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={
                "transcript": (
                    "Add a single EMR integration; reduce push notifications "
                    "to appointment reminders only."
                ),
                "project_type": "mobile_app",
                "detail_level": "medium",
                "output_format": "phases_table",
            },
        )
        assert second.status_code == 200
        # El segundo summary debe referirse al mismo proyecto (BookFlow) sin
        # que el usuario lo haya repetido — coherencia conversacional.
        second_summary = second.json()["result"]["summary"].lower()
        assert "bookflow" in second_summary or "appointments" in second_summary


@pytest.mark.integration
async def test_attachment_influences_estimation(asgi_client: AsyncClient) -> None:
    """Un adjunto con detalles técnicos debe alterar la estimación."""
    async with asgi_client as client:
        session_response = await client.post("/api/v1/sessions")
        session_id = session_response.json()["session_id"]

        docx_bytes = _make_docx_bytes(
            [
                "Project specification for an internal compliance dashboard.",
                "The project must integrate with SAP and includes a 3-month "
                "GDPR audit phase before launch.",
            ]
        )

        response = await client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data={
                "transcript": "Internal compliance dashboard for a fintech team.",
                "project_type": "internal_tool",
                "detail_level": "detailed",
                "output_format": "phases_table",
            },
            files={
                "attachments": (
                    "spec.docx",
                    docx_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert response.status_code == 200
        body = response.json()
        summary = body["result"]["summary"].lower()
        # Si el adjunto se respetó, el summary debería mencionar SAP, GDPR o
        # el audit phase derivado del documento.
        assert "sap" in summary or "gdpr" in summary or "audit" in summary


async def test_history_never_exceeds_max_turns(asgi_client: AsyncClient) -> None:
    """Tras 8 turnos sintéticos, el historial conserva max_turns=6 pares.

    No hace llamadas reales al LLM: inyecta directamente en el store para
    verificar la ventana deslizante.
    """
    async with asgi_client as client:
        session_response = await client.post("/api/v1/sessions")
        assert session_response.status_code == 201
        session_id = session_response.json()["session_id"]

        store = get_session_store()
        session = store.get(session_id)
        for i in range(8):
            session.history.append_turn(
                user_content=f"user {i}",
                assistant_content=f"assistant {i}",
            )
            session.history._truncate(max_turns=6)
        store.save(session)

        fresh = store.get(session_id)
        assert len(fresh.history.messages) == 12  # 6 pares × 2
