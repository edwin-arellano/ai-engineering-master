"""Utilidades de invocación del grafo y derivación del DSN del checkpointer."""

from __future__ import annotations

from app.domain.structured_estimation import StructuredEstimate


def checkpointer_conninfo(database_url: str) -> str:
    """AsyncPostgresSaver usa psycopg, no asyncpg: deriva un DSN psycopg desde el DSN
    async de SQLAlchemy. `postgresql+asyncpg://…` → `postgresql://…`."""
    return database_url.replace("+asyncpg", "")


async def run_estimation_graph(
    graph, *, transcript: str, thread_id: str
) -> tuple[StructuredEstimate, str]:
    """Invoca el grafo y devuelve (estimate, status). El thread_id (= id de estimación)
    persiste el checkpoint de esta ejecución en el Postgres del proyecto."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(
        {"transcript": transcript, "task_estimates": [], "errors": []}, config
    )
    return result["estimate"], result["status"]
