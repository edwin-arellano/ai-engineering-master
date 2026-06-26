"""El statement de búsqueda debe compilar a SQL que use halfvec (alineado con el índice
HNSW chunks_embedding_halfvec_idx). Unit, sin DB: compila el statement al dialecto
postgres y comprueba la expresión y el operador. Si esto se rompe, search_chunks deja
de usar el índice (fallback silencioso a seq scan)."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from app.generation.rag.persistence.models import BudgetChunkRow
from app.generation.rag.persistence.repository import (
    _build_exact_search_stmt,
    _build_halfvec_search_stmt,
)


def test_search_stmt_emits_halfvec_cast() -> None:
    stmt = _build_halfvec_search_stmt(BudgetChunkRow, query_vector=[0.0] * 1536, k=5)
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    low = sql.lower()
    assert "halfvec" in low  # expresión indexada: embedding::halfvec(1536)
    assert "<=>" in sql  # operador coseno (alineado con halfvec_cosine_ops)
    assert "order by" in low
    # No debe proyectar la columna embedding (solo la distancia).
    assert "budget_chunks.embedding as embedding" not in low


def test_exact_stmt_is_plain_vector() -> None:
    # El ground truth NO castea a halfvec: usa la columna vector tal cual.
    stmt = _build_exact_search_stmt(BudgetChunkRow, query_vector=[0.0] * 1536, k=5)
    low = str(stmt.compile(dialect=postgresql.dialect())).lower()
    assert "halfvec" not in low
    assert "<=>" in str(stmt.compile(dialect=postgresql.dialect()))
