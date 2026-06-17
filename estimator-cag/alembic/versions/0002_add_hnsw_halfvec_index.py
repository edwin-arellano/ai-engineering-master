"""add hnsw half-vec index

Índice HNSW sobre embedding::halfvec(1536) con halfvec_cosine_ops. Half-vec (float16,
2 bytes/dim) reduce el índice a la mitad sin pérdida de recall en vectores normalizados
de OpenAI: la precisión extra de float32 es ruido que se paga en disco y RAM. m=16 y
ef_construction=128 son los valores estándar para 1536 dims y deben coincidir con
config.hnsw_m / config.hnsw_ef_construction.

NO se crea el índice float32: scripts/compare_index.py lo crea/dropea ad-hoc para el
demo de tamaño (235→117 MB), así no quedan dos índices vectoriales en el schema.

Construir el índice tarda ~1-2 min con ~30k vectores (instantáneo con el corpus real).
El operador `<=>` de search_chunks debe coincidir con halfvec_cosine_ops o el planner
cae a seq scan sin avisar (ver repository._build_halfvec_search_stmt y EXPLAIN ANALYZE).

CREATE INDEX CONCURRENTLY no cabe en la transacción de Alembic; para reconstrucciones
sin downtime en producción ver el runbook en persistence/monitoring.sql.

Revision ID: 0002
Revises: 0001
Create Date: session-08
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE INDEX chunks_embedding_halfvec_idx "
        "ON chunks USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 128)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS chunks_embedding_halfvec_idx")
