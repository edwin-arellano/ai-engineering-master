"""add fulltext tsvector column and GIN index (spanish config)

Columna generada `content_tsv` (tsvector, config 'spanish') mantenida por PostgreSQL
en sincronía con `content` en cada escritura, sin triggers. Índice GIN invertido para
la rama léxica del retrieval híbrido (S10). La config 'spanish' aplica stemming y
stopwords del español; los tecnicismos en inglés (OAuth, PSD2, Stripe) pasan casi
intactos y funcionan como identificadores exactos.

No requiere reembedear ni re-ingestar: la columna es STORED generada y se rellena
sola para las filas existentes.

Revision ID: 0003
Revises: 0002
Create Date: pre-session-10
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
        """
    )
    op.execute(
        "CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
