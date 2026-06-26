"""Modelos ORM de la persistencia vectorial. Tres colecciones particionadas
(theory Opción B): cada familia tiene su tabla, su índice HNSW half-vec y su GIN
tsvector, y su ciclo de vida de ingesta independiente. Las columnas estructurales
son comunes (mixin); la divergencia de esquema vive en las CONVENCIONES del JSONB
`metadata` y en los índices/ciclo de vida por tabla.

El atributo Python se llama `metadata_` porque `metadata` está reservado por
SQLAlchemy DeclarativeBase; se mapea a la columna real `metadata`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Dimensionalidad de text-embedding-3-small. Hardcodeada a propósito: cambiarla
# implica reembedear todo el corpus, así que no es una decisión dinámica.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # metadata estable del documento (sector, año, tecnología, summary...) en JSONB.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default="{}", nullable=False
    )


class ChunkColumns:
    """Mixin con las columnas estructurales compartidas por las 3 colecciones.
    No es una tabla: cada subclase concreta fija su __tablename__. La FK a
    documents y el contrato de columnas (content/embedding/metadata/tsvector)
    son idénticos; lo que diverge entre colecciones son las CONVENCIONES del
    JSONB `metadata`, los índices y el ciclo de vida de ingesta.
    """

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Sub-discriminador dentro de la colección (budgets: budget_component|historical_task;
    # transcripts: transcript_segment; technical: technical_reference).
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # nullable: permite insertar y rellenar el embedding después (ingesta async futura).
    # En este ejercicio se ingesta chunk+embedding atómicamente.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default="{}", nullable=False
    )
    # Columna generada: tsvector 'spanish' derivado de content para la rama léxica del
    # retrieval híbrido. Read-only: PostgreSQL la mantiene (GENERATED ALWAYS ... STORED,
    # ver migraciones 0003/0004); SQLAlchemy nunca intenta escribirla.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('spanish', content)", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class BudgetChunkRow(ChunkColumns, Base):
    """Colección de presupuestos. Sub-eje overview/detalle vía chunk_type
    (budget_component | historical_task)."""

    __tablename__ = "budget_chunks"


class TranscriptChunkRow(ChunkColumns, Base):
    """Colección de transcripciones de reunión (chunk_type=transcript_segment)."""

    __tablename__ = "transcript_chunks"


class TechnicalDocChunkRow(ChunkColumns, Base):
    """Colección de documentación técnica de referencia (chunk_type=technical_reference)."""

    __tablename__ = "technical_doc_chunks"


# Back-compat: el código previo importa ChunkRow (= la colección de budgets).
ChunkRow = BudgetChunkRow
