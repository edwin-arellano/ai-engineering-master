"""Conversión de la representación intermedia al contrato Document, propagando
metadatos del catálogo. El normalizer es la capa fina que une parser y Document.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.ingest.catalog.models import CatalogSource
from app.ingest.documents import Document, DocumentMetadata


def _base_metadata(
    source: CatalogSource, ingested_at: datetime, document_id: str, **extra
) -> DocumentMetadata:
    return DocumentMetadata(
        source_name=source.name,
        source_location=source.location,
        ingested_at=ingested_at,
        source_version=str(source.refresh.observed_last_update),
        lineage_upstream=source.lineage.upstream,
        access_restrictions=source.sensitivity.access_restrictions,
        contains_pii=source.sensitivity.contains_pii,
        document_id=document_id,
        extra=extra,
    )


def normalize_budgets(
    df: pd.DataFrame, source: CatalogSource, ingested_at: datetime
) -> list[Document]:
    """Cada presupuesto válido -> un Document con content en markdown estructurado."""
    docs: list[Document] = []
    for _, row in df.iterrows():
        content = (
            f"# Presupuesto {row['budget_id']}\n\n"
            f"- Cliente: {row['client_name']}\n"
            f"- Importe total: {row['total_amount']} {row['currency']}\n"
            f"- Estado: {row['status']}\n"
            f"- Fecha de firma: {row['signed_at']}\n"
        )
        docs.append(
            Document(
                content=content,
                metadata=_base_metadata(source, ingested_at, str(row["budget_id"])),
            )
        )
    return docs


def normalize_transcript_turns(
    records: list[dict],
    source: CatalogSource,
    ingested_at: datetime,
    source_hint: str,
) -> list[Document]:
    """Un Document por transcripción (turnos concatenados); speaker/timestamp van en extra."""
    lines = [
        f"[{r['timestamp'] or '??:??:??'}] {r['speaker'] or 'UNKNOWN'}: {r['text']}"
        for r in records
    ]
    has_speakers = any(r["speaker"] for r in records)
    doc = Document(
        content="\n".join(lines),
        metadata=_base_metadata(
            source,
            ingested_at,
            document_id=source_hint,
            has_speaker_tags=has_speakers,
            turn_count=len(records),
        ),
    )
    return [doc]


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Renderiza un DataFrame como tabla markdown sin depender de `tabulate`.

    `pandas.DataFrame.to_markdown` exige el paquete opcional `tabulate`, que no
    es una dependencia del proyecto. Esta conversión cubre el caso tabular
    sencillo (rate card) sin añadir deps fuera del alcance de S06.
    """
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def normalize_tabular_generic(
    df: pd.DataFrame,
    source: CatalogSource,
    ingested_at: datetime,
    source_hint: str,
) -> list[Document]:
    """Fallback tabular (p. ej. xlsx) -> un Document en markdown de tabla."""
    return [
        Document(
            content=_df_to_markdown_table(df),
            metadata=_base_metadata(source, ingested_at, document_id=source_hint),
        )
    ]
