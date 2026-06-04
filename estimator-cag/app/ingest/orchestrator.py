"""Orquestador de ingesta: pega catálogo + loaders + parsers + cleaning +
normalizers y produce Document[]. Respeta la decisión del catálogo (solo INCLUDE).
NO hace chunking ni persistencia: los Document quedan en memoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog

from app.ingest.catalog.models import CatalogSource, DataCatalog, IngestionDecision
from app.ingest.cleaning.budgets import clean_budget_records
from app.ingest.cleaning.policy import validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord
from app.ingest.documents import Document
from app.ingest.loaders.filesystem import FilesystemLoader
from app.ingest.normalizers import canonical
from app.ingest.parsers.json_parser import JsonBudgetParser
from app.ingest.parsers.txt_parser import TxtTranscriptParser
from app.ingest.parsers.xlsx_parser import XlsxParser

logger = structlog.get_logger(__name__)

_PARSERS = {
    "json": JsonBudgetParser(),
    "txt": TxtTranscriptParser(),
    "xlsx": XlsxParser(),
}


@dataclass
class IngestionResult:
    documents: list[Document] = field(default_factory=list)
    rejected_sources: list[str] = field(default_factory=list)
    validation_reports: dict = field(default_factory=dict)
    quarantined_count: int = 0
    discarded_count: int = 0


def _ingest_source(
    source: CatalogSource,
    loader: FilesystemLoader,
    ingested_at: datetime,
    result: IngestionResult,
) -> None:
    parser = _PARSERS.get(source.format)
    if parser is None:
        raise ValueError(f"No hay parser registrado para el formato: {source.format}")

    for ref in loader.list_files(source.location):
        parsed = parser.parse(loader.read(ref), source_hint=ref.path)

        if source.format == "json":  # presupuestos: limpieza + validación Pandera
            cleaned = clean_budget_records(parsed.dataframe)
            vr = validate_with_policy(cleaned, BudgetRecord)
            result.validation_reports[source.name] = vr.report
            result.quarantined_count += len(vr.quarantined)
            result.discarded_count += len(vr.discarded)
            result.documents += canonical.normalize_budgets(
                vr.valid, source, ingested_at
            )
        elif source.format == "txt":
            result.documents += canonical.normalize_transcript_turns(
                parsed.records, source, ingested_at, ref.path
            )
        else:  # xlsx u otros tabulares
            result.documents += canonical.normalize_tabular_generic(
                parsed.dataframe, source, ingested_at, ref.path
            )


def run_ingestion(catalog: DataCatalog, *, project_root: Path) -> IngestionResult:
    loader = FilesystemLoader(project_root)
    ingested_at = datetime.now(timezone.utc)
    result = IngestionResult()

    for source in catalog.sources:
        if source.decision != IngestionDecision.INCLUDE:
            result.rejected_sources.append(source.name)
            logger.info("ingest.skip", source=source.name, decision=source.decision.value)
            continue
        _ingest_source(source, loader, ingested_at, result)

    logger.info(
        "ingest.completed",
        documents=len(result.documents),
        rejected=len(result.rejected_sources),
        quarantined=result.quarantined_count,
        discarded=result.discarded_count,
    )
    return result
