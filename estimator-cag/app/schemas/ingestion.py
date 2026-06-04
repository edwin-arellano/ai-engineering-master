from __future__ import annotations

from pydantic import BaseModel


class SourceValidationReport(BaseModel):
    total: int
    valid: int
    quarantined: int = 0
    discarded: int = 0


class IngestionRunResponse(BaseModel):
    documents_produced: int
    rejected_sources: list[str]
    quarantined_count: int
    discarded_count: int
    validation_reports: dict[str, dict]


class CatalogResponse(BaseModel):
    version: int
    last_audited: str
    included: list[str]
    review: list[str]
    excluded: list[str]
