"""Modelos tipados del catálogo de datos. El catálogo es código, no documentación:
se versiona en git, se valida al cargar y el pipeline itera sobre included_sources().
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IngestionDecision(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    REVIEW = "review"


class Volume(BaseModel):
    records: int
    size_mb: float


class Refresh(BaseModel):
    declared: str
    observed_last_update: str
    observed_lag_days: int


class Quality(BaseModel):
    completeness: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    actuality: int = Field(ge=1, le=5)
    reliability: int = Field(ge=1, le=5)

    @property
    def is_rag_ready(self) -> bool:
        """RAG-ready solo si NINGUNA dimensión cae por debajo de aceptable (3).

        Las dimensiones no se compensan entre sí: una fuente con completeness=5
        y reliability=1 no es "calidad 3", es una fuente cuyos datos pueden ser
        mentira. Cada dimensión es condición necesaria.
        """
        return all(
            s >= 3
            for s in (
                self.completeness,
                self.consistency,
                self.actuality,
                self.reliability,
            )
        )


class Sensitivity(BaseModel):
    contains_pii: bool
    pii_types: list[str] = []
    access_restrictions: Optional[str] = None


class Lineage(BaseModel):
    upstream: str
    transformations: list[str] = []


class CatalogSource(BaseModel):
    name: str
    description: str
    location: str  # relativo a la raíz del proyecto
    owner_technical: str
    owner_business: str
    format: str
    volume: Volume
    refresh: Refresh
    quality: Quality
    sensitivity: Sensitivity
    lineage: Lineage
    decision: IngestionDecision
    decision_reason: Optional[str] = None
    notes: Optional[str] = None


class DataCatalog(BaseModel):
    version: int
    last_audited: str
    sources: list[CatalogSource]

    def included_sources(self) -> list[CatalogSource]:
        return [s for s in self.sources if s.decision == IngestionDecision.INCLUDE]
