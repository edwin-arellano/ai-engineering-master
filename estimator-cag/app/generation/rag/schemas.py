"""Modelos del pipeline de embeddings. Esquema anidado de presupuesto (más rico
que los budgets planos del seed S06): cada Budget tiene client_metadata y una
lista de components. El chunker produce un Chunk por componente.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sector = Literal["finance", "ecommerce", "healthcare", "industrial", "other"]
Complexity = Literal["low", "medium", "high"]


class BudgetComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    name: str
    description: str
    tech_stack: list[str]
    estimated_hours: int = Field(ge=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class ClientMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sector: Sector
    country: str


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_id: str
    client_metadata: ClientMetadata
    project_summary: str
    main_technology: str
    year: int = Field(ge=2000, le=2100)
    total_estimated_hours: int = Field(ge=0)
    components: list[BudgetComponent]


class Chunk(BaseModel):
    """Fragmento listo para embeber. metadata son campos filtrables que NO se
    embeden pero viajan con el chunk para futuras consultas (S08 pgvector)."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int
    # huérfano: chunk demasiado pequeño (< CHUNK_ORPHAN_MIN_TOKENS). No se
    # vectoriza (no mete ruido en la futura BD vectorial); el comparador lo cuenta.
    is_orphan: bool = False


class EmbeddedChunk(Chunk):
    embedding: list[float]


class DocumentIngestRequest(BaseModel):
    """Contrato nuevo de /embeddings/ingest: un documento (un presupuesto) por llamada."""

    source_path: str = Field(min_length=1)
    document_type: str = "historical_budget"
    # JSON completo del presupuesto; se valida contra Budget en el endpoint.
    content: dict[str, Any]


class DocumentIngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)


class SearchResultItem(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    k: int
    search_time_ms: int
    results: list[SearchResultItem]
