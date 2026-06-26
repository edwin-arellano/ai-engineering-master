"""Modelos del pipeline de embeddings. Esquema anidado de presupuesto (más rico
que los budgets planos del seed S06): cada Budget tiene client_metadata y una
lista de components. El chunker produce un Chunk por componente.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Sector = Literal["finance", "ecommerce", "healthcare", "industrial", "other"]
Complexity = Literal["low", "medium", "high"]


class SearchTarget(StrEnum):
    """Colección particionada del corpus (S10). Cada valor mapea a una tabla ORM
    vía COLLECTION_MODELS (persistence/collections.py). El router en cascada decide
    contra cuál(es) buscar; el flujo por-tarea apunta siempre a BUDGETS (explícito)."""

    BUDGETS = "budgets"
    TRANSCRIPTS = "transcripts"
    TECHNICAL_DOCS = "technical_docs"


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
    # Estrategia de chunking a aplicar (S09). Default back-compat: "structural"
    # (un chunk por componente → chunk_type=budget_component). "historical_task"
    # produce un chunk por tarea atómica → chunk_type=historical_task.
    chunk_strategy: str = "structural"


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


# ---------------------------------------------------------------------------
# S09 — Pipeline RAG end-to-end (reformulación → retrieval → augmentation)
# ---------------------------------------------------------------------------


class MetadataFilters(BaseModel):
    """Filtros SQL deterministas sobre la metadata de los chunks. Todos opcionales:
    None = no filtrar por ese eje. NO se delegan a prompting (son 100% deterministas)."""

    model_config = ConfigDict(extra="forbid")

    sectors: list[Sector] | None = None
    year_min: int | None = Field(default=None, ge=2000, le=2100)
    year_max: int | None = Field(default=None, ge=2000, le=2100)
    main_technology: str | None = None
    chunk_types: list[str] | None = None  # p.ej. ["budget_component", "historical_task"]


class ReformulatedQuery(BaseModel):
    """Salida tipada de la fase de reformulación: brief estructurado + texto de
    búsqueda denso. Los campos del brief alimentan los filtros de metadata; el
    search_text es lo único que se embede para el retrieval (single-query S09)."""

    model_config = ConfigDict(extra="forbid")

    project_function: str = Field(..., min_length=1, description="Qué hace el proyecto, en una frase.")
    technologies: list[str] = Field(default_factory=list)
    sector: Sector
    scale: str = Field(..., description="Escala estimada: small | medium | large.")
    countries: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    search_text: str = Field(..., min_length=1, description="Texto corto y denso que se embede para buscar.")


class RetrievedChunk(BaseModel):
    """Un chunk recuperado, con trazabilidad. `chunk_ref` es el chunk_id de negocio
    (BUD-...::AUTH-...) guardado en metadata; se usa para las citations."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]

    @property
    def chunk_ref(self) -> str:
        return str(self.metadata.get("chunk_id", f"db:{self.chunk_id}"))


class RetrievalResult(BaseModel):
    reformulated: ReformulatedQuery
    filters: MetadataFilters
    top_k: int
    distance_threshold: float
    chunks: list[RetrievedChunk]
    search_time_ms: int


class AugmentedContext(BaseModel):
    """Contexto ensamblado a partir de los chunks, acotado por token budget.
    `included_refs` son los chunk_ref efectivamente incluidos (los que el modelo
    puede citar); `dropped` los que no cupieron en el presupuesto de tokens."""

    context_block: str
    token_count: int
    included_refs: list[str]
    dropped: int
