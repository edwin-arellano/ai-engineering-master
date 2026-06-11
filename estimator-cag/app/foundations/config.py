"""Configuración global del servicio cargada desde variables de entorno."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings del servicio validados al arrancar la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LiteLLM configuration ===
    primary_model: str = "anthropic/claude-haiku-4-5-20251001"
    fallback_model: str = "openai/gpt-4o-mini"
    llm_timeout_seconds: int = 30
    llm_num_retries: int = 2
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4000

    # === API keys ===
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # === Redis exact-match cache (S03) ===
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400
    cache_enabled: bool = True

    # === Embeddings y cache semántico (S04) ===
    embeddings_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDINGS_MODEL"
    )
    embeddings_dimensions: int = Field(default=1536, alias="EMBEDDINGS_DIMENSIONS")
    # Tamaño de batch del pipeline de embeddings (S07).
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    semantic_cache_enabled: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")
    semantic_cache_threshold: float = Field(
        default=0.92, alias="SEMANTIC_CACHE_THRESHOLD"
    )
    semantic_cache_ttl_seconds: int = Field(
        default=86400, alias="SEMANTIC_CACHE_TTL_SECONDS"
    )
    semantic_cache_name: str = Field(
        default="estimator_semantic_cache", alias="SEMANTIC_CACHE_NAME"
    )

    # === Guardrails (S04) ===
    moderation_enabled: bool = Field(default=True, alias="MODERATION_ENABLED")
    min_confidence_pct: int = Field(default=30, alias="MIN_CONFIDENCE_PCT")

    # === Versionado de prompts (S04 → v2; pre-S05 → v3) ===
    prompt_version: str = Field(default="v3", alias="PROMPT_VERSION")

    # === Sesiones conversacionales (pre-S05) ===
    max_turns: int = Field(
        default=6,
        alias="MAX_TURNS",
        description="Pares user+assistant máximos que sobreviven a la ventana deslizante.",
    )
    session_idle_ttl_seconds: int = Field(
        default=86400,
        alias="SESSION_IDLE_TTL_SECONDS",
        description="Tiempo de inactividad tras el cual una sesión se purga.",
    )
    attachment_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        alias="ATTACHMENT_MAX_BYTES",
        description="Tamaño máximo aceptado por adjunto (bytes).",
    )

    # ---- Tiers (nuevo en S05) ----
    default_tier: str = Field(default="developer", alias="DEFAULT_TIER")

    # ---- Compresión de historial (nuevo en S05) ----
    compression_policy: str = Field(
        default="anchor_hybrid",
        alias="COMPRESSION_POLICY",
        description="anchor_hybrid | sliding_window | cumulative",
    )
    compression_trigger_turns: int = Field(
        default=6,
        alias="COMPRESSION_TRIGGER_TURNS",
        description="Número de pares user+assistant a partir del cual se comprime.",
    )
    compression_keep_recent_turns: int = Field(
        default=3,
        alias="COMPRESSION_KEEP_RECENT_TURNS",
        description="Pares recientes que se mantienen sin comprimir tras una compresión.",
    )

    # ---- Actor-Critic-Boss (nuevo en S05) ----
    acb_max_iterations: int = Field(
        default=3,
        alias="ACB_MAX_ITERATIONS",
        description="Presupuesto máximo de iteraciones del boss.",
    )

    # ---- Versiones de prompt auxiliares (nuevo en S05) ----
    critic_prompt_version: str = Field(default="v1", alias="CRITIC_PROMPT_VERSION")
    boss_prompt_version: str = Field(default="v1", alias="BOSS_PROMPT_VERSION")
    summarizer_prompt_version: str = Field(
        default="v1", alias="SUMMARIZER_PROMPT_VERSION"
    )

    # ---- Ingesta (S06) ----
    ingest_seed_dir: str = Field("data/seed", alias="INGEST_SEED_DIR")
    data_catalog_path: str = Field(
        "data/catalog/data_catalog.yaml", alias="DATA_CATALOG_PATH"
    )
    catalog_evaluator_prompt_version: str = Field(
        "v1", alias="CATALOG_EVALUATOR_PROMPT_VERSION"
    )

    # ---- Viabilidad arquitectónica CAG (S06) — umbrales del marco de decisión ----
    cag_usable_context_ratio: float = Field(0.7, alias="CAG_USABLE_CONTEXT_RATIO")
    cag_latency_sla_seconds: float = Field(4.0, alias="CAG_LATENCY_SLA_SECONDS")
    cag_cost_per_turn_budget_usd: float = Field(
        0.05, alias="CAG_COST_PER_TURN_BUDGET_USD"
    )

    # ---- RAG chunking (S07) ----
    chunk_max_tokens: int = Field(512, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(80, alias="CHUNK_OVERLAP_TOKENS")
    chunk_orphan_min_tokens: int = Field(20, alias="CHUNK_ORPHAN_MIN_TOKENS")
    semantic_breakpoint_threshold: float = Field(
        0.55, alias="SEMANTIC_BREAKPOINT_THRESHOLD"
    )
    propositional_prompt_version: str = Field(
        "v1", alias="PROPOSITIONAL_PROMPT_VERSION"
    )
    contextual_retrieval_prompt_version: str = Field(
        "v1", alias="CONTEXTUAL_RETRIEVAL_PROMPT_VERSION"
    )

    # ---- Persistencia pgvector (pre-S08) ----
    database_url: str = Field(
        default="postgresql+asyncpg://estimator:estimator@localhost:5433/estimator",
        alias="DATABASE_URL",
        description=(
            "DSN async (asyncpg) de Postgres. En local apunta a localhost:5433 "
            "(host remapeado para no chocar con el Postgres de Herd en 5432). "
            "En compose se sobreescribe a host 'postgres':5432 (red interna)."
        ),
    )

    # ---- RAG índice vectorial (S08) ----
    # Parámetros HNSW. m y ef_construction son build-time (deben coincidir con la
    # migración 0002, que los hardcodea). ef_search es query-time: el punto dulce
    # recall/latencia que mide tune_ef_search.py y aplica search_chunks con SET LOCAL.
    hnsw_m: int = Field(default=16, alias="HNSW_M")
    hnsw_ef_construction: int = Field(default=128, alias="HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(default=40, alias="HNSW_EF_SEARCH")

    # === Streamlit / runtime ===
    backend_url: str = "http://localhost:8000"
    environment: str = "development"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def at_least_one_api_key(self) -> "Settings":
        """Verifica que al menos una API key esté configurada."""
        if not self.anthropic_api_key and not self.openai_api_key:
            raise ValueError(
                "Debe configurarse al menos una API key: "
                "ANTHROPIC_API_KEY o OPENAI_API_KEY."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia cacheada de Settings (singleton funcional)."""
    return Settings()
