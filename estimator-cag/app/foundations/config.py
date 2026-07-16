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

    # === Reformulator (modelo barato para la fase de reformulación, S09) ===
    reformulator_primary_model: str = Field(
        default="openai/gpt-4o-mini", alias="REFORMULATOR_PRIMARY_MODEL"
    )
    reformulator_fallback_model: str = Field(
        default="anthropic/claude-haiku-4-5-20251001", alias="REFORMULATOR_FALLBACK_MODEL"
    )
    reformulator_temperature: float = Field(default=0.0, alias="REFORMULATOR_TEMPERATURE")
    reformulator_max_tokens: int = Field(default=1200, alias="REFORMULATOR_MAX_TOKENS")

    # === Retrieval RAG (S09) ===
    rag_top_k: int = Field(
        default=25, alias="RAG_TOP_K",
        description="top_k de la estimación en una pasada (alto a propósito; ver notas).",
    )
    rag_distance_threshold: float = Field(default=0.6, alias="RAG_DISTANCE_THRESHOLD")
    rag_max_context_tokens: int = Field(default=16384, alias="RAG_MAX_CONTEXT_TOKENS")
    # Generación RAG-grounded: razonamiento "alto" modelado con temperatura baja + más tokens.
    rag_generation_temperature: float = Field(default=0.2, alias="RAG_GENERATION_TEMPERATURE")
    rag_generation_max_tokens: int = Field(default=8000, alias="RAG_GENERATION_MAX_TOKENS")
    # v2: atribución por línea con document_id + evidence literal (S11). v1 queda para rollback/A-B.
    rag_estimation_prompt_version: str = Field(default="v2", alias="RAG_ESTIMATION_PROMPT_VERSION")
    # Política ante citas colgantes: False → detectar+reportar+loguear; True → 422 (política dura S11 directo).
    reject_on_dangling: bool = Field(default=False, alias="REJECT_ON_DANGLING")
    reformulation_prompt_version: str = Field(default="v1", alias="REFORMULATION_PROMPT_VERSION")

    # === Calidad de generación (S11): augmentation. Todo off → comportamiento pre-S11. ===
    context_compression_enabled: bool = Field(
        default=False, alias="CONTEXT_COMPRESSION_ENABLED",
        description="Compresión extractiva determinista de cada chunk (limpia el vector).",
    )
    keypoint_max_chars: int = Field(default=600, alias="KEYPOINT_MAX_CHARS")
    reorder_by_edges_enabled: bool = Field(
        default=False, alias="REORDER_BY_EDGES_ENABLED",
        description="Reorden por extremos contra el 'lost-in-the-middle'.",
    )

    # === Calidad de generación (S11): gate de alucinaciones ===
    hallucination_gate_enabled: bool = Field(
        default=False, alias="HALLUCINATION_GATE_ENABLED",
        description="Ancla numérica + juez → degradar líneas no sostenidas a cero.",
    )
    hours_per_engineer_day: float = Field(
        default=8.0, alias="HOURS_PER_ENGINEER_DAY",
        description="Conversión días↔horas del ancla numérica (fricción de unidades).",
    )
    numeric_deviation_tolerance: float = Field(
        default=0.25, alias="NUMERIC_DEVIATION_TOLERANCE",
        description="Desviación relativa máxima línea-vs-evidencia antes de degradar.",
    )
    judge_enabled: bool = Field(default=True, alias="JUDGE_ENABLED")
    judge_prompt_version: str = Field(default="v1", alias="JUDGE_PROMPT_VERSION")

    # === Calidad de generación (S11): síntesis de rangos honestos ===
    synthesis_enabled: bool = Field(
        default=False, alias="SYNTHESIS_ENABLED",
        description="Sintetiza HourRange por línea desde las horas de las fuentes citadas.",
    )
    contradiction_cv_threshold: float = Field(
        default=0.5, alias="CONTRADICTION_CV_THRESHOLD",
        description="Coef. de variación por encima del cual se descarta la síntesis (contradicción).",
    )
    synthesis_reason_enabled: bool = Field(default=True, alias="SYNTHESIS_REASON_ENABLED")
    synthesis_reason_prompt_version: str = Field(
        default="v1", alias="SYNTHESIS_REASON_PROMPT_VERSION"
    )

    # === Curación del corpus (S11): gate de indexabilidad en la ingesta ===
    enforce_indexability_gate: bool = Field(
        default=True, alias="ENFORCE_INDEXABILITY_GATE",
        description="No vectorizar excepciones/casos límite (garbage-in-garbage-out).",
    )

    # === Retrieval avanzado (S10): híbrida + reranking ===
    rag_search_mode: str = Field(
        default="vector", alias="RAG_SEARCH_MODE",
        description="Modo por defecto: vector | hybrid.",
    )
    reranking_enabled: bool = Field(default=False, alias="RERANKING_ENABLED")
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL_NAME"
    )
    retrieval_candidate_pool_size: int = Field(
        default=50, alias="RETRIEVAL_CANDIDATE_POOL_SIZE",
        description="Recall amplio antes del reranking (recall-then-rerank).",
    )
    rrf_smoothing_k: int = Field(default=60, alias="RRF_SMOOTHING_K")

    # === Retrieval avanzado (S10): routing multi-índice en cascada ===
    # Toggles a False por defecto → el pipeline replica el comportamiento de pre-S10.
    routing_enabled: bool = Field(default=False, alias="ROUTING_ENABLED")
    routing_prompt_version: str = Field(default="v1", alias="ROUTING_PROMPT_VERSION")

    # === Retrieval avanzado (S10): transformación de consulta ===
    query_transform_enabled: bool = Field(default=False, alias="QUERY_TRANSFORM_ENABLED")
    query_transform_strategy: str = Field(
        default="auto",
        alias="QUERY_TRANSFORM_STRATEGY",
        description="auto | expand | decompose | off.",
    )
    query_transform_max_subqueries: int = Field(
        default=4, alias="QUERY_TRANSFORM_MAX_SUBQUERIES"
    )
    query_transform_prompt_version: str = Field(
        default="v1", alias="QUERY_TRANSFORM_PROMPT_VERSION"
    )

    # === Retrieval avanzado (S10): ponderación blanda (último ajuste, solo desempate) ===
    # Decaimiento temporal por AÑO (chunks llevan `year`, no fecha): semivida en años
    # (2.5 ≈ la "semivida de 900 días" del directo). Boosts contextuales conservadores.
    temporal_decay_enabled: bool = Field(default=False, alias="TEMPORAL_DECAY_ENABLED")
    temporal_half_life_years: float = Field(
        default=2.5,
        alias="TEMPORAL_HALF_LIFE_YEARS",
        description="Años tras los que el peso temporal cae a 0.5.",
    )
    contextual_weighting_enabled: bool = Field(
        default=False, alias="CONTEXTUAL_WEIGHTING_ENABLED"
    )
    contextual_tech_boost: float = Field(
        default=1.3,
        alias="CONTEXTUAL_TECH_BOOST",
        description="Boost si la tecnología del chunk coincide con la del brief.",
    )
    contextual_sector_boost: float = Field(
        default=1.2,
        alias="CONTEXTUAL_SECTOR_BOOST",
        description="Boost si el sector del chunk coincide con el del brief.",
    )

    # === Flujo invertido de estimación (S10) ===
    # Fase 1 — esqueleto (CAG, sin horas). Alias 'estimator' (potente).
    structure_prompt_version: str = Field(default="v1", alias="STRUCTURE_PROMPT_VERSION")
    structure_temperature: float = Field(default=0.2, alias="STRUCTURE_TEMPERATURE")
    structure_max_tokens: int = Field(default=4000, alias="STRUCTURE_MAX_TOKENS")
    # Fase 2 — horas por-tarea (RAG determinista, consenso de vecinos).
    per_task_top_k: int = Field(default=5, alias="PER_TASK_TOP_K")
    per_task_search_mode: str = Field(default="vector", alias="PER_TASK_SEARCH_MODE")
    per_task_reranking: bool = Field(default=True, alias="PER_TASK_RERANKING")
    per_task_close_distance: float = Field(
        default=0.45,
        alias="PER_TASK_CLOSE_DISTANCE",
        description="Distancia coseno bajo la cual un vecino cuenta como 'cercano'.",
    )
    per_task_min_neighbors_high: int = Field(
        default=2,
        alias="PER_TASK_MIN_NEIGHBORS_HIGH",
        description="Nº de vecinos cercanos para fiabilidad 'high'.",
    )

    # === S12: Agente (Responses API nativa de OpenAI; NO pasa por el Router de LiteLLM) ===
    agent_model: str = Field(default="gpt-5", alias="AGENT_MODEL")
    agent_debug_model: str = Field(default="gpt-5-mini", alias="AGENT_DEBUG_MODEL")
    agent_reasoning_effort: str = Field(default="medium", alias="AGENT_REASONING_EFFORT")
    # "auto" para capturar reasoning summaries en la traza; "detailed"/"concise" también valen.
    agent_reasoning_summary: str = Field(default="auto", alias="AGENT_REASONING_SUMMARY")
    agent_max_steps: int = Field(default=8, alias="AGENT_MAX_STEPS")
    # Recorte del top-k que ve el modelo por búsqueda (observación de alto valor, no volcado crudo).
    agent_search_top_k: int = Field(default=5, alias="AGENT_SEARCH_TOP_K")
    agent_search_mode: str = Field(default="hybrid", alias="AGENT_SEARCH_MODE")
    agent_reranking: bool = Field(default=True, alias="AGENT_RERANKING")
    # calculate_estimate: buffer de contingencia plano y transparente (sin multiplicadores ocultos).
    agent_contingency_factor: float = Field(default=0.15, alias="AGENT_CONTINGENCY_FACTOR")
    # validate_estimate (extensión opcional). Toggle ON por defecto.
    agent_validate_enabled: bool = Field(default=True, alias="AGENT_VALIDATE_ENABLED")
    agent_validate_tolerance_hours: float = Field(
        default=0.5, alias="AGENT_VALIDATE_TOLERANCE_HOURS"
    )
    agent_validate_max_component_hours: float = Field(
        default=5000.0, alias="AGENT_VALIDATE_MAX_COMPONENT_HOURS"
    )

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
