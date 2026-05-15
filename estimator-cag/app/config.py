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

    # === Versionado de prompts (S04) ===
    prompt_version: str = Field(default="v2", alias="PROMPT_VERSION")

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
