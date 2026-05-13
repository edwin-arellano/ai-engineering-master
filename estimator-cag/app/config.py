"""Configuración global del servicio cargada desde variables de entorno."""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings del servicio validados al arrancar la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === LiteLLM configuration (REEMPLAZA llm_provider/llm_model de session-02) ===
    # Formato: "<provider>/<model>" tal como lo espera LiteLLM
    primary_model: str = "anthropic/claude-haiku-4-5-20251001"
    fallback_model: str = "openai/gpt-4o-mini"
    llm_timeout_seconds: int = 30
    llm_num_retries: int = 2
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4000

    # === API keys ===
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # === Redis cache ===
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400  # 24h
    cache_enabled: bool = True

    # === CAG defaults ===
    default_num_examples: int = 3
    default_preprocessing: str = "none"
    default_output_format: str = "markdown"

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
