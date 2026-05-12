"""Configuración global del servicio cargada desde variables de entorno."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings del servicio validados al arrancar la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM configuration
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4000

    # API keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # CAG configuration (defaults — el request puede overridearlos)
    default_num_examples: int = 3
    default_preprocessing: str = "none"  # "none" | "inline_cleaning" | "two_phase"
    default_output_format: str = "markdown"  # "markdown" | "json"

    # Runtime
    environment: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia cacheada de Settings (singleton funcional)."""
    return Settings()
