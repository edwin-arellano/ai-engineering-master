"""Configuración global del servicio cargada desde variables de entorno."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings del servicio validados al arrancar la aplicación.

    Si una variable obligatoria falta o tiene un tipo incorrecto, la
    aplicación falla rápido en el arranque en lugar de fallar al
    recibir la primera petición.
    """

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
    llm_max_tokens: int = 2048

    # API keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Runtime
    environment: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia cacheada de Settings (singleton funcional)."""
    return Settings()
