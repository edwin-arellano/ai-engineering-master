"""Contratos de entrada y salida del endpoint de estimaciones."""

from pydantic import BaseModel, Field


class EstimationRequest(BaseModel):
    """Petición de estimación.

    La transcripción debe tener al menos 50 caracteres para evitar
    llamadas inútiles al LLM con inputs vacíos o triviales.
    """

    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión con el cliente.",
    )


class EstimationResponse(BaseModel):
    """Respuesta del endpoint con la estimación generada y metadatos."""

    estimation: str = Field(
        ...,
        description="Estimación de software generada por el LLM en Markdown.",
    )
    model: str = Field(
        ...,
        description="Modelo del LLM que generó la respuesta.",
    )
    provider: str = Field(
        ...,
        description="Proveedor del LLM (anthropic | openai).",
    )
