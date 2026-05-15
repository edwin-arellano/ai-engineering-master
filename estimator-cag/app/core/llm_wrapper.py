"""Wrapper alrededor de LiteLLM Router + Instructor.

Único método público: `complete_structured`. Toda la lógica de selección de
proveedor, retry y fallback vive dentro del Router de LiteLLM; Instructor se
encarga de hacer cumplir el `response_model` (Pydantic) usando tool calling.

Cuando el modelo devuelve un payload que no respeta el schema o falla un
`model_validator` de Pydantic, Instructor reintenta automáticamente mostrando
el error al modelo, hasta `max_retries` veces. Este es nuestro mecanismo de
política "fix con retry" para guardrails de output (capa 4).
"""

from __future__ import annotations

import time
from typing import TypeVar

import instructor
import structlog
from litellm import Router

from app.config import Settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Etiqueta única que usamos en el Router para identificar nuestro "modelo
# lógico". El Router resuelve esta etiqueta a primary_model y, si falla, a
# fallback_model, en este orden.
ROUTER_ALIAS = "estimator"


def _parse_provider(model_name: str) -> str:
    """Devuelve el slug del proveedor a partir del nombre de modelo de LiteLLM."""
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    return "openai"


def _build_router(settings: Settings) -> Router:
    """Construye el Router con primary + fallback bajo la misma alias."""
    deployments = []

    # Primary
    primary_provider = _parse_provider(settings.primary_model)
    deployments.append(
        {
            "model_name": ROUTER_ALIAS,
            "litellm_params": {
                "model": settings.primary_model,
                "api_key": (
                    settings.anthropic_api_key
                    if primary_provider == "anthropic"
                    else settings.openai_api_key
                ),
                "timeout": settings.llm_timeout_seconds,
            },
        }
    )

    # Fallback
    fallback_provider = _parse_provider(settings.fallback_model)
    deployments.append(
        {
            "model_name": ROUTER_ALIAS,
            "litellm_params": {
                "model": settings.fallback_model,
                "api_key": (
                    settings.anthropic_api_key
                    if fallback_provider == "anthropic"
                    else settings.openai_api_key
                ),
                "timeout": settings.llm_timeout_seconds,
            },
        }
    )

    return Router(
        model_list=deployments,
        num_retries=settings.llm_num_retries,
        timeout=settings.llm_timeout_seconds,
        fallbacks=[{ROUTER_ALIAS: [ROUTER_ALIAS]}],
        routing_strategy="simple-shuffle",
    )


class LLMWrapper:
    """Wrapper público del servicio IA. Punto único de entrada al LLM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.router = _build_router(settings)
        # `mode=instructor.Mode.TOOLS` funciona con OpenAI (Structured Outputs
        # vía tool calling) y con Anthropic (tool use forzado). Es el mínimo
        # común denominador que mantiene la portabilidad del Router.
        self._structured_client = instructor.from_litellm(
            self.router.completion,
            mode=instructor.Mode.TOOLS,
        )

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        max_tokens: int = 4000,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> T:
        """Llama al LLM y devuelve una instancia tipada de `response_model`.

        - `max_retries` controla los reintentos de Instructor cuando los
          validators de Pydantic fallan (no los reintentos de red del Router).
        - El retorno es directamente la instancia validada; si Instructor agota
          sus reintentos, propaga la excepción al servicio que llamó.
        """
        started_at = time.perf_counter()
        logger.info(
            "llm_call_started",
            response_model=response_model.__name__,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        try:
            # Instructor 1.x expone una API tipo chat.completions.create(...)
            # — no es invocable directamente como en versiones antiguas.
            result = self._structured_client.chat.completions.create(
                model=ROUTER_ALIAS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=response_model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            logger.error(
                "llm_call_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=round(latency_ms, 2),
            )
            raise
        latency_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "llm_call_completed",
            response_model=response_model.__name__,
            latency_ms=round(latency_ms, 2),
        )
        return result
