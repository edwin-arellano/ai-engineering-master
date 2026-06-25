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

from app.foundations.config import Settings
from app.foundations.metrics import CallMetrics, TurnMetrics
from app.foundations.pricing import cost_for, is_known_model

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Etiquetas de los "modelos lógicos" del Router. Cada alias resuelve a su
# primary_model y, si falla, a su fallback_model. Modelamos "modelo por fase"
# (S09): `reformulator` (barato, reformulación) y `estimator` (potente, generación).
REFORMULATOR_ALIAS = "reformulator"
ESTIMATOR_ALIAS = "estimator"


def _parse_provider(model_name: str) -> str:
    """Devuelve el slug del proveedor a partir del nombre de modelo de LiteLLM."""
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    return "openai"


def _deployment(model_name: str, settings: Settings, model: str) -> dict:
    """Construye un deployment del Router resolviendo la API key por proveedor."""
    provider = _parse_provider(model)
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "api_key": (
                settings.anthropic_api_key
                if provider == "anthropic"
                else settings.openai_api_key
            ),
            "timeout": settings.llm_timeout_seconds,
        },
    }


def _build_router(settings: Settings) -> Router:
    """Router con dos modelos lógicos: reformulator (barato) y estimator (potente).
    Cada uno con su primary y su fallback bajo el mismo model_name."""
    deployments = [
        _deployment(ESTIMATOR_ALIAS, settings, settings.primary_model),
        _deployment(ESTIMATOR_ALIAS, settings, settings.fallback_model),
        _deployment(REFORMULATOR_ALIAS, settings, settings.reformulator_primary_model),
        _deployment(REFORMULATOR_ALIAS, settings, settings.reformulator_fallback_model),
    ]
    return Router(
        model_list=deployments,
        num_retries=settings.llm_num_retries,
        timeout=settings.llm_timeout_seconds,
        fallbacks=[
            {ESTIMATOR_ALIAS: [ESTIMATOR_ALIAS]},
            {REFORMULATOR_ALIAS: [REFORMULATOR_ALIAS]},
        ],
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
        alias: str = ESTIMATOR_ALIAS,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        max_retries: int = 3,
        metrics: TurnMetrics | None = None,
    ) -> T:
        """Llama al LLM y devuelve una instancia tipada de `response_model`.

        - `max_retries` controla los reintentos de Instructor cuando los
          validators de Pydantic fallan (no los reintentos de red del Router).
        - El retorno es directamente la instancia validada; si Instructor agota
          sus reintentos, propaga la excepción al servicio que llamó.
        - `metrics` es un sink opcional: cuando se pasa, se le agrega una
          `CallMetrics` con tokens y coste de esta llamada. Cuando es `None`
          (default), el comportamiento es idéntico al histórico.
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
            # `create_with_completion` devuelve `(instancia, completion_cruda)`.
            # El completion crudo expone `usage` (tokens) y `model` (el modelo
            # efectivo que respondió), que Instructor descarta en `create`.
            result, completion = (
                self._structured_client.chat.completions.create_with_completion(
                    model=alias,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_model=response_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                )
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
        call = self._build_call_metrics(completion, latency_ms)
        logger.info(
            "llm_call_completed",
            response_model=response_model.__name__,
            latency_ms=round(latency_ms, 2),
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            cost_usd=round(call.cost_usd, 6),
            model=call.model,
        )
        if metrics is not None:
            metrics.add(call)
        return result

    def complete_structured_with_messages(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        alias: str = ESTIMATOR_ALIAS,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        max_retries: int = 3,
        metrics: TurnMetrics | None = None,
    ) -> T:
        """Variante de ``complete_structured`` que recibe el array ``messages`` ya armado.

        Útil para flujos conversacionales donde el historial vive entre el
        system y el último user. La lógica del LLM es la misma; solo cambia el
        shape del input para no obligar al caller a pasarlo como un par
        ``(system_prompt, user_message)``. ``metrics`` es el mismo sink opcional
        que en ``complete_structured``.
        """
        started_at = time.perf_counter()
        logger.info(
            "llm_call_started",
            response_model=response_model.__name__,
            messages_count=len(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        try:
            result, completion = (
                self._structured_client.chat.completions.create_with_completion(
                    model=alias,
                    messages=messages,
                    response_model=response_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_retries=max_retries,
                )
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
        call = self._build_call_metrics(completion, latency_ms)
        logger.info(
            "llm_call_completed",
            response_model=response_model.__name__,
            latency_ms=round(latency_ms, 2),
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            cost_usd=round(call.cost_usd, 6),
            model=call.model,
        )
        if metrics is not None:
            metrics.add(call)
        return result

    def _build_call_metrics(self, completion, latency_ms: float) -> CallMetrics:
        """Extrae usage del completion de LiteLLM y calcula coste.

        LiteLLM normaliza `completion.usage` con `prompt_tokens`/`completion_tokens`.
        `completion.model` trae el modelo efectivo (el que respondió: primary o fallback).
        """
        usage = getattr(completion, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0
        model = getattr(completion, "model", "unknown") or "unknown"
        provider = _parse_provider(model)  # reusa el helper de módulo existente
        if not is_known_model(model):
            logger.warning("pricing_model_unknown", model=model)
        cost = cost_for(model, tokens_in, tokens_out)
        return CallMetrics(
            model=model,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
