"""LLM wrapper basado en LiteLLM Router con cache exact-match y fallback.

Diseño totalmente síncrono. Expone dos métodos:
- `complete(...)`: devuelve un dict normalizado, para el endpoint no-stream.
- `complete_stream(...)`: devuelve un Iterator[str] de deltas de texto, para el
  endpoint SSE. El puente al event loop async se hace en el endpoint usando
  `run_in_executor` + `asyncio.Queue`.

Ambos métodos:
1. Comprueban la cache antes de llamar al LLM.
2. Llaman al Router de LiteLLM (que gestiona fallback y reintentos).
3. Normalizan la respuesta a un dict uniforme.
4. Escriben en cache al terminar.
5. Loguean exhaustivamente cada fase.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from hashlib import sha256
from typing import Any

import structlog
from litellm import Router

from app.config import Settings, get_settings
from app.core.cache import ExactMatchCache

logger = structlog.get_logger()


def _build_router(settings: Settings) -> Router:
    """Construye el Router de LiteLLM con modelo primario y fallback.

    El nombre lógico del modelo es "estimator" — esto es lo que usa el código
    cliente. El Router resuelve a `primary_model` por defecto y rota a
    `fallback_model` si el primario falla.
    """
    return Router(
        model_list=[
            {
                "model_name": "estimator",
                "litellm_params": {
                    "model": settings.primary_model,
                    "timeout": settings.llm_timeout_seconds,
                },
            },
            {
                "model_name": "estimator",
                "litellm_params": {
                    "model": settings.fallback_model,
                    "timeout": settings.llm_timeout_seconds,
                },
            },
        ],
        fallbacks=[{"estimator": ["estimator"]}],
        num_retries=settings.llm_num_retries,
    )


def _make_cache_key(
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    thinking_budget: int = 0,
) -> str:
    """Genera SHA-256 de los parámetros que determinan la respuesta del LLM."""
    payload = json.dumps(
        {
            "system": system_prompt,
            "user": user_message,
            "model": model,
            "max_tokens": max_tokens,
            "thinking_budget": thinking_budget,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"llm:{sha256(payload.encode('utf-8')).hexdigest()}"


def _parse_provider(model: str) -> str:
    """Extrae el proveedor del nombre de modelo devuelto por LiteLLM.

    LiteLLM devuelve nombres como "claude-haiku-4-5-20251001" o
    "gpt-4o-mini" (sin prefijo necesariamente). Aplicamos heurísticas.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    if model.startswith(("claude-", "anthropic-")):
        return "anthropic"
    if model.startswith(("gpt-", "o3-", "o4-")):
        return "openai"
    return "unknown"


class LLMWrapper:
    """Wrapper unificado con cache, fallback y logging.

    Singleton funcional: instanciar UNA vez y reutilizar. Ver
    `app.services.llm_service.get_wrapper()`.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.router = _build_router(self.settings)
        self.cache = ExactMatchCache(
            redis_url=self.settings.redis_url,
            ttl_seconds=self.settings.cache_ttl_seconds,
            enabled=self.settings.cache_enabled,
        )

    # === Sync completion ===

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        thinking_budget: int = 0,
    ) -> dict[str, Any]:
        """Llamada síncrona con cache lookup, fallback y logging.

        Devuelve dict con: text, model, provider, finish_reason,
        input_tokens, output_tokens, cache_hit.
        """
        log = logger.bind(operation="llm_complete")

        cache_key = _make_cache_key(
            system_prompt, user_message, self.settings.primary_model,
            max_tokens, thinking_budget,
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            log.info("llm_cache_hit", cache_key_prefix=cache_key[:32])
            cached["cache_hit"] = True
            return cached

        log.info(
            "llm_call_started",
            primary_model=self.settings.primary_model,
            fallback_model=self.settings.fallback_model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

        start = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": "estimator",  # nombre lógico — Router resuelve
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if thinking_budget > 0:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                # Anthropic 4.5+ no permite temperature con thinking habilitado
                kwargs.pop("temperature", None)

            response = self.router.completion(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            choice = response.choices[0]
            normalized: dict[str, Any] = {
                "text": choice.message.content or "",
                "model": response.model,
                "provider": _parse_provider(response.model),
                "finish_reason": choice.finish_reason or "stop",
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "cache_hit": False,
            }

            log.info(
                "llm_call_completed",
                model=normalized["model"],
                provider=normalized["provider"],
                finish_reason=normalized["finish_reason"],
                input_tokens=normalized["input_tokens"],
                output_tokens=normalized["output_tokens"],
                latency_ms=latency_ms,
                cache_hit=False,
            )

            self.cache.set(cache_key, normalized)
            return normalized

        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

    # === Sync streaming (pure sync, bridged to async via run_in_executor in the endpoint) ===

    def complete_stream(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Yield text deltas síncrono. Si hay cache hit, yield la respuesta completa en un solo string.

        Este método es SÍNCRONO porque LiteLLM Router expone `completion(...)`
        síncrono y porque queremos un wrapper uniforme. El endpoint SSE
        (`/api/v1/estimate/stream`) hace el puente al event loop async usando
        `loop.run_in_executor(...)` con una `asyncio.Queue` como canal.

        El consumer no necesita distinguir hit vs miss — ambos casos producen
        strings que se envían como eventos SSE.
        """
        log = logger.bind(operation="llm_complete_stream")

        cache_key = _make_cache_key(
            system_prompt, user_message, self.settings.primary_model,
            max_tokens, thinking_budget=0,
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            log.info("llm_cache_hit", cache_key_prefix=cache_key[:32], streaming=True)
            yield cached["text"]
            return

        log.info(
            "llm_stream_started",
            primary_model=self.settings.primary_model,
            max_tokens=max_tokens,
        )

        start = time.time()
        full_parts: list[str] = []
        finish_reason = "stop"
        last_model = self.settings.primary_model

        try:
            response = self.router.completion(
                model="estimator",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in response:
                # LiteLLM normaliza chunks al formato OpenAI
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    full_parts.append(content)
                    yield content

                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr

                model = getattr(chunk, "model", None)
                if model:
                    last_model = model

            latency_ms = int((time.time() - start) * 1000)
            full_text = "".join(full_parts)

            log.info(
                "llm_stream_completed",
                model=last_model,
                finish_reason=finish_reason,
                output_chars=len(full_text),
                latency_ms=latency_ms,
                cache_hit=False,
            )

            # Escribir en cache para futuras llamadas idénticas
            self.cache.set(
                cache_key,
                {
                    "text": full_text,
                    "model": last_model,
                    "provider": _parse_provider(last_model),
                    "finish_reason": finish_reason,
                    "input_tokens": 0,  # no siempre disponibles en streaming
                    "output_tokens": 0,
                    "cache_hit": False,
                },
            )

        except Exception as exc:
            latency_ms = int((time.time() - start) * 1000)
            log.error(
                "llm_stream_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise
