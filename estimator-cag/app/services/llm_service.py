"""Pipeline principal del servicio: orquesta guardrails, cache y LLM.

Orden estricto (cualquier reordenación tiene implicaciones de seguridad):

    1. Input guardrails (capa 2). Si fallan → InputGuardrailError → 400.
    2. Cache exact-match. Si hit → devolver con cache_level="exact_match".
    3. Cache semántico. Si hit → poblar exact-match y devolver con cache_level="semantic".
    4. Render del prompt v2 con Jinja2.
    5. Llamada al LLM con `complete_structured` (capas 3 y 4: prompt robusto +
       validators de Pydantic con retry automático).
    6. Output guardrails (capa 5). Decide si entra a cache.
    7. Si es cacheable, escribir tanto en exact-match como en semantic.
    8. Devolver el resultado al cliente.

El campo `cached` y `cache_level` de `EstimationResponse` permiten al frontend
mostrar el origen de la respuesta y a observabilidad medir la tasa real de
hits por capa.
"""

from __future__ import annotations

from functools import lru_cache

import structlog

from app.config import get_settings
from app.core.llm_wrapper import LLMWrapper
from app.guardrails import (
    should_cache_result,
    validate_input,
)
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)
from app.services.cache import (
    ExactMatchCache,
    SemanticCacheService,
    make_exact_match_key,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


# Settings es un BaseSettings de Pydantic — no es hasheable, así que NO puede
# ir como argumento de funciones cacheadas con lru_cache. Las tres funciones
# leen el singleton vía get_settings() (ya cacheado) y no exponen parámetros.


@lru_cache(maxsize=1)
def _get_wrapper() -> LLMWrapper:
    return LLMWrapper(get_settings())


@lru_cache(maxsize=1)
def _get_exact_cache() -> ExactMatchCache:
    s = get_settings()
    return ExactMatchCache(
        redis_url=s.redis_url,
        ttl_seconds=s.cache_ttl_seconds,
        enabled=s.cache_enabled,
    )


@lru_cache(maxsize=1)
def _get_semantic_cache() -> SemanticCacheService:
    return SemanticCacheService(get_settings())


# ---------------------------------------------------------------------------
# API pública del servicio
# ---------------------------------------------------------------------------


def generate_estimation(request: EstimationRequest) -> EstimationResponse:
    """Ejecuta el pipeline completo y devuelve la respuesta del endpoint.

    Si los input guardrails fallan, propaga `InputGuardrailError`. El router
    HTTP es el que lo traduce a `HTTPException 400` con detalle estructurado.
    """
    settings = get_settings()
    prompt_version = settings.prompt_version

    # Capa 2: input guardrails. Si dispara, no toca cache ni LLM.
    validate_input(request.description, settings)

    exact_cache = _get_exact_cache()
    semantic_cache = _get_semantic_cache()

    # Paso 2: exact-match (más barato que embedding).
    exact_key = make_exact_match_key(request, prompt_version)
    cached_exact = exact_cache.get(exact_key)
    if cached_exact is not None:
        logger.info("cache_hit", level="exact_match", key=exact_key)
        return EstimationResponse(
            result=cached_exact,
            prompt_version=prompt_version,
            cached=True,
            cache_level="exact_match",
        )

    # Paso 3: cache semántico. Más caro (embedding ~50-100 ms), pero captura
    # reformulaciones del mismo input.
    cached_semantic = semantic_cache.lookup(request, prompt_version)
    if cached_semantic is not None:
        logger.info("cache_hit", level="semantic")
        # Poblamos exact-match para que la próxima request idéntica sea gratis.
        exact_cache.set(exact_key, cached_semantic)
        return EstimationResponse(
            result=cached_semantic,
            prompt_version=prompt_version,
            cached=True,
            cache_level="semantic",
        )

    # Paso 4: render del prompt v2.
    system_prompt, user_message = render_estimation_prompt(
        request, version=prompt_version
    )

    # Paso 5: llamada al LLM con structured outputs.
    wrapper = _get_wrapper()
    result = wrapper.complete_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=EstimationResult,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        max_retries=3,
    )

    # Paso 6 + 7: output guardrails + cache write.
    if should_cache_result(result, settings):
        exact_cache.set(exact_key, result)
        semantic_cache.store(request, prompt_version, result)
    else:
        logger.info(
            "estimation_not_cached",
            confidence_pct=result.confidence_pct,
            out_of_scope=result.summary.startswith("Out of scope:"),
        )

    return EstimationResponse(
        result=result,
        prompt_version=prompt_version,
        cached=False,
        cache_level=None,
    )


__all__ = ["generate_estimation"]
