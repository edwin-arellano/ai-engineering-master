"""Cache semántico basado en `redisvl.SemanticCache`.

Estructura de claves:
- Bucket determinista: "{prompt_version}:{project_type}:{detail_level}:{output_format}"
  (filtra por similitud solo dentro del mismo "tipo de request").
- Vector: embedding de `request.description` con `OpenAITextVectorizer`.

Política de fallo: si OpenAI no está configurado o redisvl no puede conectarse,
el servicio se deshabilita silenciosamente (`enabled=False`) y `lookup`/`store`
se vuelven no-ops. Mantiene el servicio funcional cuando solo hay Anthropic.

Sin tests unitarios en esta sesión: requeriría Redis con módulo de search en CI;
queda para sesiones 7-8 cuando entendamos embeddings y vector search a fondo.
"""

from __future__ import annotations

import structlog

from app.config import Settings
from app.schemas.estimation import EstimationRequest, EstimationResult

logger = structlog.get_logger(__name__)


def make_bucket_key(request: EstimationRequest, prompt_version: str) -> str:
    """Bucket determinista que aísla la búsqueda semántica por tipo de request.

    Usamos `-` como separador (no `:`) porque `:` es metacarácter de RediSearch
    en tag queries y rompería el filtro `@bucket:{...}` al combinar con el valor.
    """
    return "-".join(
        [
            prompt_version,
            request.project_type.value,
            request.detail_level.value,
            request.output_format.value,
        ]
    )


class SemanticCacheService:
    """Wrapper alrededor de `redisvl.SemanticCache`."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = False
        self._cache = None
        self._distance_threshold = 1.0 - settings.semantic_cache_threshold

        if not settings.semantic_cache_enabled:
            logger.info("semantic_cache_disabled_by_config")
            return

        if not settings.openai_api_key:
            logger.warning(
                "semantic_cache_disabled",
                reason="OPENAI_API_KEY no configurada; embeddings no disponibles",
            )
            return

        try:
            # Imports locales para no penalizar el arranque cuando el cache
            # está deshabilitado.
            from redisvl.extensions.llmcache import SemanticCache
            from redisvl.utils.vectorize import OpenAITextVectorizer

            vectorizer = OpenAITextVectorizer(
                model=settings.embeddings_model,
                api_config={"api_key": settings.openai_api_key},
            )
            self._cache = SemanticCache(
                name=settings.semantic_cache_name,
                redis_url=settings.redis_url,
                vectorizer=vectorizer,
                distance_threshold=self._distance_threshold,
                ttl=settings.semantic_cache_ttl_seconds,
                filterable_fields=[{"name": "bucket", "type": "tag"}],
            )
            self.enabled = True
            logger.info(
                "semantic_cache_initialized",
                threshold=settings.semantic_cache_threshold,
                model=settings.embeddings_model,
                dimensions=settings.embeddings_dimensions,
            )
        except Exception as exc:  # noqa: BLE001
            # Cualquier fallo (Redis sin RediSearch, key inválida, etc.) deja el
            # cache desactivado pero no rompe el servicio.
            logger.warning("semantic_cache_init_failed", error=str(exc))
            self._cache = None

    def lookup(
        self, request: EstimationRequest, prompt_version: str
    ) -> EstimationResult | None:
        """Busca una respuesta cacheada semánticamente similar dentro del bucket."""
        if not self.enabled or self._cache is None:
            return None

        bucket = make_bucket_key(request, prompt_version)
        try:
            hits = self._cache.check(
                prompt=request.description,
                filter_expression=f"@bucket:{{{bucket}}}",
                num_results=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_lookup_failed", error=str(exc), bucket=bucket)
            return None

        if not hits:
            return None

        payload = hits[0].get("response")
        if not payload:
            return None

        try:
            return EstimationResult.model_validate_json(payload)
        except ValueError as exc:
            logger.warning(
                "semantic_cache_payload_invalid", error=str(exc), bucket=bucket
            )
            return None

    def store(
        self,
        request: EstimationRequest,
        prompt_version: str,
        result: EstimationResult,
    ) -> None:
        """Guarda la respuesta junto con el embedding de la descripción."""
        if not self.enabled or self._cache is None:
            return

        bucket = make_bucket_key(request, prompt_version)
        try:
            self._cache.store(
                prompt=request.description,
                response=result.model_dump_json(),
                metadata={"bucket": bucket},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic_cache_store_failed", error=str(exc), bucket=bucket)
