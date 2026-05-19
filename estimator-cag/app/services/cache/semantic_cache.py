"""Cache semántico basado en `redisvl.SemanticCache`.

Estructura de claves:
- Bucket determinista almacenado en `metadata.bucket` (p.ej.
  "v2_mobile_app_detailed_phases_table"). Aísla en código Python qué hits
  son válidos para el tipo de request actual; se evita usar `filter_expression`
  de RediSearch porque su tokenizador trata `_`, `-` y `:` de formas
  inconsistentes y dispara Syntax error o miss silencioso.
- Vector: embedding de `request.description` con `OpenAITextVectorizer`.

Política de fallo: si OpenAI no está configurado o redisvl no puede conectarse,
el servicio se deshabilita silenciosamente (`enabled=False`) y `lookup`/`store`
se vuelven no-ops. Mantiene el servicio funcional cuando solo hay Anthropic.

Sin tests unitarios en esta sesión: requeriría Redis con módulo de search en CI;
queda para sesiones 7-8 cuando entendamos embeddings y vector search a fondo.
"""

from __future__ import annotations

import json

import structlog

from app.config import Settings
from app.schemas.estimation import EstimationResult
from app.schemas.estimation_compat import CachedRequest

logger = structlog.get_logger(__name__)

# Recuperamos varios candidatos por similitud y filtramos por bucket en Python.
# Subir este número aumenta la probabilidad de encontrar un hit del bucket
# correcto cuando el cache tiene mezcla de project_type/detail_level/...
SEMANTIC_LOOKUP_CANDIDATES = 5


def make_bucket_key(request: CachedRequest, prompt_version: str) -> str:
    """Bucket determinista que aísla la búsqueda semántica por tipo de request.

    El bucket viaja en `metadata` y se compara en Python. El separador `_` es
    arbitrario en este punto porque ya no se usa en queries RediSearch.
    """
    return "_".join(
        [
            prompt_version,
            request.project_type.value,
            request.detail_level.value,
            request.output_format.value,
        ]
    )


def _extract_bucket_from_metadata(hit: dict) -> str | None:
    """Devuelve el bucket guardado en metadata, tolerando dict o JSON string.

    redisvl serializa `metadata` como JSON string al guardar, así que en los
    hits suele venir como string. Si en algún momento viene como dict, también
    se soporta.
    """
    metadata = hit.get("metadata")
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        value = metadata.get("bucket")
        return value if isinstance(value, str) else None
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except ValueError:
            return None
        if isinstance(parsed, dict):
            value = parsed.get("bucket")
            return value if isinstance(value, str) else None
    return None


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
        self, request: CachedRequest, prompt_version: str
    ) -> EstimationResult | None:
        """Busca una respuesta cacheada semánticamente similar dentro del bucket.

        Recupera los `SEMANTIC_LOOKUP_CANDIDATES` hits más cercanos por
        similitud y devuelve el primero cuyo bucket coincida con el del
        request. El filtrado se hace en Python para evitar las idiosincrasias
        del tokenizer de RediSearch en queries tag con caracteres no
        alfanuméricos.
        """
        if not self.enabled or self._cache is None:
            return None

        expected_bucket = make_bucket_key(request, prompt_version)
        try:
            hits = self._cache.check(
                prompt=request.description,
                num_results=SEMANTIC_LOOKUP_CANDIDATES,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "semantic_cache_lookup_failed",
                error=str(exc),
                bucket=expected_bucket,
            )
            return None

        if not hits:
            return None

        for hit in hits:
            bucket = _extract_bucket_from_metadata(hit)
            if bucket != expected_bucket:
                continue
            payload = hit.get("response")
            if not payload:
                continue
            try:
                return EstimationResult.model_validate_json(payload)
            except ValueError as exc:
                logger.warning(
                    "semantic_cache_payload_invalid",
                    error=str(exc),
                    bucket=expected_bucket,
                )
                return None

        # Había candidatos por similitud pero ninguno del bucket correcto.
        logger.info(
            "semantic_cache_bucket_mismatch",
            expected_bucket=expected_bucket,
            candidate_count=len(hits),
        )
        return None

    def store(
        self,
        request: CachedRequest,
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
