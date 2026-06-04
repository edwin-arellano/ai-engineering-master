"""Cache exact-match sobre Redis.

La clave se construye a partir de los parámetros estructurados del request más
la versión del prompt. Esto permite invalidar implícitamente la cache al
promover el prompt a una versión nueva (las claves dejan de coincidir).

Tolera caídas de Redis: si el cliente no responde, los métodos loguean un
warning y devuelven `None` / no-op. Una caída del cache nunca rompe una
request del usuario.
"""

from __future__ import annotations

import hashlib
import json

import redis
import structlog

from app.domain.estimation import EstimationResult
from app.domain.estimation_compat import CachedRequest

logger = structlog.get_logger(__name__)

CACHE_KEY_PREFIX = "estimation:exact"


def make_exact_match_key(request: CachedRequest, prompt_version: str) -> str:
    """Genera la clave exact-match a partir del request + versión del prompt.

    La serialización es JSON con claves ordenadas para que dos requests idénticos
    produzcan el mismo hash independientemente del orden de los campos en memoria.
    """
    payload = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "prompt_version": prompt_version,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return f"{CACHE_KEY_PREFIX}:{digest}"


class ExactMatchCache:
    """Adaptador alrededor del cliente síncrono de Redis."""

    def __init__(self, redis_url: str, ttl_seconds: int, enabled: bool = True) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._client: redis.Redis | None
        if not enabled:
            logger.info("exact_cache_disabled_by_config")
            self._client = None
            return
        try:
            self._client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            self._client.ping()
            logger.info(
                "exact_cache_connected",
                redis_url=redis_url,
                ttl_seconds=ttl_seconds,
            )
        except redis.RedisError as exc:
            logger.warning("exact_cache_connect_failed", error=str(exc))
            self._client = None

    def get(self, key: str) -> EstimationResult | None:
        """Recupera un `EstimationResult` o devuelve None si no hay hit."""
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
        except redis.RedisError as exc:
            logger.warning("exact_cache_get_failed", error=str(exc), key=key)
            return None
        if raw is None:
            return None
        try:
            return EstimationResult.model_validate_json(raw)
        except ValueError as exc:
            logger.warning("exact_cache_payload_invalid", error=str(exc), key=key)
            return None

    def set(self, key: str, value: EstimationResult) -> None:
        """Guarda un `EstimationResult` con el TTL configurado."""
        if not self._client:
            return
        try:
            self._client.set(
                key,
                value.model_dump_json(),
                ex=self.ttl_seconds,
            )
        except redis.RedisError as exc:
            logger.warning("exact_cache_set_failed", error=str(exc), key=key)
