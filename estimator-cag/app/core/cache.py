"""Cache Redis exact-match para respuestas del LLM.

Sin distinción sync/async: usamos cliente síncrono porque las operaciones
de cache son sub-milisegundo y no bloquean el event loop de forma significativa.
"""

from __future__ import annotations

import json
from typing import Any

import redis
import structlog

logger = structlog.get_logger()


class ExactMatchCache:
    """Cache exact-match sobre Redis con TTL configurable.

    Tolera caídas de Redis: si la conexión falla en cualquier momento,
    `enabled` pasa a False y los métodos se vuelven no-ops. Nunca rompe
    requests por errores de cache.
    """

    def __init__(self, redis_url: str, ttl_seconds: int, enabled: bool = True):
        self.ttl = ttl_seconds
        self.enabled = enabled
        self.client: redis.Redis | None = None

        if not enabled:
            logger.info("cache_disabled_by_config")
            return

        try:
            self.client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            self.client.ping()
            logger.info("cache_connected", redis_url=redis_url, ttl_seconds=ttl_seconds)
        except Exception as exc:
            logger.warning(
                "cache_connection_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                redis_url=redis_url,
            )
            self.enabled = False
            self.client = None

    def get(self, key: str) -> dict[str, Any] | None:
        """Recupera valor cacheado, o None si miss/disabled/error."""
        if not self.enabled or self.client is None:
            return None
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("cache_get_failed", error_type=type(exc).__name__, error=str(exc))
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Almacena valor con TTL configurado."""
        if not self.enabled or self.client is None:
            return
        try:
            self.client.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            logger.warning("cache_set_failed", error_type=type(exc).__name__, error=str(exc))
