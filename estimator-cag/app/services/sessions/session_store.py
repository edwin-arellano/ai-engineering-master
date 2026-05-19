"""Almacenamiento de sesiones en memoria del proceso.

Implementación deliberadamente simple para pre-S05:

- Diccionario ``dict[session_id, Session]`` con ``threading.Lock`` para acceso
  thread-safe entre requests concurrentes.
- TTL idle configurable: cuando llega un request, se purgan las sesiones que
  llevan más de ``session_idle_ttl_seconds`` sin actividad.
- Sin persistencia: el reinicio del servicio borra todas las sesiones.

Limitación conocida: en modo multi-worker (uvicorn con ``--workers N``) cada
worker tiene su propio store y un cliente puede aterrizar en un worker que no
conoce su ``session_id``. Para multi-worker hay que migrar a Redis como
backend de sesiones; queda fuera de pre-S05.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import structlog

from app.config import get_settings
from app.schemas.session import Session

logger = structlog.get_logger(__name__)


class SessionNotFoundError(KeyError):
    """Se lanza cuando el ``session_id`` no existe (o expiró por TTL idle)."""


class SessionStore:
    """Adaptador in-memory del CRUD mínimo sobre sesiones."""

    def __init__(self, idle_ttl_seconds: int) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._idle_ttl = timedelta(seconds=idle_ttl_seconds)

    def create(self) -> Session:
        """Crea una sesión nueva y la registra."""
        session = Session()
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info("session_created", session_id=session.session_id)
        return session

    def get(self, session_id: str) -> Session:
        """Devuelve la sesión o lanza ``SessionNotFoundError``.

        Purga oportunista: aprovecha la llamada para limpiar sesiones que
        superaron el TTL idle desde el último acceso.
        """
        self._purge_expired()
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def save(self, session: Session) -> None:
        """Persiste cambios sobre una sesión existente y refresca el TTL idle."""
        session.touch()
        with self._lock:
            self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> bool:
        """Elimina la sesión si existe. Devuelve True si se eliminó."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _purge_expired(self) -> None:
        """Elimina sesiones cuya última actividad supera el TTL idle."""
        if self._idle_ttl.total_seconds() <= 0:
            # TTL=0 se interpreta como "purgar todas las sesiones existentes
            # antes de cualquier acceso", útil para tests deterministas.
            with self._lock:
                expired = list(self._sessions.keys())
                self._sessions.clear()
            if expired:
                logger.info("sessions_purged", count=len(expired))
            return

        now = datetime.now(timezone.utc)
        with self._lock:
            expired_ids = [
                sid
                for sid, sess in self._sessions.items()
                if now - sess.last_activity_at > self._idle_ttl
            ]
            for sid in expired_ids:
                del self._sessions[sid]
        if expired_ids:
            logger.info("sessions_purged", count=len(expired_ids))

    # ---- Helpers para tests / introspección ----

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """Singleton del SessionStore (uno por proceso).

    Settings es un BaseSettings de Pydantic — no es hasheable, así que no
    puede ir como argumento de una función cacheada con ``lru_cache``. Se lee
    internamente vía ``get_settings()`` (que ya es singleton).
    """
    settings = get_settings()
    return SessionStore(idle_ttl_seconds=settings.session_idle_ttl_seconds)
