"""Configuración de structlog y middleware de FastAPI para request_id."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response


def configure_logging() -> None:
    """Configura structlog con output dual: ConsoleRenderer en dev, JSON en prod.

    Llamar UNA VEZ al arrancar la aplicación, antes de instanciar FastAPI.
    """
    # Import diferido para evitar ciclo (config también loguea al arrancar)
    from app.foundations.config import get_settings

    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.EventRenamer("msg"),
    ]

    if settings.environment == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Genera un request_id único por petición y lo vincula al contexto structlog.

    Si el cliente envía `X-Request-ID` en cabecera, se respeta. Si no, se genera
    un UUID4. El id viaja también de vuelta al cliente en `X-Request-ID`.

    Todos los logs emitidos durante el ciclo de vida de la request llevan
    automáticamente `request_id`, `method` y `path`.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
