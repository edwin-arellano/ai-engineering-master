"""Configura Logfire e instrumenta FastAPI/asyncpg/httpx. `instrument_httpx` captura
las llamadas a la OpenAI Responses API (el SDK va por httpx). Convive con structlog:
no lo reemplaza. Sin LOGFIRE_TOKEN, send_to_logfire='if-token-present' manda los spans
a consola local (no rompe CI ni local)."""

from __future__ import annotations

import structlog

from app.foundations.config import Settings

logger = structlog.get_logger(__name__)


def configure_logfire(app, settings: Settings) -> None:
    if not settings.logfire_enabled:
        return
    import logfire

    logfire.configure(
        service_name=settings.logfire_service_name,
        # El token viaja por Settings (None ⇒ Logfire cae a su env var, si la hubiera).
        token=settings.logfire_token or None,
        send_to_logfire="if-token-present",
    )
    # Cada instrumentación depende de su paquete OTel opcional. Un fallo aquí no debe
    # tumbar el arranque: la observabilidad es aditiva, no un requisito de servicio.
    for name, instrument in (
        ("fastapi", lambda: logfire.instrument_fastapi(app)),
        ("asyncpg", logfire.instrument_asyncpg),
        ("httpx", logfire.instrument_httpx),
    ):
        try:
            instrument()
        except Exception as exc:  # noqa: BLE001
            logger.warning("logfire.instrument_failed", target=name, error=str(exc))
    logger.info("logfire.configured", service=settings.logfire_service_name)
