"""Punto de entrada de la aplicación FastAPI."""

from fastapi import FastAPI

from app.core.logging_config import configure_logging, request_id_middleware
from app.routers import estimations

# IMPORTANTE: configurar logging ANTES de instanciar FastAPI para que los logs
# del arranque también queden estructurados.
configure_logging()

app = FastAPI(
    title="Estimator CAG",
    description=(
        "Servicio de estimación de software con arquitectura CAG. "
        "Incluye wrapper LiteLLM, cache exact-match Redis y streaming SSE."
    ),
    version="0.3.0",
)

# Middleware de request_id (debe ir antes de incluir routers)
app.middleware("http")(request_id_middleware)

app.include_router(estimations.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Endpoint de health check."""
    return {"status": "healthy"}
