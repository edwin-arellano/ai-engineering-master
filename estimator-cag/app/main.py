"""Punto de entrada de la aplicación FastAPI."""

from fastapi import FastAPI

from app.foundations.logging_config import configure_logging, request_id_middleware
from app.api.routers import embeddings, ingestion, sessions

# IMPORTANTE: configurar logging ANTES de instanciar FastAPI para que los logs
# del arranque también queden estructurados.
configure_logging()

app = FastAPI(
    title="Estimator CAG",
    description=(
        "Servicio de estimación de software conversacional. Sesiones con "
        "memoria persistente (project_metadata), historial con ventana "
        "deslizante, adjuntos PDF/.docx con extracción local, structured "
        "outputs con Instructor y cinco capas de guardrails. Incluye un "
        "subsistema de ingesta de datos aislado (S06) y un pipeline de "
        "embeddings y chunking aislado (S07)."
    ),
    version="0.7.0",
)

# Middleware de request_id (debe ir antes de incluir routers)
app.middleware("http")(request_id_middleware)

app.include_router(sessions.router)
app.include_router(ingestion.router)
app.include_router(embeddings.router)  # prefix /embeddings vive en el router


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Endpoint de health check."""
    return {"status": "healthy"}
