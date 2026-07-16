"""Punto de entrada de la aplicación FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.foundations.logging_config import configure_logging, request_id_middleware
from app.api.routers import agentic, embeddings, ingestion, rag_estimation, search, sessions
from app.generation.rag.persistence.database import engine

# IMPORTANTE: configurar logging ANTES de instanciar FastAPI para que los logs
# del arranque también queden estructurados.
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cierra el engine async de SQLAlchemy (pool de conexiones) al apagar la app."""
    yield
    await engine.dispose()


app = FastAPI(
    title="Estimator CAG",
    description=(
        "Servicio de estimación de software conversacional. Sesiones con "
        "memoria persistente (project_metadata), historial con ventana "
        "deslizante, adjuntos PDF/.docx con extracción local, structured "
        "outputs con Instructor y cinco capas de guardrails. Incluye un "
        "subsistema de ingesta de datos aislado (S06) y un pipeline de "
        "embeddings y chunking aislado (S07) persistido en pgvector (pre-S08). "
        "Incluye el flujo RAG end-to-end (S09): reformulación → retrieval con "
        "filtros de metadata → augmentation → generación RAG-grounded → "
        "verificación, expuesto en POST /api/v1/estimate-from-transcript."
    ),
    version="0.9.0",
    lifespan=lifespan,
)

# Middleware de request_id (debe ir antes de incluir routers)
app.middleware("http")(request_id_middleware)

app.include_router(sessions.router)
app.include_router(ingestion.router)
app.include_router(embeddings.router)  # prefix /embeddings vive en el router
app.include_router(search.router)
app.include_router(rag_estimation.router)  # prefix /api/v1 vive en el router
app.include_router(agentic.router)  # prefix /api/v1 vive en el router


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Endpoint de health check."""
    return {"status": "healthy"}
