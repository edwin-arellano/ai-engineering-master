"""Punto de entrada de la aplicación FastAPI."""

from fastapi import FastAPI

from app.routers import estimations

app = FastAPI(
    title="Estimator CAG",
    description=(
        "Servicio de estimación de software con arquitectura CAG "
        "(Cache Augmented Generation). Recibe transcripciones de reuniones "
        "y devuelve estimaciones generadas por un LLM con contexto estático "
        "inyectado en el prompt."
    ),
    version="0.1.0",
)

app.include_router(estimations.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Endpoint de health check para verificación de vida."""
    return {"status": "healthy"}
