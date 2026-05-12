"""Endpoint HTTP para generación de estimaciones."""

from fastapi import APIRouter

from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Genera una estimación de software a partir de una transcripción.

    El router es deliberadamente delgado: solo recibe, delega y devuelve.
    Toda la lógica de prompt, llamada al LLM y normalización vive en
    `app.services.llm_service`.
    """
    result = await generate_estimation(request.transcription)
    return EstimationResponse(**result)
