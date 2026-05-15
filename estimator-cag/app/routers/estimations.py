"""Router del endpoint de estimaciones."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, status

from app.guardrails import InputGuardrailError
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.llm_service import generate_estimation

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["estimations"])


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
)
def estimate(request: EstimationRequest) -> EstimationResponse:
    """Genera una estimación estructurada para una descripción de proyecto."""
    try:
        return generate_estimation(request)
    except InputGuardrailError as exc:
        logger.warning(
            "input_guardrail_blocked",
            category=exc.category.value,
            reason=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "input_guardrail",
                "category": exc.category.value,
                "reason": str(exc),
            },
        )
