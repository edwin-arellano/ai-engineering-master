"""Endpoint del flujo de estimación vía grafo LangGraph (S13). Mismo contrato de
salida que S10/S12 (StructuredEstimate) + el status del grafo, en un envelope."""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.domain.structured_estimation import StructuredEstimate
from app.generation.graph.service import run_estimation_graph

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/graph", tags=["graph-estimation"])


class GraphEstimateRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=60000)
    # thread_id del checkpointer. Si se omite, se genera uno (una ejecución nueva).
    estimation_id: str | None = None


class GraphEstimateResponse(BaseModel):
    estimate: StructuredEstimate
    status: str  # "validated" | "needs_review"
    thread_id: str


@router.post("/estimate", response_model=GraphEstimateResponse)
async def estimate_via_graph(
    request: GraphEstimateRequest, http_request: Request
) -> GraphEstimateResponse:
    graph = getattr(http_request.app.state, "estimation_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Grafo no inicializado")
    thread_id = request.estimation_id or str(uuid4())
    try:
        estimate, status = await run_estimation_graph(
            graph, transcript=request.transcript, thread_id=thread_id
        )
    except Exception:  # noqa: BLE001
        logger.exception("graph.estimate.failed", thread_id=thread_id)
        raise HTTPException(status_code=500, detail="Error ejecutando el grafo")
    return GraphEstimateResponse(estimate=estimate, status=status, thread_id=thread_id)
