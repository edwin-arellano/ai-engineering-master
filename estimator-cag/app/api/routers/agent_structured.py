"""Endpoints del flujo híbrido (S12): el agente entra en dos fases del flujo estructurado.

Son DOS endpoints y no uno a propósito: entre ellos vive la puerta humana. El cliente recibe
la estructura propuesta (fase 1), la valida o la edita, y solo entonces pide las horas
(fase 2). Un único endpoint one-shot se saltaría esa revisión, que es justo la guarda que
hace utilizable el sistema.

El backend de negocio no ve el bucle: entra transcripción, sale `EstimateSkeleton`; entra
esqueleto, sale `StructuredEstimate`. Los mismos contratos de S9/S10, con la traza adjunta.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.domain.structured_estimation import (
    EstimateSkeleton,
    StructureProposal,
    StructuredEstimate,
)
from app.foundations.config import get_settings
from app.generation.agentic.structure_agent import propose_structure
from app.generation.agentic.structured_service import estimate_task_hours_agentic
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/agent", tags=["agentic-structured"])


class ProposeStructureRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=60000)
    # None → settings.agent_model (gpt-5). Útil para forzar gpt-5-mini en depuración.
    model: str | None = None


class EstimateTaskHoursRequest(BaseModel):
    # El esqueleto que el humano validó tras la fase 1 — puede no ser el que propuso Neo.
    skeleton: EstimateSkeleton
    model: str | None = None
    stub: bool = False  # red de seguridad: bucle de recuperación sin BD


@router.post("/propose-structure", response_model=StructureProposal)
async def propose_structure_endpoint(request: ProposeStructureRequest) -> StructureProposal:
    """Fase 1: transcripción → estructura módulos/tareas (sin horas) + traza."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        return await propose_structure(
            request.transcript, client=client, settings=settings, model=request.model
        )
    except Exception:  # noqa: BLE001
        logger.exception("agent.structure.endpoint_failed")
        raise HTTPException(status_code=500, detail="Error proponiendo la estructura")


@router.post("/estimate-task-hours", response_model=StructuredEstimate)
async def estimate_task_hours_endpoint(
    request: EstimateTaskHoursRequest,
) -> StructuredEstimate:
    """Fase 2: esqueleto validado → horas deterministas + recuperación agéntica + traza."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    pipeline = RetrievalPipeline(
        embedder=LiteLLMEmbedder(), session_factory=AsyncSessionLocal
    )
    try:
        return await estimate_task_hours_agentic(
            request.skeleton,
            client=client,
            pipeline=pipeline,
            settings=settings,
            model=request.model,
            stub=request.stub,
        )
    except Exception:  # noqa: BLE001
        logger.exception("agent.task_hours.endpoint_failed")
        raise HTTPException(status_code=500, detail="Error estimando las horas por tarea")
