"""Endpoint del agente de estimación (S12). Contrato tipo S9 (transcripción → estimación
estructurada), pero por debajo corre el bucle agéntico. El backend de negocio enruta por
`status`: done → guardar; max_steps_exceeded → revisión manual. El bucle es invisible fuera."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.foundations.config import get_settings
from app.generation.agentic.agent import run_agent
from app.generation.agentic.agent_schemas import AgentResult
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["agentic-estimation"])


class EstimateAgenticRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=60000)
    # None → settings.agent_model (gpt-5). Útil para forzar gpt-5-mini en depuración.
    model: str | None = None
    stub: bool = False  # red de seguridad: bucle sin BD (reference_retrieval)


@router.post("/estimate-agentic", response_model=AgentResult)
async def estimate_agentic_endpoint(request: EstimateAgenticRequest) -> AgentResult:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    pipeline = RetrievalPipeline(embedder=LiteLLMEmbedder(), session_factory=AsyncSessionLocal)
    try:
        return await run_agent(
            request.transcript,
            client=client,
            pipeline=pipeline,
            settings=settings,
            model=request.model,
            stub=request.stub,
        )
    except Exception:  # noqa: BLE001
        logger.exception("agent.endpoint_failed")
        raise HTTPException(status_code=500, detail="Error ejecutando el agente de estimación")
