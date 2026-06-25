"""Endpoint de producto del flujo RAG end-to-end (S09). NO sustituye a
/api/v1/sessions/{id}/estimate (CAG conversacional); coexisten."""

from __future__ import annotations

from functools import lru_cache

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval import (
    EstimateFromTranscriptResult,
    estimate_from_transcript,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["rag-estimation"])


class EstimateFromTranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=10, max_length=60000)


@lru_cache(maxsize=1)
def _wrapper() -> LLMWrapper:
    return LLMWrapper(get_settings())


@router.post("/estimate-from-transcript", response_model=EstimateFromTranscriptResult)
async def estimate_from_transcript_endpoint(
    request: EstimateFromTranscriptRequest,
) -> EstimateFromTranscriptResult:
    settings = get_settings()
    try:
        return await estimate_from_transcript(
            transcript=request.transcript,
            wrapper=_wrapper(),
            embedder=LiteLLMEmbedder(),
            session_factory=AsyncSessionLocal,
            settings=settings,
        )
    except Exception:  # noqa: BLE001
        logger.exception("rag.estimate_from_transcript_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generando la estimación desde la transcripción",
        )
