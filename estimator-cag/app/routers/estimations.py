"""Endpoints HTTP de estimación."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.schemas.legacy_estimation import (
    LegacyExampleFormat,
    LegacyOutputFormat,
    LegacyPreprocessingType,
    StreamEstimationRequest,
)
from app.services.llm_service import (
    build_legacy_system_prompt,
    generate_estimation,
    get_wrapper,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

# Sentinela para el patrón Queue + run_in_executor del stream (sin cambios)
_STREAM_END = object()


# === Endpoint principal (pre-session-04): formulario tipado ===

@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Genera una estimación a partir de parámetros tipados.

    El prompt se compone en el backend desde templates Jinja2 versionados.
    La respuesta es texto libre (markdown); structured outputs llegan en
    session-04.
    """
    return await generate_estimation(request)


# === Endpoint legacy (session-03): streaming SSE con el viejo schema ===

@router.post("/estimate/stream")
async def estimate_stream(request: StreamEstimationRequest):
    """Endpoint legacy de streaming SSE (sin cambios funcionales respecto a session-03).

    Sigue aceptando el viejo `StreamEstimationRequest` con `transcription`.
    Mantenido por simetría con session-03 y para no perder infraestructura
    de streaming. El cliente Streamlit nuevo NO lo consume.
    """
    settings = get_settings()
    wrapper = get_wrapper()

    system_prompt = build_legacy_system_prompt(
        num_examples=request.num_examples,
        example_format=LegacyExampleFormat.MARKDOWN,
        output_format=LegacyOutputFormat.MARKDOWN,
        preprocessing=LegacyPreprocessingType.NONE,
        deterministic=True,
    )

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        """Puente sync→async vía Queue + run_in_executor (sin cambios desde session-03)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def produce_chunks() -> None:
            try:
                for chunk in wrapper.complete_stream(
                    system_prompt=system_prompt,
                    user_message=request.transcription,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop)

        loop.run_in_executor(None, produce_chunks)

        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, Exception):
                    logger.error(
                        "stream_endpoint_failed",
                        error_type=type(item).__name__,
                        error=str(item),
                    )
                    yield {"event": "error", "data": str(item)}
                    return
                yield {"event": "delta", "data": item}
            yield {"event": "done", "data": ""}
        except asyncio.CancelledError:
            logger.info("stream_client_disconnected")
            raise

    return EventSourceResponse(event_generator())
