"""Endpoints HTTP de estimación."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    ExampleFormat,
    OutputFormat,
    PreprocessingType,
    StreamEstimationRequest,
)
from app.services.llm_service import (
    build_system_prompt,
    generate_estimation,
    get_wrapper,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

# Sentinela para señalar el fin del stream desde el thread productor
_STREAM_END = object()


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Genera una estimación. Soporta preprocessing, evaluation, JSON output, thinking_budget."""
    return await generate_estimation(request)


@router.post("/estimate/stream")
async def estimate_stream(request: StreamEstimationRequest):
    """Genera la estimación en streaming token a token vía SSE.

    Deliberadamente simple: sin preprocessing, sin evaluation, sin thinking_budget.
    Si la respuesta está cacheada, se devuelve completa en un único evento SSE
    (no se simula streaming artificial).

    Patrón de implementación (siguiendo a Antonio en S3 live):
    - El wrapper expone un iterador síncrono (`complete_stream`).
    - Aquí lanzamos ese iterador en un thread vía `loop.run_in_executor`.
    - Una `asyncio.Queue` actúa de canal entre el thread productor y la corutina
      consumidora que emite eventos SSE.
    - Cuando el iterador termina (o lanza), se empuja un sentinela / la
      excepción a la cola y la corutina cierra el stream limpiamente.
    """
    settings = get_settings()
    wrapper = get_wrapper()

    # System prompt determinista para que la cache exact-match funcione
    system_prompt = build_system_prompt(
        num_examples=request.num_examples,
        example_format=ExampleFormat.MARKDOWN,
        output_format=OutputFormat.MARKDOWN,
        preprocessing=PreprocessingType.NONE,
        deterministic=True,
    )

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        """Puente sync→async vía Queue + run_in_executor.

        El iterador del wrapper corre en un thread del executor por defecto
        de asyncio (ThreadPoolExecutor). Cada chunk se encola con
        `run_coroutine_threadsafe` (porque estamos en otro thread).
        Esta corutina consume la cola y emite los eventos SSE.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def produce_chunks() -> None:
            """Ejecuta el iterador síncrono del wrapper en un thread."""
            try:
                for chunk in wrapper.complete_stream(
                    system_prompt=system_prompt,
                    user_message=request.transcription,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                ):
                    # Estamos en un thread; usar run_coroutine_threadsafe
                    # para empujar de forma segura a la queue del event loop.
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop)

        # Lanzar el productor en el executor por defecto (ThreadPoolExecutor)
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
            # Cliente cerró la conexión. sse-starlette gestiona la limpieza.
            logger.info("stream_client_disconnected")
            raise

    return EventSourceResponse(event_generator())
