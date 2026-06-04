"""Endpoints del flujo conversacional de sesiones.

Dos endpoints:

- ``POST /api/v1/sessions`` crea una sesión vacía y devuelve ``session_id``.
- ``POST /api/v1/sessions/{session_id}/estimate`` recibe ``multipart/form-data``
  con ``transcript``, los enums tipados como ``Form()`` y una lista opcional
  de ``attachments`` (``UploadFile``).

Errores HTTP:

- 400 si los guardrails de entrada rechazan el input (categoría en ``detail``).
- 404 si el ``session_id`` no existe (o expiró por TTL idle).
- 413 si un adjunto excede ``ATTACHMENT_MAX_BYTES``.
- 415 si un adjunto tiene un tipo MIME no soportado.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.cag.guardrails import InputGuardrailError
from app.domain.estimation import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.domain.observability import SessionDebugResponse, TurnObserved
from app.domain.session import (
    EstimationMode,
    Session,
    SessionCreateRequest,
    SessionCreateResponse,
)
from app.generation.cag.attachments import (
    AttachmentTooLargeError,
    ExtractedAttachment,
    UnsupportedAttachmentError,
    extract_attachment,
)
from app.generation.cag.llm_service import generate_estimation_in_session
from app.generation.cag.sessions import (
    SessionNotFoundError,
    SessionStore,
    get_session_store,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sessions"])


@lru_cache(maxsize=1)
def _get_wrapper() -> LLMWrapper:
    """Singleton del wrapper (Settings no es hasheable; se lee dentro)."""
    return LLMWrapper(get_settings())


@router.post(
    "/sessions",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    body: SessionCreateRequest | None = None,
) -> SessionCreateResponse:
    """Crea una sesión con el modo de estimación indicado (default ``actor``)."""
    store = get_session_store()
    mode = body.estimation_mode if body else EstimationMode.ACTOR
    session = Session(estimation_mode=mode)
    # `save` actúa como upsert y refresca TTL idle; cubre el alta y la persistencia
    # del modo sin tocar la firma de `SessionStore.create()`.
    store.save(session)
    logger.info("session_created", session_id=session.session_id, mode=mode.value)
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        estimation_mode=session.estimation_mode,
    )


@router.post(
    "/sessions/{session_id}/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
)
async def estimate_in_session(
    session_id: str,
    transcript: str = Form(..., min_length=10, max_length=4000),
    project_type: ProjectType = Form(default=ProjectType.OTHER),
    detail_level: DetailLevel = Form(default=DetailLevel.MEDIUM),
    output_format: OutputFormat = Form(default=OutputFormat.PHASES_TABLE),
    attachments: list[UploadFile] | None = File(default=None),
) -> EstimationResponse:
    """Genera una estimación dentro de una sesión, opcionalmente con adjuntos."""
    settings = get_settings()
    store: SessionStore = get_session_store()

    # 1. Cargar la sesión.
    try:
        session: Session = store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "session_id": session_id},
        )

    # 2. Procesar adjuntos.
    extracted: list[ExtractedAttachment] = []
    for upload in attachments or []:
        try:
            extracted.append(
                await extract_attachment(upload, settings.attachment_max_bytes)
            )
        except UnsupportedAttachmentError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={"error": "unsupported_attachment", "reason": str(exc)},
            )
        except AttachmentTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"error": "attachment_too_large", "reason": str(exc)},
            )

    # 3. Delegar al pipeline conversacional.
    try:
        return generate_estimation_in_session(
            session=session,
            transcript=transcript,
            project_type=project_type.value,
            detail_level=detail_level.value,
            output_format=output_format.value,
            attachments=extracted,
            wrapper=_get_wrapper(),
            session_store=store,
            settings=settings,
        )
    except InputGuardrailError as exc:
        logger.warning(
            "input_guardrail_blocked",
            category=exc.category.value,
            reason=str(exc),
            session_id=session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "input_guardrail",
                "category": exc.category.value,
                "reason": str(exc),
            },
        )


@router.get("/sessions/{session_id}", response_model=SessionDebugResponse)
def get_session_debug(session_id: str) -> SessionDebugResponse:
    """Expone el estado de una sesión + el último turn_observed (debug/stress)."""
    store = get_session_store()
    try:
        session = store.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "session_id": session_id},
        )
    last = session.last_turn_observed
    return SessionDebugResponse(
        session_id=session.session_id,
        estimation_mode=session.estimation_mode.value,
        turn_count=session.turn_count,
        message_count=len(session.history.messages),
        anchors_count=len(session.history.anchored_facts),
        summary_chars=len(session.history.running_summary or ""),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
        last_turn_observed=TurnObserved(**last) if last else None,
        last_summary=session.history.running_summary,
        anchored_facts=session.history.anchored_facts,
        project_metadata=session.project_metadata.model_dump(),
    )
