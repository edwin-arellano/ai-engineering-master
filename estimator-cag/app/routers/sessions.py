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

from app.config import get_settings
from app.core.llm_wrapper import LLMWrapper
from app.guardrails import InputGuardrailError
from app.schemas.estimation import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.schemas.session import Session, SessionCreateResponse
from app.services.attachments import (
    AttachmentTooLargeError,
    ExtractedAttachment,
    UnsupportedAttachmentError,
    extract_attachment,
)
from app.services.llm_service import generate_estimation_in_session
from app.services.sessions import (
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
def create_session() -> SessionCreateResponse:
    """Crea una sesión vacía y devuelve su identificador."""
    store = get_session_store()
    session = store.create()
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at,
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
