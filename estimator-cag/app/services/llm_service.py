"""Pipeline conversacional del servicio.

El flujo de cada turno dentro de una sesión es:

    1. Validar input (transcript + texto extraído de adjuntos) con los
       guardrails de S04. Se aplica en TODOS los turnos, no solo el primero.
    2. Renderizar el prompt v3 inyectando ``project_metadata`` actual y el
       bloque de adjuntos.
    3. Construir el array ``messages`` con ``history.to_api_messages(system)``
       y añadir el nuevo user al final. El system prompt se regenera en cada
       turno desde la memoria actual: es lo que da resistencia al truncado.
    4. Llamar al LLM con ``complete_structured_with_messages`` para obtener
       un ``EstimationResult`` tipado.
    5. Llamar al LLM extractor para obtener un ``ProjectMetadataUpdate`` y
       aplicarlo como patch sobre la sesión.
    6. Registrar el turno en el historial (con ventana deslizante) y
       persistir la sesión.
    7. Devolver ``EstimationResponse`` con el resultado.

El cache de S04 NO se invoca en este flujo. En multi-turno la cache key
depende de transcript + adjuntos + historial + metadata, así que la tasa de
hits sería ínfima. Las clases siguen en ``app/services/cache/`` por si una
sesión futura las reactiva (cacheando, por ejemplo, las extracciones de PDF).
"""

from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.core.llm_wrapper import LLMWrapper
from app.guardrails import validate_input
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationResponse, EstimationResult
from app.schemas.session import Session
from app.services.attachments import ExtractedAttachment, build_attachments_block
from app.services.metadata_extractor import extract_metadata_update
from app.services.sessions import SessionStore

logger = structlog.get_logger(__name__)


def generate_estimation_in_session(
    *,
    session: Session,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments: list[ExtractedAttachment],
    wrapper: LLMWrapper,
    session_store: SessionStore,
    settings: Settings | None = None,
) -> EstimationResponse:
    """Ejecuta un turno dentro de la sesión y devuelve la estimación.

    Lanza ``InputGuardrailError`` si la validación de entrada falla; el
    router HTTP la traduce a 400 con detalle estructurado. Lanza
    ``ValidationError`` (Pydantic) si tras los reintentos de Instructor el
    modelo sigue sin respetar el schema.
    """
    s = settings or get_settings()
    prompt_version = s.prompt_version

    # 1. Guardrails de entrada sobre transcript + adjuntos concatenados.
    attachments_block = build_attachments_block(attachments)
    combined_input_for_guardrails = "\n\n".join(
        filter(None, [transcript, attachments_block])
    )
    validate_input(combined_input_for_guardrails, s)

    # 2. Renderizar prompt v3 con la memoria actual y el bloque de adjuntos.
    system_prompt, user_message = render_estimation_prompt(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        project_metadata=session.project_metadata,
        attachments_text=attachments_block,
        version=prompt_version,
    )

    # 3. Array messages: system + historial + user nuevo. El system se
    #    regenera en cada llamada desde el project_metadata actual.
    messages = session.history.to_api_messages(system_prompt)
    messages.append({"role": "user", "content": user_message})

    # 4. Llamada al LLM con structured outputs.
    result: EstimationResult = wrapper.complete_structured_with_messages(
        messages=messages,
        response_model=EstimationResult,
        max_tokens=s.llm_max_tokens,
        temperature=s.llm_temperature,
        max_retries=3,
    )

    # 5. LLM extractor → patch del project_metadata.
    patch = extract_metadata_update(
        wrapper=wrapper,
        transcript=transcript,
        assistant_response=result.model_dump_json(),
        current_metadata=session.project_metadata,
    )
    session.project_metadata = session.project_metadata.apply_patch(patch)

    # 6. Registrar el turno en el historial (ventana deslizante) y persistir.
    session.history.append_turn(
        user_content=user_message,
        assistant_content=result.model_dump_json(),
        max_turns=s.max_turns,
    )
    session_store.save(session)

    logger.info(
        "session_turn_completed",
        session_id=session.session_id,
        prompt_version=prompt_version,
        history_size=len(session.history.messages),
        project_name=session.project_metadata.project_name,
        attachments_count=len(attachments),
    )

    return EstimationResponse(
        result=result,
        prompt_version=prompt_version,
        cached=False,
        cache_level=None,
    )


__all__ = ["generate_estimation_in_session"]
