"""Extractor de ``ProjectMetadata`` basado en una segunda llamada al LLM.

Tras cada turno de estimación, el extractor recibe el último intercambio
user/assistant más el ``ProjectMetadata`` actual y devuelve un patch
(``ProjectMetadataUpdate``) que se aplica al estado de la sesión.

El extractor reutiliza la infraestructura de ``complete_structured`` del
wrapper para forzar el shape del output con Instructor. No es un wrapper
nuevo: es otra llamada ``complete_structured`` con un ``response_model``
distinto.

Política de fallo: si la llamada al LLM falla, se devuelve un patch vacío y
se loguea un warning. Nunca propaga la excepción al pipeline: la extracción
de metadata es opcional, no debe romper la sesión del usuario.
"""

from __future__ import annotations

import structlog

from app.foundations.llm_wrapper import LLMWrapper
from app.foundations.metrics import TurnMetrics
from app.foundations.prompts.loader import render_metadata_extractor_prompt
from app.domain.session import ProjectMetadata, ProjectMetadataUpdate

logger = structlog.get_logger(__name__)


def extract_metadata_update(
    *,
    wrapper: LLMWrapper,
    transcript: str,
    assistant_response: str,
    current_metadata: ProjectMetadata,
    metrics: TurnMetrics | None = None,
) -> ProjectMetadataUpdate:
    """Llama al LLM para deducir qué hechos nuevos aporta el último turno.

    Devuelve siempre un ``ProjectMetadataUpdate`` (posiblemente vacío). Nunca
    propaga excepciones: si el LLM falla, el log captura el motivo y la
    sesión sigue su flujo sin enriquecer la memoria.
    """
    system_prompt, user_message = render_metadata_extractor_prompt(
        transcript=transcript,
        assistant_response=assistant_response,
        current_metadata=current_metadata,
    )
    try:
        return wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ProjectMetadataUpdate,
            max_tokens=1000,
            temperature=0.0,
            max_retries=2,
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 — política deliberada
        logger.warning("metadata_extractor_failed", error=str(exc))
        return ProjectMetadataUpdate()
