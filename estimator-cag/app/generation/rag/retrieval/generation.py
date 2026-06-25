"""Fase 4 del RAG: generación estructurada RAG-grounded. Usa el alias 'estimator'
(potente, razonamiento alto modelado con temperatura baja + más tokens). El output
es un RagEstimate validado por Instructor + model_validators."""

from __future__ import annotations

import structlog

from app.domain.rag_estimation import RagEstimate
from app.foundations.config import Settings
from app.foundations.llm_wrapper import ESTIMATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_rag_estimation_prompt
from app.generation.rag.schemas import AugmentedContext, ReformulatedQuery

logger = structlog.get_logger(__name__)


def generate_rag_estimate(
    *,
    reformulated: ReformulatedQuery,
    context: AugmentedContext,
    wrapper: LLMWrapper,
    settings: Settings,
) -> RagEstimate:
    system_prompt = render_rag_estimation_prompt(settings.rag_estimation_prompt_version)
    user_message = (
        f"<brief>\n{reformulated.model_dump_json(indent=2)}\n</brief>\n\n"
        f"<context_blocks>\n{context.context_block}\n</context_blocks>\n\n"
        "Genera la estimación basándote ÚNICAMENTE en los context_blocks. "
        "Cita los source_id de los que deriva cada tarea."
    )
    estimate = wrapper.complete_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=RagEstimate,
        alias=ESTIMATOR_ALIAS,
        temperature=settings.rag_generation_temperature,
        max_tokens=settings.rag_generation_max_tokens,
    )
    logger.info(
        "rag.generated",
        confidence=estimate.confidence.value,
        modules=len(estimate.modules),
        total_days=estimate.total_engineer_days,
    )
    return estimate
