"""Fase 1 del RAG: transcripción ruidosa → brief tipado + search_text denso.
Usa el alias 'reformulator' (modelo barato). Sin acceso al store todavía."""

from __future__ import annotations

import structlog

from app.foundations.config import Settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_reformulation_prompt
from app.generation.rag.schemas import ReformulatedQuery

logger = structlog.get_logger(__name__)


def reformulate_transcript(
    *, transcript: str, wrapper: LLMWrapper, settings: Settings
) -> ReformulatedQuery:
    system_prompt = render_reformulation_prompt(settings.reformulation_prompt_version)
    reformulated = wrapper.complete_structured(
        system_prompt=system_prompt,
        user_message=transcript,
        response_model=ReformulatedQuery,
        alias=REFORMULATOR_ALIAS,
        temperature=settings.reformulator_temperature,
        max_tokens=settings.reformulator_max_tokens,
    )
    logger.info(
        "rag.reformulated",
        sector=reformulated.sector,
        techs=len(reformulated.technologies),
        search_text_chars=len(reformulated.search_text),
    )
    return reformulated
