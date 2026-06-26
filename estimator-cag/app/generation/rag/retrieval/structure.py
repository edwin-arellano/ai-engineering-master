"""Fase 1 del flujo invertido: transcripción → esqueleto de módulos+tareas SIN horas.
Es CAG puro (no consulta el store). Usa el alias 'estimator' (potente). El prompt
prohíbe explícitamente estimar horas: solo estructura, revisable por el humano."""

from __future__ import annotations

import structlog

from app.domain.structured_estimation import EstimateSkeleton
from app.foundations.config import Settings
from app.foundations.llm_wrapper import ESTIMATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_estimation_skeleton_prompt

logger = structlog.get_logger(__name__)


def generate_skeleton(
    *, transcript: str, wrapper: LLMWrapper, settings: Settings
) -> EstimateSkeleton:
    skeleton = wrapper.complete_structured(
        system_prompt=render_estimation_skeleton_prompt(settings.structure_prompt_version),
        user_message=transcript,
        response_model=EstimateSkeleton,
        alias=ESTIMATOR_ALIAS,
        temperature=settings.structure_temperature,
        max_tokens=settings.structure_max_tokens,
    )
    n_tasks = sum(len(m.tasks) for m in skeleton.modules)
    logger.info("structure.generated", modules=len(skeleton.modules), tasks=n_tasks)
    return skeleton
