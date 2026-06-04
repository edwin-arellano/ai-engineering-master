"""Critic: audita la estimación del actor y devuelve feedback estructurado.

Aislado: no conoce al actor ni al boss. Solo recibe una estimación y la
transcripción, y produce CriticFeedback. Nunca reescribe la estimación.

Fallback graceful: si la llamada al LLM rompe, devuelve un feedback neutro de
aceptación para que el boss pueda seguir con el resultado del actor en lugar de
tumbar todo el proceso.
"""

from __future__ import annotations

import structlog

from app.foundations.config import Settings, get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.foundations.metrics import TurnMetrics
from app.foundations.prompts.loader import render_critic_prompt
from app.domain.actor_critic_boss import CriticFeedback
from app.domain.estimation import EstimationResult
from app.domain.session import ProjectMetadata
from app.domain.tier import UserTier

logger = structlog.get_logger(__name__)


class CriticService:
    def __init__(self, wrapper: LLMWrapper, settings: Settings | None = None) -> None:
        self.wrapper = wrapper
        self.settings = settings or get_settings()

    def review(
        self,
        *,
        estimation: EstimationResult,
        transcript: str,
        project_metadata: ProjectMetadata,
        tier: UserTier,
        metrics: TurnMetrics | None = None,
    ) -> CriticFeedback:
        """Audita la estimación. Devuelve CriticFeedback (nunca lanza)."""
        system_prompt, user_message = render_critic_prompt(
            transcript=transcript,
            project_metadata=project_metadata,
            tier=tier.value,
            estimation_json=estimation.model_dump_json(),
            version=self.settings.critic_prompt_version,
        )
        try:
            feedback = self.wrapper.complete_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=CriticFeedback,
                max_tokens=2000,
                temperature=0.0,
                max_retries=2,
                metrics=metrics,
            )
            logger.info(
                "critic_review_completed",
                verdict=feedback.verdict.value,
                issues=len(feedback.issues),
                confidence=feedback.review_confidence_pct,
            )
            return feedback
        except Exception as exc:  # noqa: BLE001
            logger.warning("critic_failed_graceful_fallback", error=str(exc))
            return CriticFeedback.empty_accept()
