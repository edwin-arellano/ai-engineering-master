"""Boss: orquesta el loop Actor↔Critic y decide cuándo parar.

El boss NO conoce el contenido técnico de la estimación; gobierna el proceso.
Su decisión es deterministamente simple:
- Si el crítico acepta (sin issues bloqueantes) → accept.
- Si quedan iteraciones y hay issues corregibles → iterate.
- Si se agotan las iteraciones o el crítico rechaza → synthesize.

La síntesis es el ÚNICO punto donde el boss llama al LLM, y es el camino
habitual: con modelos de tier bajo y 3 iteraciones, el actor y el crítico
raramente convergen. La síntesis integra los fixes claros, baja la confianza
y entrega la mejor respuesta disponible.

El boss recibe la función `run_actor` por inyección para no acoplarse al
módulo del actor (aislamiento de roles).
"""

from __future__ import annotations

from typing import Callable

import structlog

from app.config import Settings, get_settings
from app.core.llm_wrapper import LLMWrapper
from app.core.metrics import TurnMetrics
from app.prompts.loader import render_boss_prompt
from app.schemas.actor_critic_boss import (
    ActorCriticBossResult,
    BossDecision,
    BossDecisionType,
    BossIteration,
    CriticFeedback,
    CriticVerdict,
)
from app.schemas.estimation import EstimationResult
from app.schemas.session import Session
from app.schemas.tier import UserTier
from app.services.actor_critic_boss.critic import CriticService

logger = structlog.get_logger(__name__)

RunActor = Callable[..., EstimationResult]


class BossService:
    def __init__(self, wrapper: LLMWrapper, settings: Settings | None = None) -> None:
        self.wrapper = wrapper
        self.settings = settings or get_settings()
        self.critic = CriticService(wrapper, self.settings)

    def run(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: str,
        detail_level: str,
        output_format: str,
        attachments_text: str,
        tier: UserTier,
        run_actor: RunActor,
        metrics: TurnMetrics | None = None,
    ) -> ActorCriticBossResult:
        max_iterations = self.settings.acb_max_iterations
        iterations: list[BossIteration] = []
        critic_feedback: CriticFeedback | None = None
        last_result: EstimationResult | None = None

        for iteration in range(max_iterations):
            # Actor genera (con el feedback del crítico si lo hay).
            # Fallback: si el actor rompe en una iteración >0 (típico cuando el
            # `<critic_feedback>` empuja al modelo a violar los validators),
            # sintetizamos desde el último draft bueno. En iteración 0 no hay
            # nada que rescatar, así que propagamos.
            try:
                last_result = run_actor(
                    wrapper=self.wrapper,
                    session=session,
                    transcript=transcript,
                    project_type=project_type,
                    detail_level=detail_level,
                    output_format=output_format,
                    attachments_text=attachments_text,
                    tier=tier,
                    critic_feedback=critic_feedback,
                    settings=self.settings,
                    metrics=metrics,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "actor_failed_in_iteration",
                    iteration=iteration,
                    error=str(exc),
                )
                if last_result is None:
                    raise
                fallback_feedback = critic_feedback or CriticFeedback.empty_accept()
                synthesized = self._synthesize(
                    last_result, fallback_feedback, metrics=metrics
                )
                return ActorCriticBossResult(
                    final_result=synthesized,
                    iterations=iterations,
                    converged=False,
                    total_iterations=iteration + 1,
                )

            # Critic audita.
            critic_feedback = self.critic.review(
                estimation=last_result,
                transcript=transcript,
                project_metadata=session.project_metadata,
                tier=tier,
                metrics=metrics,
            )

            # Boss decide.
            is_last = iteration == max_iterations - 1
            decision = self._decide(critic_feedback, is_last)
            iterations.append(
                BossIteration(
                    iteration=iteration,
                    critic_feedback=critic_feedback,
                    boss_decision=decision,
                )
            )
            logger.info(
                "boss_iteration",
                iteration=iteration,
                verdict=critic_feedback.verdict.value,
                decision=decision.decision.value,
            )

            if decision.decision is BossDecisionType.ACCEPT:
                return ActorCriticBossResult(
                    final_result=last_result,
                    iterations=iterations,
                    converged=True,
                    total_iterations=iteration + 1,
                )
            if decision.decision is BossDecisionType.SYNTHESIZE:
                synthesized = self._synthesize(
                    last_result, critic_feedback, metrics=metrics
                )
                return ActorCriticBossResult(
                    final_result=synthesized,
                    iterations=iterations,
                    converged=False,
                    total_iterations=iteration + 1,
                )
            # iterate → siguiente vuelta del for con el critic_feedback actual.

        # Si el for termina sin return (no debería con la lógica de is_last),
        # sintetizamos por seguridad.
        synthesized = self._synthesize(last_result, critic_feedback, metrics=metrics)  # type: ignore[arg-type]
        return ActorCriticBossResult(
            final_result=synthesized,
            iterations=iterations,
            converged=False,
            total_iterations=max_iterations,
        )

    def _decide(self, feedback: CriticFeedback, is_last_iteration: bool) -> BossDecision:
        """Decisión de gobernanza. Deliberadamente simple.

        - accept: el crítico acepta o no hay issues bloqueantes.
        - synthesize: el crítico rechaza, o es la última iteración con issues.
        - iterate: quedan iteraciones y hay issues corregibles.
        """
        if feedback.verdict is CriticVerdict.ACCEPT and not feedback.has_blocking():
            return BossDecision(
                decision=BossDecisionType.ACCEPT,
                reasoning="El crítico aceptó la estimación sin issues bloqueantes.",
            )
        if feedback.verdict is CriticVerdict.REJECT:
            return BossDecision(
                decision=BossDecisionType.SYNTHESIZE,
                reasoning=(
                    "El crítico rechazó la estimación; se sintetiza la mejor "
                    "versión disponible."
                ),
            )
        if is_last_iteration:
            return BossDecision(
                decision=BossDecisionType.SYNTHESIZE,
                reasoning=(
                    "Se agotó el presupuesto de iteraciones sin acuerdo; se "
                    "sintetiza."
                ),
            )
        return BossDecision(
            decision=BossDecisionType.ITERATE,
            reasoning=(
                "Hay issues corregibles y quedan iteraciones; se devuelve al "
                "actor."
            ),
        )

    def _synthesize(
        self,
        last_result: EstimationResult,
        feedback: CriticFeedback,
        metrics: TurnMetrics | None = None,
    ) -> EstimationResult:
        """Genera la estimación final integrando los fixes y bajando confianza.

        Único punto donde el boss llama al LLM. Si la síntesis rompe, devuelve
        el último resultado del actor con la confianza reducida en código.
        """
        if not feedback.issues:
            return last_result

        system_prompt, user_message = render_boss_prompt(
            draft_json=last_result.model_dump_json(),
            issues=feedback.issues,
            version=self.settings.boss_prompt_version,
        )
        try:
            synthesized = self.wrapper.complete_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=EstimationResult,
                max_tokens=4000,
                temperature=0.2,
                max_retries=3,
                metrics=metrics,
            )
            logger.info(
                "boss_synthesized",
                original_confidence=last_result.confidence_pct,
                synthesized_confidence=synthesized.confidence_pct,
            )
            return synthesized
        except Exception as exc:  # noqa: BLE001
            logger.warning("boss_synthesis_failed_fallback_to_last", error=str(exc))
            # Fallback: bajamos la confianza del último draft en código.
            reduced = last_result.model_copy(
                update={"confidence_pct": max(0, last_result.confidence_pct - 20)}
            )
            return reduced
