"""Pipeline conversacional del servicio (S05).

Orden estricto del turno (el orden importa):
    1. Guardrails de entrada (transcript + adjuntos).
    2. Resolver tier (heurístico).
    3. Render del prompt v3 (con tier + project_metadata + adjuntos).
    4. Despacho por modo:
       - actor: una llamada al actor.
       - actor_critic_boss: el boss orquesta actor↔critic en bucle.
    5. (output handling — is_out_of_scope se evalúa para el cliente).
    6. Aplicar compresión al histórico.
    7. Actualizar project_metadata (LLM extractor).
    8. Persistir y devolver.
"""

from __future__ import annotations

import time

import structlog

from app.config import Settings, get_settings
from app.core.llm_wrapper import LLMWrapper
from app.core.metrics import TurnMetrics
from app.guardrails import validate_input
from app.prompts.loader import render_estimation_prompt
from app.schemas.actor_critic_boss import ActorCriticBossResult, CriticFeedback
from app.schemas.estimation import EstimationResponse, EstimationResult
from app.schemas.session import EstimationMode, Session
from app.schemas.tier import UserTier
from app.services.actor_critic_boss import BossService
from app.services.attachments import ExtractedAttachment, build_attachments_block
from app.services.metadata_extractor import extract_metadata_update
from app.services.sessions import SessionStore
from app.services.sessions.compression import apply_compression
from app.services.tiers import TierContext, get_tier_resolver

logger = structlog.get_logger(__name__)


def _run_actor(
    *,
    wrapper: LLMWrapper,
    session: Session,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments_text: str,
    tier: UserTier,
    critic_feedback: CriticFeedback | None,
    settings: Settings,
    metrics: TurnMetrics | None = None,
) -> EstimationResult:
    """Genera UN draft de estimación. Reutilizado en modo actor y en ACB.

    Cuando `critic_feedback` no es None, el prompt v3 inyecta el bloque
    `<critic_feedback>` para que el actor refine en base a las correcciones.
    """
    system_prompt, user_message = render_estimation_prompt(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        project_metadata=session.project_metadata,
        attachments_text=attachments_text,
        tier=tier.value,
        critic_feedback=critic_feedback,
        version=settings.prompt_version,
    )
    history_messages = session.history.to_api_messages(system_prompt)
    history_messages.append({"role": "user", "content": user_message})

    return wrapper.complete_structured_with_messages(
        messages=history_messages,
        response_model=EstimationResult,
        max_tokens=4000,
        temperature=0.3,
        max_retries=3,
        metrics=metrics,
    )


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
    s = settings or get_settings()
    # Wall-clock del turno completo (lo que percibe el cliente) y acumulador de
    # métricas de todas las llamadas LLM del turno. El sink es opcional aguas
    # abajo: si no se propagara, el comportamiento sería idéntico al histórico.
    turn_started = time.perf_counter()
    metrics = TurnMetrics()

    # 1. Guardrails de entrada.
    attachments_block = build_attachments_block(attachments)
    combined = "\n\n".join(filter(None, [transcript, attachments_block]))
    validate_input(combined, s)

    # 2. Resolver tier.
    resolver = get_tier_resolver(s)
    resolution = resolver.resolve(
        TierContext(transcript=transcript, project_metadata=session.project_metadata)
    )
    tier = resolution.tier

    # 3-4. Despacho por modo.
    acb_result: ActorCriticBossResult | None = None
    if session.estimation_mode is EstimationMode.ACTOR_CRITIC_BOSS:
        boss = BossService(wrapper=wrapper, settings=s)
        acb_result = boss.run(
            session=session,
            transcript=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            attachments_text=attachments_block,
            tier=tier,
            run_actor=_run_actor,
            metrics=metrics,
        )
        result = acb_result.final_result
    else:
        result = _run_actor(
            wrapper=wrapper,
            session=session,
            transcript=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            attachments_text=attachments_block,
            tier=tier,
            critic_feedback=None,
            settings=s,
            metrics=metrics,
        )

    # 6. Registrar el turno y comprimir el histórico.
    session.history.append_turn(
        user_content=_build_turn_user_record(transcript, attachments_block),
        assistant_content=result.model_dump_json(),
    )
    apply_compression(history=session.history, wrapper=wrapper, settings=s)

    # 7. Actualizar metadata.
    patch = extract_metadata_update(
        wrapper=wrapper,
        transcript=transcript,
        assistant_response=result.model_dump_json(),
        current_metadata=session.project_metadata,
        metrics=metrics,
    )
    session.project_metadata = session.project_metadata.apply_patch(patch)

    # ---- Observabilidad del turno (Bloque 1) ----
    # `latency_ms` es wall-clock del turno completo (lo que percibe el cliente),
    # no la suma de latencias LLM (esa vive aparte en `metrics.llm_latency_ms`).
    # `cost_usd`/`tokens_*` son el agregado de TODAS las llamadas del turno.
    session.turn_count += 1
    session.last_resolved_tier = tier.value
    session.last_tier_rule = resolution.rule_name
    wall_clock_ms = (time.perf_counter() - turn_started) * 1000

    turn_observed = {
        "turn_index": session.turn_count,
        "session_id": session.session_id,
        "enriched_transcript_chars": len(combined),
        "attachments_total_chars": len(attachments_block),
        "messages_in_window": len(session.history.messages),
        "anchors_count": len(session.history.anchored_facts),
        "summary_chars": len(session.history.running_summary or ""),
        "tokens_in": metrics.tokens_in,
        "tokens_out": metrics.tokens_out,
        "cost_usd": round(metrics.cost_usd, 6),
        "latency_ms": round(wall_clock_ms, 2),
        "cache_hit_kind": "none",  # caché dormido en el flujo conversacional
        "last_resolved_tier": tier.value,
    }
    session.last_turn_observed = turn_observed
    logger.info("turn_observed", **turn_observed)

    # 8. Persistir y devolver.
    session_store.save(session)

    logger.info(
        "session_turn_completed",
        session_id=session.session_id,
        mode=session.estimation_mode.value,
        tier=tier.value,
        converged=(acb_result.converged if acb_result else None),
        total_iterations=(acb_result.total_iterations if acb_result else 1),
        history_messages=len(session.history.messages),
        anchors=len(session.history.anchored_facts),
    )

    return EstimationResponse(
        result=result,
        prompt_version=s.prompt_version,
        cached=False,
        cache_level=None,
        tier=tier.value,
        estimation_mode=session.estimation_mode.value,
        acb_converged=(acb_result.converged if acb_result else None),
        acb_total_iterations=(acb_result.total_iterations if acb_result else None),
        acb_iterations=(acb_result.iterations if acb_result else None),
    )


def _build_turn_user_record(transcript: str, attachments_block: str) -> str:
    """Texto que se guarda como mensaje de usuario en el historial.

    Importante: el detector de anclas escanea este contenido, así que conviene
    que incluya el transcript literal (donde el usuario menciona NDAs, etc.).
    """
    if attachments_block:
        return f"{transcript}\n\n[attachments included]"
    return transcript


__all__ = ["generate_estimation_in_session", "_run_actor"]
