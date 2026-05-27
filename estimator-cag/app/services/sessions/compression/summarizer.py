"""Resumen acumulativo del historial vía LLM.

Genera un resumen plano de los turnos a comprimir, integrando el resumen
previo (si existe) y respetando las anclas detectadas. NO encadena resúmenes
de resúmenes: el resumen previo ya forma parte del contexto que se resume,
pero el resultado es un resumen acumulativo nuevo, no un resumen de resúmenes.
"""

from __future__ import annotations

import structlog

from app.core.llm_wrapper import LLMWrapper
from app.prompts.loader import render_summarizer_prompt
from app.schemas.session import ChatMessage

logger = structlog.get_logger(__name__)


class Summarizer:
    """Resumidor acumulativo basado en el wrapper del proyecto."""

    def __init__(self, wrapper: LLMWrapper, version: str = "v1") -> None:
        self.wrapper = wrapper
        self.version = version

    def summarize(
        self,
        *,
        messages_to_compress: list[ChatMessage],
        previous_summary: str | None,
        anchored_facts: list[str],
    ) -> str:
        """Devuelve un resumen acumulativo plano del bloque a comprimir."""
        transcript_block = "\n".join(
            f"{m.role}: {m.content}" for m in messages_to_compress
        )
        system_prompt, user_message = render_summarizer_prompt(
            transcript_block=transcript_block,
            previous_summary=previous_summary,
            anchored_facts=anchored_facts,
            version=self.version,
        )
        try:
            # Import diferido: `_SummaryEnvelope` es interno al summarizer y
            # vive en `app.schemas.session` para mantener Instructor en una
            # sola capa de schemas.
            from app.schemas.session import _SummaryEnvelope

            envelope = self.wrapper.complete_structured_with_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=_SummaryEnvelope,
                max_tokens=1500,
                temperature=0.2,
                max_retries=2,
            )
            logger.info(
                "history_summarized",
                compressed_messages=len(messages_to_compress),
                summary_chars=len(envelope.summary),
            )
            return envelope.summary
        except Exception as exc:  # noqa: BLE001
            # Si el resumen falla, devolvemos el resumen previo intacto (o vacío).
            # Mejor un resumen viejo que tirar el proceso.
            logger.warning("summarizer_failed", error=str(exc))
            return previous_summary or ""
