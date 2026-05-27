"""Política de compresión del historial.

Tres policies:
- `anchor_hybrid` (default): detecta anclas en los mensajes del usuario y, si
  el historial supera el umbral, resume los turnos antiguos en un resumen
  acumulativo, manteniendo una ventana reciente. Las anclas siempre se
  preservan en `anchored_facts`.
- `sliding_window`: el comportamiento de pre-S05. Descarta los pares antiguos.
- `cumulative`: resumen acumulativo sin anclas.

`apply_compression` muta el `ConversationHistory` in-place y se llama DESPUÉS
de registrar el turno, antes de persistir la sesión.
"""

from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.core.llm_wrapper import LLMWrapper
from app.schemas.session import ConversationHistory
from app.services.sessions.compression.anchors import detect_anchors
from app.services.sessions.compression.summarizer import Summarizer

logger = structlog.get_logger(__name__)


def apply_compression(
    *,
    history: ConversationHistory,
    wrapper: LLMWrapper,
    settings: Settings | None = None,
) -> None:
    """Aplica la policy de compresión configurada al historial (in-place)."""
    s = settings or get_settings()
    policy = s.compression_policy

    if policy == "sliding_window":
        _apply_sliding_window(history, s.compression_trigger_turns)
        return

    # anchor_hybrid y cumulative comparten el disparo por umbral de turnos.
    pairs = len(history.messages) // 2
    if pairs <= s.compression_trigger_turns:
        # Aún no toca comprimir, pero en anchor_hybrid actualizamos anclas en
        # cada turno para no perderlas si el turno cae luego.
        if policy == "anchor_hybrid":
            _refresh_anchors(history)
        return

    keep_recent = s.compression_keep_recent_turns * 2  # en mensajes
    to_compress = history.messages[:-keep_recent] if keep_recent else history.messages
    recent = history.messages[-keep_recent:] if keep_recent else []

    if policy == "anchor_hybrid":
        _refresh_anchors(history)

    summarizer = Summarizer(wrapper, version=s.summarizer_prompt_version)
    new_summary = summarizer.summarize(
        messages_to_compress=to_compress,
        previous_summary=history.running_summary,
        anchored_facts=history.anchored_facts if policy == "anchor_hybrid" else [],
    )

    history.running_summary = new_summary
    history.messages = recent

    logger.info(
        "compression_applied",
        policy=policy,
        compressed_pairs=len(to_compress) // 2,
        kept_pairs=len(recent) // 2,
        anchors=len(history.anchored_facts),
    )


def _apply_sliding_window(history: ConversationHistory, max_turns: int) -> None:
    history._truncate(max_turns)
    logger.info("compression_applied", policy="sliding_window", kept_pairs=max_turns)


def _refresh_anchors(history: ConversationHistory) -> None:
    """Detecta anclas nuevas en el historial y las añade sin duplicar."""
    detected = detect_anchors(history.messages)
    for anchor in detected:
        if anchor not in history.anchored_facts:
            history.anchored_facts.append(anchor)
