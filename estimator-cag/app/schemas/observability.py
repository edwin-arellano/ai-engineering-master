"""Schemas de observabilidad: el evento de turno y el snapshot de sesión."""

from __future__ import annotations

from pydantic import BaseModel


class TurnObserved(BaseModel):
    """Los 13 campos del evento `turn_observed` (Bloque 1 del ejercicio)."""

    turn_index: int
    session_id: str
    enriched_transcript_chars: int
    attachments_total_chars: int
    messages_in_window: int
    anchors_count: int
    summary_chars: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    cache_hit_kind: str  # "none" | "exact" | "semantic" (siempre "none" aquí)
    last_resolved_tier: str


class SessionDebugResponse(BaseModel):
    """Respuesta de `GET /api/v1/sessions/{id}`."""

    session_id: str
    estimation_mode: str
    turn_count: int
    message_count: int
    anchors_count: int
    summary_chars: int
    last_resolved_tier: str | None
    last_tier_rule: str | None
    last_turn_observed: TurnObserved | None
