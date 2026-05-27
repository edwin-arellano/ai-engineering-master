"""Schemas del dominio de sesiones conversacionales.

S05 añade:
- `estimation_mode` en la sesión (actor | actor_critic_boss).
- `running_summary` y `anchored_facts` en el historial, para la compresión
  híbrida con anclas. `to_api_messages` ahora antepone el resumen y las anclas
  al bloque de mensajes recientes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Modo de estimación
# ---------------------------------------------------------------------------


class EstimationMode(StrEnum):
    """Modo con el que se generan las estimaciones de una sesión."""

    ACTOR = "actor"
    ACTOR_CRITIC_BOSS = "actor_critic_boss"


# ---------------------------------------------------------------------------
# Memoria
# ---------------------------------------------------------------------------


class ProjectMetadata(BaseModel):
    project_name: str | None = Field(default=None, max_length=200)
    assumed_team_size: int | None = Field(default=None, ge=0, le=200)
    mentioned_technologies: list[str] = Field(default_factory=list, max_length=50)
    agreed_scope: str | None = Field(default=None, max_length=2000)

    def is_empty(self) -> bool:
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def apply_patch(self, patch: "ProjectMetadataUpdate") -> "ProjectMetadata":
        merged_technologies = list(self.mentioned_technologies)
        for tech in patch.mentioned_technologies or []:
            if tech and tech not in merged_technologies:
                merged_technologies.append(tech)
        return ProjectMetadata(
            project_name=patch.project_name or self.project_name,
            assumed_team_size=(
                patch.assumed_team_size
                if patch.assumed_team_size is not None
                else self.assumed_team_size
            ),
            mentioned_technologies=merged_technologies,
            agreed_scope=patch.agreed_scope or self.agreed_scope,
        )


class ProjectMetadataUpdate(BaseModel):
    project_name: str | None = Field(default=None, max_length=200)
    assumed_team_size: int | None = Field(default=None, ge=0, le=200)
    mentioned_technologies: list[str] | None = Field(default=None, max_length=50)
    agreed_scope: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Historial con soporte de compresión
# ---------------------------------------------------------------------------


ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content no puede estar vacío")
        return value


class ConversationHistory(BaseModel):
    """Historial con ventana reciente + resumen acumulativo + anclas.

    `messages` mantiene los pares user/assistant recientes (no comprimidos).
    `running_summary` es el resumen acumulativo plano de los turnos antiguos.
    `anchored_facts` son los hechos críticos detectados heurísticamente que
    siempre viajan al contexto, sobrevivan o no a la compresión.
    """

    messages: list[ChatMessage] = Field(default_factory=list)
    running_summary: str | None = Field(default=None)
    anchored_facts: list[str] = Field(default_factory=list)

    def append_turn(self, user_content: str, assistant_content: str) -> None:
        """Añade un par user+assistant SIN truncar.

        El truncado/compresión lo gestiona `apply_compression` (S05). Mantener
        el append separado del truncado permite elegir la policy en runtime.
        """
        self.messages.append(ChatMessage(role="user", content=user_content))
        self.messages.append(ChatMessage(role="assistant", content=assistant_content))

    def _truncate(self, max_turns: int) -> None:
        """Ventana deslizante. Disponible como policy `sliding_window`.

        Ya no es el camino por defecto (ver `apply_compression`).
        """
        max_messages = max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def to_api_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Construye el array `messages` para el LLM.

        Orden: system → (resumen + anclas, si existen) → mensajes recientes.
        El resumen y las anclas se inyectan como un único mensaje de rol `user`
        etiquetado, para que el modelo lo trate como contexto previo.
        """
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        context_parts: list[str] = []
        if self.running_summary:
            context_parts.append(
                f"<conversation_summary>\n{self.running_summary}\n</conversation_summary>"
            )
        if self.anchored_facts:
            facts = "\n".join(f"- {fact}" for fact in self.anchored_facts)
            context_parts.append(f"<anchored_facts>\n{facts}\n</anchored_facts>")

        if context_parts:
            api_messages.append(
                {"role": "user", "content": "\n\n".join(context_parts)}
            )

        api_messages.extend(
            {"role": msg.role, "content": msg.content} for msg in self.messages
        )
        return api_messages


# ---------------------------------------------------------------------------
# Sesión
# ---------------------------------------------------------------------------


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    estimation_mode: EstimationMode = Field(default=EstimationMode.ACTOR)
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)

    # Observabilidad (pre-S06). El servicio los puebla al cerrar cada turno.
    # `last_turn_observed` se guarda como dict para no acoplar este módulo con
    # `app/schemas/observability.py`; el router lo reconstruye en `TurnObserved`.
    turn_count: int = Field(default=0)
    last_resolved_tier: str | None = Field(default=None)
    last_tier_rule: str | None = Field(default=None)
    last_turn_observed: dict | None = Field(default=None)

    def touch(self) -> None:
        self.last_activity_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# DTOs de la API
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    """Body opcional de `POST /api/v1/sessions`."""

    estimation_mode: EstimationMode = Field(default=EstimationMode.ACTOR)


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime
    estimation_mode: EstimationMode


# ---------------------------------------------------------------------------
# Envelopes internos (uso de Instructor en el summarizer S05)
# ---------------------------------------------------------------------------


class _SummaryEnvelope(BaseModel):
    """Envoltorio del resumen para Instructor (response_model)."""

    summary: str = Field(..., min_length=1, max_length=4000)
