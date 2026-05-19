"""Schemas del dominio de sesiones conversacionales.

Tres estructuras de estado independientes:

- ``ConversationHistory``: array de mensajes con lógica de ventana deslizante.
  Persiste solo los últimos N pares user+assistant.
- ``ProjectMetadata``: hechos destilados sobre el proyecto en curso. Sobrevive
  al truncado del historial — esta es la propiedad clave que da resistencia
  a la conversación frente al límite de la ventana.
- ``Session``: agregado que contiene ambas estructuras más metadatos
  (``session_id``, timestamps).

``ProjectMetadataUpdate`` es el shape que el LLM extractor devuelve tras cada
turno: una versión parcial de ``ProjectMetadata`` donde todos los campos son
opcionales y se aplica como patch sobre el estado actual.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Memoria: hechos destilados del proyecto
# ---------------------------------------------------------------------------


class ProjectMetadata(BaseModel):
    """Conocimiento acumulado sobre el proyecto que se está estimando.

    Independiente del historial: sobrevive al truncado de la ventana deslizante.
    Se inyecta en el system prompt de cada turno como bloque
    ``<project_metadata>``.
    """

    project_name: str | None = Field(default=None, max_length=200)
    assumed_team_size: int | None = Field(default=None, ge=0, le=200)
    mentioned_technologies: list[str] = Field(default_factory=list, max_length=50)
    agreed_scope: str | None = Field(default=None, max_length=2000)

    def is_empty(self) -> bool:
        """True si ningún campo se ha poblado todavía."""
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def apply_patch(self, patch: "ProjectMetadataUpdate") -> "ProjectMetadata":
        """Aplica un patch del LLM extractor preservando lo ya conocido.

        Reglas:
        - Campos escalares (``project_name``, ``assumed_team_size``,
          ``agreed_scope``) solo se sobrescriben si el patch trae un valor no
          nulo.
        - ``mentioned_technologies`` se mergea por unión (nunca se reemplaza),
          conservando el orden de aparición.
        """
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
    """Shape que el LLM extractor devuelve después de cada turno.

    Todos los campos son opcionales: el modelo solo rellena los que ha podido
    deducir del intercambio user/assistant más reciente.
    """

    project_name: str | None = Field(default=None, max_length=200)
    assumed_team_size: int | None = Field(default=None, ge=0, le=200)
    mentioned_technologies: list[str] | None = Field(default=None, max_length=50)
    agreed_scope: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Historial: mensajes de la conversación
# ---------------------------------------------------------------------------


ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    """Mensaje individual del historial. El system prompt vive aparte."""

    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content no puede estar vacío")
        return value


class ConversationHistory(BaseModel):
    """Lista limitada de mensajes con ventana deslizante.

    El system prompt NO vive aquí: se regenera en cada llamada al LLM a partir
    del ``project_metadata`` actual de la sesión, por lo que ``messages`` solo
    contiene pares user/assistant.
    """

    messages: list[ChatMessage] = Field(default_factory=list)

    def append_turn(
        self, user_content: str, assistant_content: str, max_turns: int
    ) -> None:
        """Añade un par user+assistant y aplica la ventana deslizante."""
        self.messages.append(ChatMessage(role="user", content=user_content))
        self.messages.append(ChatMessage(role="assistant", content=assistant_content))
        self._truncate(max_turns)

    def _truncate(self, max_turns: int) -> None:
        """Descarta los pares más antiguos hasta dejar ``max_turns`` pares."""
        max_messages = max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def to_api_messages(self, system_prompt: str) -> list[dict[str, str]]:
        """Construye el array ``messages`` listo para pasar al LLM."""
        return [{"role": "system", "content": system_prompt}] + [
            {"role": msg.role, "content": msg.content} for msg in self.messages
        ]


# ---------------------------------------------------------------------------
# Sesión: agregado de historial + memoria + metadatos
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """Agregado de estado de una sesión conversacional.

    Volatilidad aceptada: vive solo en memoria del proceso. Si el servicio se
    reinicia, las sesiones se pierden. La persistencia es responsabilidad del
    bloque de producción/escalado, no del modelado de CAG.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    project_metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)

    def touch(self) -> None:
        """Actualiza ``last_activity_at`` para que el TTL idle no purgue la sesión."""
        self.last_activity_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# DTOs de la API
# ---------------------------------------------------------------------------


class SessionCreateResponse(BaseModel):
    """Body devuelto por ``POST /api/v1/sessions``."""

    session_id: str
    created_at: datetime
