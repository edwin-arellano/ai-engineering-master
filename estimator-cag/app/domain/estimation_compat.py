"""Compat para el código del cache (S04) que sigue tipando con un objeto
similar a ``EstimationRequest``.

El nuevo flujo conversacional (pre-S05) no invoca el cache, pero las clases de
``app/services/cache/`` siguen en el repo como infraestructura dormida. Para
que sigan compilando sin reintroducir el ``EstimationRequest`` eliminado,
exponemos un subset mínimo con los campos que el cache leía.

Si en una sesión futura se reactiva el cache para el flujo conversacional,
este shim desaparece sustituido por el nuevo schema de input.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.estimation import DetailLevel, OutputFormat, ProjectType


class CachedRequest(BaseModel):
    """Subset mínimo del request que el cache usaba en S04."""

    description: str = Field(..., min_length=1)
    project_type: ProjectType = Field(default=ProjectType.OTHER)
    detail_level: DetailLevel = Field(default=DetailLevel.MEDIUM)
    output_format: OutputFormat = Field(default=OutputFormat.PHASES_TABLE)
