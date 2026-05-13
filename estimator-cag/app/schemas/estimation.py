"""Schemas del endpoint /api/v1/estimate (pre-session-04 en adelante).

El contrato cambia drásticamente respecto a session-02/03:
- El usuario ya no envía una transcripción cruda, envía parámetros tipados
  capturados por un formulario en el cliente.
- El prompt se compone en el backend a partir de templates Jinja2 versionados.
- La respuesta sigue siendo texto libre (la estructuraremos en session-04 con
  Instructor + structured outputs).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    """Tipo de proyecto a estimar."""

    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    """Nivel de detalle del entregable."""

    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    """Formato del entregable.

    Distinto del LegacyOutputFormat (markdown/json) de session-02/03. Aquí
    representa estructura del contenido, no sintaxis del documento.
    """

    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Request del endpoint POST /api/v1/estimate.

    Producido por el formulario del cliente. Los cuatro campos son los
    parámetros del prompt — el prompt en sí vive en app/prompts/.
    """

    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Descripción libre del proyecto a estimar.",
    )
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat


class EstimationResponse(BaseModel):
    """Response del endpoint POST /api/v1/estimate.

    `text` es texto libre (markdown). `prompt_version` permite al cliente
    saber qué versión del prompt produjo la respuesta — útil para A/B y
    para depuración.
    """

    text: str
    prompt_version: str
