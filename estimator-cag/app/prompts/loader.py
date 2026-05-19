"""Carga y renderizado de los templates Jinja2 del servicio.

Punto único donde el código Python toca los templates. Dos familias:

- ``render_estimation_prompt`` para los templates ``estimation/v*``. Acepta
  los kwargs que cada versión necesita; los bloques opcionales (metadata,
  adjuntos) se omiten cuando no se pasan.
- ``render_metadata_extractor_prompt`` para los templates
  ``metadata_extractor/v*``. Reutiliza la misma infraestructura Jinja.

``StrictUndefined`` está activo a propósito: cualquier variable no definida
rompe el render con un error claro en lugar de producir un prompt malformado
silenciosamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.session import ProjectMetadata

# Directorio raíz de los templates: app/prompts/
PROMPTS_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_estimation_prompt(
    *,
    description: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    project_metadata: ProjectMetadata | None = None,
    attachments_text: str = "",
    version: str = "v3",
) -> tuple[str, str]:
    """Renderiza el par (system, user) para el endpoint de estimación.

    Los kwargs ``project_metadata`` y ``attachments_text`` son opcionales: si
    no se pasan o vienen vacíos, los bloques condicionales correspondientes
    desaparecen del prompt.

    El parámetro ``version`` apunta al subdirectorio bajo ``estimation/``.
    Cambiar a otra versión sin tocar el resto del código permite A/B y
    rollback inmediato.
    """
    context: dict[str, Any] = {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "project_metadata": project_metadata,
        "attachments_text": attachments_text,
    }

    system_template = _env.get_template(f"estimation/{version}/system.j2")
    user_template = _env.get_template(f"estimation/{version}/user.j2")

    return (
        system_template.render(**context),
        user_template.render(**context),
    )


def render_metadata_extractor_prompt(
    *,
    transcript: str,
    assistant_response: str,
    current_metadata: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza el par (system, user) del LLM extractor de metadata."""
    context = {
        "transcript": transcript,
        "assistant_response": assistant_response,
        "current_metadata": current_metadata,
    }
    system_template = _env.get_template(f"metadata_extractor/{version}/system.j2")
    user_template = _env.get_template(f"metadata_extractor/{version}/user.j2")
    return (
        system_template.render(**context),
        user_template.render(**context),
    )
