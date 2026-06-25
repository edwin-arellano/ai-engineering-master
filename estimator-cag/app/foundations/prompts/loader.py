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

from app.domain.actor_critic_boss import CriticFeedback
from app.domain.session import ProjectMetadata

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
    tier: str = "developer",
    critic_feedback: CriticFeedback | None = None,
    version: str = "v3",
) -> tuple[str, str]:
    """Renderiza el par (system, user) para el endpoint de estimación.

    Los kwargs ``project_metadata`` y ``attachments_text`` son opcionales: si
    no se pasan o vienen vacíos, los bloques condicionales correspondientes
    desaparecen del prompt.

    `tier` materializa el bloque condicional `<tier_guidance>` (S05).
    `critic_feedback` activa el bloque `<critic_feedback>` cuando el actor
    está iterando dentro del flujo Actor-Critic-Boss (S05).

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
        "tier": tier,
        "critic_feedback": critic_feedback,
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


def render_critic_prompt(
    *,
    transcript: str,
    project_metadata: ProjectMetadata,
    tier: str,
    estimation_json: str,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza el par (system, user) del crítico del flujo Actor-Critic-Boss."""
    context: dict[str, Any] = {
        "transcript": transcript,
        "project_metadata": project_metadata,
        "tier": tier,
        "estimation_json": estimation_json,
    }
    system_template = _env.get_template(f"critic/{version}/system.j2")
    user_template = _env.get_template(f"critic/{version}/user.j2")
    return (
        system_template.render(**context),
        user_template.render(**context),
    )


def render_boss_prompt(
    *,
    draft_json: str,
    issues: list[Any],
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza el par (system, user) del boss en su modo de síntesis."""
    context: dict[str, Any] = {
        "draft_json": draft_json,
        "issues": issues,
    }
    system_template = _env.get_template(f"boss/{version}/system.j2")
    user_template = _env.get_template(f"boss/{version}/user.j2")
    return (
        system_template.render(**context),
        user_template.render(**context),
    )


def render_catalog_evaluator_prompt(version: str = "v1") -> str:
    """Renderiza el system prompt del evaluador de fuentes del catálogo (S06).

    A diferencia de las demás familias, el evaluador solo necesita el system
    prompt: el user message es el JSON de hechos factuales que arma el propio
    evaluador (ver ``app.ingest.catalog.evaluator``). No recibe contexto extra,
    de ahí que no haya plantilla ``user.j2`` para esta familia.
    """
    system_template = _env.get_template(f"catalog_evaluator/{version}/system.j2")
    return system_template.render()


def render_propositional_prompt(version: str = "v1") -> str:
    """Renderiza el system prompt del chunker propositional (S07).

    Solo system: el user message es el texto del componente a descomponer en
    proposiciones (ver ``app.generation.rag.chunking.strategies.propositional``).
    """
    system_template = _env.get_template(f"propositional/{version}/system.j2")
    return system_template.render()


def render_contextual_retrieval_prompt(version: str = "v1") -> str:
    """Renderiza el system prompt del chunker contextual_retrieval (S07).

    Solo system: el user message lleva el documento completo y el chunk (ver
    ``app.generation.rag.chunking.strategies.contextual_retrieval``).
    """
    system_template = _env.get_template(f"contextual_retrieval/{version}/system.j2")
    return system_template.render()


def render_reformulation_prompt(version: str = "v1") -> str:
    """Renderiza el system prompt de la fase de reformulación RAG (S09).

    Solo system: el user message es la transcripción completa (ver
    ``app.generation.rag.retrieval.reformulation``).
    """
    return _env.get_template(f"reformulation/{version}/system.j2").render()


def render_rag_estimation_prompt(version: str = "v1") -> str:
    """Renderiza el system prompt de la generación RAG-grounded (S09).

    Solo system: el user message lleva el brief + los context_blocks (ver
    ``app.generation.rag.retrieval.generation``).
    """
    return _env.get_template(f"rag_estimation/{version}/system.j2").render()


def render_summarizer_prompt(
    *,
    transcript_block: str,
    previous_summary: str | None,
    anchored_facts: list[str],
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza el par (system, user) del summarizer acumulativo (S05).

    Los templates concretos viven en `app/prompts/summarizer/v*/`. Esta función
    es el único punto donde el código Python toca esos templates; los kwargs
    opcionales (`previous_summary`, `anchored_facts`) los gestiona la plantilla
    con bloques condicionales.
    """
    context: dict[str, Any] = {
        "transcript_block": transcript_block,
        "previous_summary": previous_summary,
        "anchored_facts": anchored_facts,
    }
    system_template = _env.get_template(f"summarizer/{version}/system.j2")
    user_template = _env.get_template(f"summarizer/{version}/user.j2")
    return (
        system_template.render(**context),
        user_template.render(**context),
    )
