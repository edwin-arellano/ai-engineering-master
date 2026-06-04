"""Tests del template estimation v1.

Estos tests verifican que el render del template produce los strings
esperados según los parámetros del request. No tocan APIs externas
y se ejecutan en milisegundos.
"""

from __future__ import annotations

import pytest

from app.foundations.prompts.loader import render_estimation_prompt
from app.domain.estimation import (
    DetailLevel,
    OutputFormat,
    ProjectType,
)


def _make_kwargs(**overrides) -> dict:
    """Construye un dict con los kwargs por defecto del loader, sobreescribibles.

    Pre-S05 reemplaza `EstimationRequest` por kwargs explícitos en el loader.
    Los tests v1/v2 fijan ``version`` al template histórico que están
    verificando.
    """
    defaults = {
        "description": (
            "Mobile app with login, chat and push notifications for a "
            "fitness tracker product targeting iOS and Android."
        ),
        "project_type": ProjectType.MOBILE_APP.value,
        "detail_level": DetailLevel.DETAILED.value,
        "output_format": OutputFormat.PHASES_TABLE.value,
        "version": "v1",
    }
    defaults.update(overrides)
    return defaults


def test_user_template_includes_description_literally():
    """La descripción del usuario debe aparecer literalmente en el bloque user."""
    description = (
        "Mobile app with login, chat and push notifications for a "
        "fitness tracker product targeting iOS and Android."
    )
    _, user = render_estimation_prompt(**_make_kwargs(description=description))

    assert "<project_description>" in user
    assert "</project_description>" in user
    assert description in user


def test_output_format_phases_table_includes_table_keywords():
    """Con output_format=phases_table aparecen las columnas de la tabla.

    Y con output_format=narrative no aparecen (porque el if las omite).
    """
    system_table, _ = render_estimation_prompt(
        **_make_kwargs(output_format=OutputFormat.PHASES_TABLE.value)
    )
    system_narrative, _ = render_estimation_prompt(
        **_make_kwargs(output_format=OutputFormat.NARRATIVE.value)
    )

    assert "duration_weeks" in system_table
    assert "cost_eur" in system_table
    assert "confidence_pct" in system_table

    assert "Return a Markdown table" not in system_narrative
    assert "Return a flowing prose estimate" in system_narrative


def test_detail_level_detailed_adds_assumptions_instruction():
    """Con detail_level=detailed se añade la instrucción de listar asunciones."""
    system_detailed, _ = render_estimation_prompt(
        **_make_kwargs(detail_level=DetailLevel.DETAILED.value)
    )
    system_summary, _ = render_estimation_prompt(
        **_make_kwargs(detail_level=DetailLevel.SUMMARY.value)
    )

    assert "confidence interval" in system_detailed.lower()
    assert "list the assumptions you made" in system_detailed.lower()

    assert "list the assumptions you made" not in system_summary.lower()
    assert "confidence interval" not in system_summary.lower()


def test_project_type_is_humanised_in_system():
    """El project_type debe aparecer con guiones bajos reemplazados por espacios."""
    system, _ = render_estimation_prompt(
        **_make_kwargs(project_type=ProjectType.INTERNAL_TOOL.value)
    )

    assert (
        "experience in\ninternal tool projects" in system
        or "experience in internal tool projects" in system
    )


def test_examples_block_is_included():
    """El bloque de ejemplos few-shot debe estar incluido en el system."""
    system, _ = render_estimation_prompt(**_make_kwargs())

    assert "<examples>" in system
    assert "</examples>" in system


def test_strict_undefined_raises_on_missing_variable():
    """Si un template referencia una variable no provista, debe romper claramente.

    Este test verifica que la configuración de StrictUndefined está activa,
    no que nuestro template tenga variables sin definir.
    """
    from jinja2 import UndefinedError

    from app.foundations.prompts.loader import _env

    template = _env.from_string("Hello {{ missing_variable }}")
    with pytest.raises(UndefinedError):
        template.render()
