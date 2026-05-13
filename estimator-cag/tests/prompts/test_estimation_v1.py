"""Tests del template estimation v1.

Estos tests verifican que el render del template produce los strings
esperados según los parámetros del request. No tocan APIs externas
y se ejecutan en milisegundos.
"""

from __future__ import annotations

import pytest

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _make_request(**overrides) -> EstimationRequest:
    """Construye un EstimationRequest con defaults sensatos, sobreescribibles."""
    defaults = {
        "description": (
            "Mobile app with login, chat and push notifications for a "
            "fitness tracker product targeting iOS and Android."
        ),
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.DETAILED,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def test_user_template_includes_description_literally():
    """La descripción del usuario debe aparecer literalmente en el bloque user."""
    description = (
        "Mobile app with login, chat and push notifications for a "
        "fitness tracker product targeting iOS and Android."
    )
    request = _make_request(description=description)

    _, user = render_estimation_prompt(request)

    assert "<project_description>" in user
    assert "</project_description>" in user
    assert description in user


def test_output_format_phases_table_includes_table_keywords():
    """Con output_format=phases_table aparecen las columnas de la tabla.

    Y con output_format=narrative no aparecen (porque el if las omite).
    """
    request_table = _make_request(output_format=OutputFormat.PHASES_TABLE)
    request_narrative = _make_request(output_format=OutputFormat.NARRATIVE)

    system_table, _ = render_estimation_prompt(request_table)
    system_narrative, _ = render_estimation_prompt(request_narrative)

    # Con phases_table aparecen las columnas
    assert "duration_weeks" in system_table
    assert "cost_eur" in system_table
    assert "confidence_pct" in system_table

    # Con narrative SÍ aparecen las menciones porque los ejemplos few-shot
    # (incluidos en todos los formatos) usan esas columnas. Lo que NO
    # aparece con narrative es la instrucción de tabla en la sección
    # output_format del system prompt.
    assert "Return a Markdown table" not in system_narrative
    assert "Return a flowing prose estimate" in system_narrative


def test_detail_level_detailed_adds_assumptions_instruction():
    """Con detail_level=detailed se añade la instrucción de listar asunciones.

    Con detail_level=summary la instrucción NO aparece.
    """
    request_detailed = _make_request(detail_level=DetailLevel.DETAILED)
    request_summary = _make_request(detail_level=DetailLevel.SUMMARY)

    system_detailed, _ = render_estimation_prompt(request_detailed)
    system_summary, _ = render_estimation_prompt(request_summary)

    assert "confidence interval" in system_detailed.lower()
    assert "list the assumptions you made" in system_detailed.lower()

    # En summary la instrucción explícita no aparece. Los ejemplos few-shot
    # sí pueden mencionar la palabra "assumptions" porque uno de ellos la
    # incluye, así que comprobamos solo la instrucción específica.
    assert "list the assumptions you made" not in system_summary.lower()
    assert "confidence interval" not in system_summary.lower()


def test_project_type_is_humanised_in_system():
    """El project_type debe aparecer con guiones bajos reemplazados por espacios.

    El filtro `replace` se aplica en la línea "experienced in <project_type>"
    del system prompt. Otras menciones (en los ejemplos few-shot, etc.) sí
    usan el formato snake_case original.
    """
    request = _make_request(project_type=ProjectType.DATA_PIPELINE)

    system, _ = render_estimation_prompt(request)

    # La línea humanizada existe
    assert "experience in\ndata pipeline projects" in system or "experience in data pipeline projects" in system


def test_examples_block_is_included():
    """El bloque de ejemplos few-shot debe estar incluido en el system."""
    request = _make_request()

    system, _ = render_estimation_prompt(request)

    assert "<examples>" in system
    assert "</examples>" in system


def test_strict_undefined_raises_on_missing_variable():
    """Si un template referencia una variable no provista, debe romper claramente.

    Este test verifica que la configuración de StrictUndefined está activa,
    no que nuestro template tenga variables sin definir.
    """
    from jinja2 import UndefinedError

    from app.prompts.loader import _env

    template = _env.from_string("Hello {{ missing_variable }}")
    with pytest.raises(UndefinedError):
        template.render()
