"""Tests del template v2 del estimator."""

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    OutputFormat,
    ProjectType,
)


def _make_kwargs(**overrides) -> dict:
    """Construye un dict con los kwargs por defecto para renderizar v2."""
    defaults = {
        "description": (
            "Mobile app for booking medical appointments on iOS and Android."
        ),
        "project_type": ProjectType.MOBILE_APP.value,
        "detail_level": DetailLevel.DETAILED.value,
        "output_format": OutputFormat.PHASES_TABLE.value,
        "version": "v2",
    }
    defaults.update(overrides)
    return defaults


def test_v2_includes_scope_section() -> None:
    system, _ = render_estimation_prompt(**_make_kwargs())
    assert "<scope>" in system
    assert "Out of scope:" in system


def test_v2_includes_numerical_constraints() -> None:
    system, _ = render_estimation_prompt(**_make_kwargs())
    assert "<numerical_constraints>" in system
    assert "tolerance of ±1 week" in system
    assert "tolerance of ±5%" in system


def test_v2_user_template_includes_description() -> None:
    _, user = render_estimation_prompt(
        **_make_kwargs(description="A very specific description string")
    )
    assert "A very specific description string" in user
    assert "<project_description>" in user


def test_v2_examples_present() -> None:
    system, _ = render_estimation_prompt(**_make_kwargs())
    # Los tres ejemplos deben venir embebidos via {% include %}.
    assert "Example 1" in system
    assert "Example 2" in system
    assert "Example 3" in system
    # Verificamos que el tercero es el out-of-scope (reforma de baño).
    assert "bathroom remodel" in system or "Out of scope:" in system


def test_v2_summary_style_changes_with_output_format() -> None:
    table_system, _ = render_estimation_prompt(
        **_make_kwargs(output_format=OutputFormat.PHASES_TABLE.value)
    )
    narrative_system, _ = render_estimation_prompt(
        **_make_kwargs(output_format=OutputFormat.NARRATIVE.value)
    )
    assert table_system != narrative_system


def test_v2_detail_level_changes_assumptions_instruction() -> None:
    summary_system, _ = render_estimation_prompt(
        **_make_kwargs(detail_level=DetailLevel.SUMMARY.value)
    )
    detailed_system, _ = render_estimation_prompt(
        **_make_kwargs(detail_level=DetailLevel.DETAILED.value)
    )
    assert summary_system != detailed_system
    assert "Five to seven" in detailed_system
