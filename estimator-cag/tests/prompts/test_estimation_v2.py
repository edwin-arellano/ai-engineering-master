"""Tests del template v2 del estimator."""

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _build_request(**overrides) -> EstimationRequest:
    defaults = dict(
        description="Mobile app for booking medical appointments on iOS and Android.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )
    defaults.update(overrides)
    return EstimationRequest(**defaults)


def test_v2_includes_scope_section() -> None:
    request = _build_request()
    system, _ = render_estimation_prompt(request, version="v2")
    assert "<scope>" in system
    assert "Out of scope:" in system


def test_v2_includes_numerical_constraints() -> None:
    request = _build_request()
    system, _ = render_estimation_prompt(request, version="v2")
    assert "<numerical_constraints>" in system
    assert "tolerance of ±1 week" in system
    assert "tolerance of ±5%" in system


def test_v2_user_template_includes_description() -> None:
    request = _build_request(description="A very specific description string")
    _, user = render_estimation_prompt(request, version="v2")
    assert "A very specific description string" in user
    assert "<project_description>" in user


def test_v2_examples_present() -> None:
    request = _build_request()
    system, _ = render_estimation_prompt(request, version="v2")
    # Los tres ejemplos deben venir embebidos via {% include %}.
    assert "Example 1" in system
    assert "Example 2" in system
    assert "Example 3" in system
    # Verificamos que el tercero es el out-of-scope (reforma de baño).
    assert "bathroom remodel" in system or "Out of scope:" in system


def test_v2_summary_style_changes_with_output_format() -> None:
    table_system, _ = render_estimation_prompt(
        _build_request(output_format=OutputFormat.PHASES_TABLE), version="v2"
    )
    narrative_system, _ = render_estimation_prompt(
        _build_request(output_format=OutputFormat.NARRATIVE), version="v2"
    )
    assert table_system != narrative_system


def test_v2_detail_level_changes_assumptions_instruction() -> None:
    summary_system, _ = render_estimation_prompt(
        _build_request(detail_level=DetailLevel.SUMMARY), version="v2"
    )
    detailed_system, _ = render_estimation_prompt(
        _build_request(detail_level=DetailLevel.DETAILED), version="v2"
    )
    assert summary_system != detailed_system
    assert "Five to seven" in detailed_system
