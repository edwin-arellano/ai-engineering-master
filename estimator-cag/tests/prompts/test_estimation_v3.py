"""Tests del template v3 (con bloques opcionales)."""

from __future__ import annotations

from app.foundations.prompts.loader import render_estimation_prompt
from app.domain.session import ProjectMetadata


def _kwargs(**overrides) -> dict:
    defaults = {
        "description": "Project description here",
        "project_type": "other",
        "detail_level": "medium",
        "output_format": "narrative",
        "version": "v3",
    }
    defaults.update(overrides)
    return defaults


def test_v3_without_metadata_omits_block() -> None:
    system, _ = render_estimation_prompt(
        **_kwargs(
            description="Mobile app for booking medical appointments.",
            project_type="mobile_app",
            detail_level="detailed",
            output_format="phases_table",
        )
    )
    assert "<project_metadata>" not in system


def test_v3_with_metadata_includes_block() -> None:
    metadata = ProjectMetadata(
        project_name="BookFlow",
        assumed_team_size=4,
        mentioned_technologies=["Swift", "Kotlin"],
        agreed_scope="Login, calendar, push reminders, single EMR.",
    )
    system, _ = render_estimation_prompt(
        **_kwargs(
            description="Refine the previous estimate.",
            project_type="mobile_app",
            detail_level="detailed",
            output_format="phases_table",
            project_metadata=metadata,
        )
    )
    assert "<project_metadata>" in system
    assert "BookFlow" in system
    assert "Swift" in system
    assert "Kotlin" in system


def test_v3_user_without_attachments_omits_block() -> None:
    _, user = render_estimation_prompt(**_kwargs())
    assert "<attachments>" not in user


def test_v3_user_with_attachments_includes_block() -> None:
    _, user = render_estimation_prompt(
        **_kwargs(
            attachments_text='<attachment filename="spec.pdf">Page 1...</attachment>',
        )
    )
    assert "<attachments>" in user
    assert "spec.pdf" in user


def test_v3_keeps_scope_and_constraints() -> None:
    system, _ = render_estimation_prompt(**_kwargs())
    assert "<scope>" in system
    assert "<numerical_constraints>" in system
    assert "Out of scope:" in system


def test_v3_user_uses_transcript_block() -> None:
    """v3 cambia <project_description> por <transcript> en el user template."""
    _, user = render_estimation_prompt(
        **_kwargs(description="A very specific transcript string")
    )
    assert "<transcript>" in user
    assert "A very specific transcript string" in user
