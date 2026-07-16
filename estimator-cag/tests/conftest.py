"""Fixtures compartidas por la suite."""

from __future__ import annotations

import pathlib

import pytest

# Raíz del proyecto (tests/ cuelga de ella).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_transcript_complex() -> str:
    """Transcripción multi-módulo real del repo (la misma que usan los ejemplos S9–S12).
    Se lee de examples/transcripts/ en vez de duplicarla bajo tests/fixtures/."""
    return (
        PROJECT_ROOT / "examples" / "transcripts" / "sample_transcript_complex.txt"
    ).read_text(encoding="utf-8")
