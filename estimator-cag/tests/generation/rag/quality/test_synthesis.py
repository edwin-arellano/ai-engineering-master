"""Síntesis de rangos honestos (S11): número determinista, contradicción por dispersión.
El LLM de explicación se desactiva (synthesis_reason_enabled=False) para no pegar al modelo."""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.generation.rag.quality.synthesis import synthesize_range


@pytest.fixture
def settings():
    return get_settings().model_copy(
        update={"synthesis_reason_enabled": False, "contradiction_cv_threshold": 0.5}
    )


def test_low_dispersion_returns_range(settings):
    result = synthesize_range([40, 45, 50], wrapper=None, settings=settings)
    assert result is not None
    assert result.min == 40.0
    assert result.max == 50.0
    assert result.dispersion < 0.5


def test_high_dispersion_is_discarded_as_contradiction(settings):
    result = synthesize_range([10, 200], wrapper=None, settings=settings)
    assert result is None


def test_single_value_returns_none(settings):
    assert synthesize_range([40], wrapper=None, settings=settings) is None


def test_empty_returns_none(settings):
    assert synthesize_range([], wrapper=None, settings=settings) is None
