"""Juez de alucinaciones (S11) — integración: pega al LLM real (alias reformulator).
Marcado integration; no corre en la suite por defecto."""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.quality.judge import judge_lines


@pytest.mark.integration
async def test_judge_flags_unsupported_line():
    settings = get_settings()
    wrapper = LLMWrapper(settings)
    lines = [
        # Evidencia dice 120 horas (≈15 días) pero la línea afirma 40 días → no soportada.
        {"index": 0, "title": "Auth backend", "engineer_days": 40.0, "evidence": "Estimated hours: 120"},
        # Coherente: 90 horas ≈ 11.25 días.
        {"index": 1, "title": "PSD2 module", "engineer_days": 11.25, "evidence": "Estimated hours: 90"},
    ]
    verdicts = await judge_lines(lines=lines, wrapper=wrapper, settings=settings)
    assert set(verdicts) == {0, 1}
    assert verdicts[0].supported is False  # el juez detecta la incoherencia de effort
