"""Fase 1 del flujo híbrido (S12): el agente de estructura. Pega a la Responses API real.

Usa gpt-5-mini (barato) y no necesita BD: esta fase no consulta el store, solo estructura.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openai import AsyncOpenAI

from app.foundations.config import get_settings
from app.generation.agentic.structure_agent import propose_structure

pytestmark = pytest.mark.integration


async def test_propone_estructura_con_traza_de_un_step():
    settings = get_settings()
    transcript = Path("examples/transcripts/sample_transcript_complex.txt").read_text("utf-8")
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    proposal = await propose_structure(
        transcript, client=client, settings=settings, model=settings.agent_debug_model
    )

    # Piezas independientes → módulos separados (criterio del ejercicio).
    assert len(proposal.skeleton.modules) > 1
    assert all(m.tasks for m in proposal.skeleton.modules)
    # Sin tools ⇒ exactamente un step, el del razonamiento que produjo la estructura.
    assert proposal.agent_trace is not None
    assert proposal.agent_trace.step_count == 1
    assert proposal.agent_trace.phase == "structure"
    assert proposal.agent_trace.agent == settings.agent_profile_name
    step = proposal.agent_trace.steps[0]
    assert step.action == "propose_structure"
    assert step.observation["modules"] == len(proposal.skeleton.modules)
