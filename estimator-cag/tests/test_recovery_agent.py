"""Fase 2b del flujo híbrido (S12): el agente de recuperación. Pega a la Responses API real.

Con `stub=True` el bucle usa el corpus canned de `reference_retrieval` en vez de pgvector:
lo que se verifica aquí es el BUCLE y el merge, no el retrieval (que ya cubre S10).
"""

from __future__ import annotations

import pytest
from openai import AsyncOpenAI

from app.domain.structured_estimation import (
    EstimatedModule,
    Reliability,
    TaskEstimate,
    TaskNeighbor,
)
from app.foundations.config import get_settings
from app.generation.agentic.recovery_agent import run_task_hours_recovery_agent

pytestmark = pytest.mark.integration


def _modules() -> list[EstimatedModule]:
    """Un módulo con una tarea sana (sin flag) y una flaggeada sin histórico."""
    healthy = TaskEstimate(
        title="Implementar API REST de pedidos",
        suggested_hours=120.0,
        reliability=Reliability.HIGH,
        neighbors=[
            TaskNeighbor(
                budget_id="BUD-1", chunk_ref="ref-1", estimated_hours=120.0, distance=0.1
            )
        ],
        needs_human_input=False,
        flag_reason=None,
    )
    flagged = TaskEstimate(
        title="Autenticación OAuth2 con JWT y refresh tokens",
        suggested_hours=None,
        reliability=Reliability.NONE,
        neighbors=[],
        needs_human_input=True,
        flag_reason="no historical match within distance threshold",
    )
    return [EstimatedModule(name="Backend", tasks=[healthy, flagged])]


async def test_recovery_solo_entra_en_las_flaggeadas():
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    modules = _modules()
    healthy_before = modules[0].tasks[0].model_dump()

    modules, trace = await run_task_hours_recovery_agent(
        modules,
        client=client,
        pipeline=None,  # stub=True ⇒ el bucle no toca la BD
        settings=settings,
        model=settings.agent_debug_model,
        stub=True,
    )

    # La tarea sana no la mira: pasa idéntica.
    assert modules[0].tasks[0].model_dump() == healthy_before
    # La traza solo contiene pasos de la flaggeada, y son búsquedas.
    assert trace.phase == "recovery"
    assert trace.step_count >= 1
    assert all(s.action == "search_budgets" for s in trace.steps)
    assert [s.step for s in trace.steps] == list(range(1, trace.step_count + 1))
    # El stub tiene histórico de OAuth (BUD-AUTH-*): debería recuperarla sin inventar.
    recovered = modules[0].tasks[1]
    assert recovered.suggested_hours is not None
    assert recovered.flag_reason is None


async def test_sin_tareas_flaggeadas_no_llama_al_modelo():
    """El bucle es caro: si el determinista resolvió todo, el agente ni se instancia."""
    settings = get_settings()
    modules = _modules()
    modules[0].tasks[1].flag_reason = None  # ninguna flaggeada

    modules, trace = await run_task_hours_recovery_agent(
        modules,
        client=None,  # si intentara llamar al modelo, reventaría aquí
        pipeline=None,
        settings=settings,
        stub=True,
    )

    assert trace.step_count == 0
    assert trace.phase == "recovery"
