"""Flagging determinista de tareas dudosas (S12, fase 2a). Sin LLM, sin BD.

Es la pieza que decide dónde entra el agente, así que sus fronteras importan: una tarea
flaggeada de más cuesta un bucle; una de menos deja pasar un número malo sin revisar.
"""

from __future__ import annotations

import pytest

from app.domain.structured_estimation import Reliability, TaskEstimate, TaskNeighbor
from app.foundations.config import get_settings
from app.generation.agentic.flagging import flag_task_estimates


@pytest.fixture
def settings():
    return get_settings().model_copy(update={"agent_flag_dispersion_threshold": 0.35})


def _task(
    *,
    hours: float | None,
    reliability: Reliability,
    neighbor_hours: list[float] | None = None,
) -> TaskEstimate:
    neighbors = [
        TaskNeighbor(
            budget_id=f"BUD-{i}", chunk_ref=f"ref-{i}", estimated_hours=h, distance=0.2
        )
        for i, h in enumerate(neighbor_hours or [])
    ]
    return TaskEstimate(
        title="Sincronizar catálogo con el ERP",
        suggested_hours=hours,
        reliability=reliability,
        neighbors=neighbors,
        needs_human_input=hours is None,
    )


def test_sin_match_historico_se_flagea(settings):
    task = _task(hours=None, reliability=Reliability.NONE)
    flag_task_estimates([task], settings)
    assert task.flag_reason == "no historical match within distance threshold"


def test_fiabilidad_baja_se_flagea(settings):
    task = _task(hours=30.0, reliability=Reliability.LOW, neighbor_hours=[28.0, 32.0])
    flag_task_estimates([task], settings)
    assert task.flag_reason == "low reliability (no close neighbors)"


def test_fuentes_en_conflicto_se_flagean(settings):
    # El caso del directo: 16h en una fuente y 60h en otra (cv ≈ 0.58) no promedian a nada
    # defendible, aunque la fiabilidad por cercanía sea alta.
    task = _task(hours=38.0, reliability=Reliability.HIGH, neighbor_hours=[16.0, 60.0])
    flag_task_estimates([task], settings)
    assert task.flag_reason is not None
    assert "conflicting sources" in task.flag_reason


def test_fuentes_consistentes_no_se_flagean(settings):
    # cv ≈ 0.09: los vecinos concuerdan, el determinista ya la resolvió. El agente no entra.
    task = _task(hours=45.0, reliability=Reliability.HIGH, neighbor_hours=[40.0, 45.0, 50.0])
    flag_task_estimates([task], settings)
    assert task.flag_reason is None


def test_un_solo_vecino_no_es_conflicto(settings):
    # Con una única fuente no hay discrepancia que medir; la debilidad la captura reliability.
    task = _task(hours=45.0, reliability=Reliability.MEDIUM, neighbor_hours=[45.0])
    flag_task_estimates([task], settings)
    assert task.flag_reason is None


def test_umbral_de_dispersion_es_configurable(settings):
    task = _task(hours=45.0, reliability=Reliability.HIGH, neighbor_hours=[40.0, 45.0, 50.0])
    strict = settings.model_copy(update={"agent_flag_dispersion_threshold": 0.05})
    flag_task_estimates([task], strict)
    assert task.flag_reason is not None and "conflicting sources" in task.flag_reason


def test_flagging_es_idempotente(settings):
    # Se vuelve a pasar sobre tareas ya flaggeadas (p. ej. tras la recuperación): recalcula
    # desde el estado actual en vez de acumular.
    ok = _task(hours=45.0, reliability=Reliability.HIGH, neighbor_hours=[40.0, 45.0, 50.0])
    bad = _task(hours=None, reliability=Reliability.NONE)
    flag_task_estimates([ok, bad], settings)
    first = (ok.flag_reason, bad.flag_reason)
    flag_task_estimates([ok, bad], settings)
    assert (ok.flag_reason, bad.flag_reason) == first


def test_solo_se_flagean_las_dudosas(settings):
    tasks = [
        _task(hours=45.0, reliability=Reliability.HIGH, neighbor_hours=[40.0, 45.0, 50.0]),
        _task(hours=None, reliability=Reliability.NONE),
        _task(hours=30.0, reliability=Reliability.LOW, neighbor_hours=[30.0, 30.0]),
    ]
    flag_task_estimates(tasks, settings)
    assert [t.flag_reason is not None for t in tasks] == [False, True, True]
