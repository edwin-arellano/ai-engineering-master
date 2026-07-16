"""Fase 2a del flujo híbrido (S12): marcar las tareas que el retrieval determinista no cerró.

NO usa LLM, y es la pieza que hace barata la fase 2: decide DÓNDE merece la pena gastar un
agente. Sin este filtro habría que correr el bucle sobre todas las tareas, cuando la mayoría
ya las resolvió bien el retrieval determinista (que es barato y reproducible).

Tres motivos de flag, en orden de gravedad:
- sin match histórico (no hay de dónde derivar las horas),
- fiabilidad baja (vecinos lejanos: hay dato, pero no se parece),
- fuentes en conflicto (los vecinos discrepan tanto entre sí que su mediana no significa nada).

Reutiliza el `Reliability` de `per_task` y el coeficiente de variación de la síntesis (S11):
la misma vara que ya usa el resto del sistema para medir discrepancia.
"""

from __future__ import annotations

import structlog

from app.domain.structured_estimation import Reliability, TaskEstimate
from app.foundations.config import Settings
from app.generation.rag.quality.synthesis import coefficient_of_variation

logger = structlog.get_logger(__name__)


def _flag_reason(task: TaskEstimate, settings: Settings) -> str | None:
    """Motivo por el que esta tarea necesita una segunda mirada, o None si no la necesita."""
    if task.suggested_hours is None or task.reliability is Reliability.NONE:
        return "no historical match within distance threshold"
    if task.reliability is Reliability.LOW:
        return "low reliability (no close neighbors)"
    hours = [n.estimated_hours for n in task.neighbors]
    if len(hours) >= 2:
        # Con un solo vecino no hay discrepancia posible; con ninguno ya salió arriba.
        cv = coefficient_of_variation(hours)
        if cv >= settings.agent_flag_dispersion_threshold:
            return (
                f"conflicting sources ({len(hours)} neighbors between "
                f"{min(hours):g}h and {max(hours):g}h, dispersion {cv:.2f})"
            )
    return None


def flag_task_estimates(tasks: list[TaskEstimate], settings: Settings) -> list[TaskEstimate]:
    """Puebla `flag_reason` en cada tarea (in-place) y devuelve la misma lista.

    Idempotente: reescribe el flag de cada tarea desde su estado actual, así que volver a
    pasarlo tras la recuperación recalcula los flags en vez de acumularlos.
    """
    for task in tasks:
        task.flag_reason = _flag_reason(task, settings)
    flagged = sum(1 for t in tasks if t.flag_reason is not None)
    logger.info("agent.flagging.done", total=len(tasks), flagged=flagged)
    return tasks
