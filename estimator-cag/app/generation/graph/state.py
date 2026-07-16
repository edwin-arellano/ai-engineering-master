"""Estado compartido del grafo de estimación (S13).

Identificadores de nodo = enunciado del curso; campos del estado = vocabulario de
dominio del repo. `task_estimates` es el reducer ACUMULADOR: en secuencial lo puebla
un solo nodo, pero con `operator.add` queda listo para el fan-out por componente
(Send API) que llega en el directo. `errors` es un segundo acumulador de diagnóstico.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.domain.agent_trace import AgentTrace
from app.domain.structured_estimation import (
    EstimateSkeleton,
    StructuredEstimate,
    TaskEstimate,
)
from app.generation.rag.schemas import ReformulatedQuery


class EstimationState(TypedDict, total=False):
    # Entrada (requerido en la invocación).
    transcript: str
    # extract_requirements → provenance de la reformulación S9 (contexto, no consumido
    # por el camino determinista por-tarea, que reconstruye su query desde el título).
    reformulated: ReformulatedQuery
    # classify_components → esqueleto módulos/tareas (los "components" del enunciado).
    skeleton: EstimateSkeleton
    # search_budgets → ACUMULADOR: una TaskEstimate por tarea del esqueleto (orden plano
    # alineado con el recorrido del skeleton).
    task_estimates: Annotated[list[TaskEstimate], operator.add]
    # generate_estimate / validate_and_consolidate → estimación consolidada.
    estimate: StructuredEstimate
    # validate_and_consolidate → traza de la recuperación agéntica (None si no corrió).
    agent_trace: AgentTrace | None
    # validate_and_consolidate → "validated" | "needs_review".
    status: str
    # Diagnóstico acumulado.
    errors: Annotated[list[str], operator.add]
