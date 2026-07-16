"""Tests del grafo de estimación (S13).

Unit: nodos y orquestación con dobles (sin BD ni LLM real) — verifica forma del estado,
el reducer acumulador y el cableado de los 5 nodos. Integration: e2e sobre el Postgres
real con checkpointer (requiere infra levantada y OPENAI_API_KEY).
"""

from __future__ import annotations

import operator
from typing import get_type_hints

import pytest

from app.domain.structured_estimation import (
    Coverage,
    EstimateSkeleton,
    EstimatedModule,
    Reliability,
    SkeletonModule,
    SkeletonTask,
    StructuredEstimate,
    TaskEstimate,
)
from app.foundations.config import get_settings
from app.generation.graph import build_graph
from app.generation.graph.deps import GraphDeps
from app.generation.graph.nodes import (
    _consolidate,
    _coverage,
    _needs_review,
    route_after_validation,
)
from app.generation.graph.service import checkpointer_conninfo, run_estimation_graph
from app.generation.graph.state import EstimationState
from app.generation.rag.schemas import ReformulatedQuery


def test_conninfo_strips_asyncpg():
    dsn = "postgresql+asyncpg://u:p@localhost:5433/db"
    assert checkpointer_conninfo(dsn) == "postgresql://u:p@localhost:5433/db"


def test_state_has_accumulator_reducer():
    # El campo task_estimates debe llevar un reducer operator.add (Annotated).
    hints = get_type_hints(EstimationState, include_extras=True)
    meta = getattr(hints["task_estimates"], "__metadata__", ())
    assert operator.add in meta


def test_state_errors_is_also_accumulator():
    hints = get_type_hints(EstimationState, include_extras=True)
    assert operator.add in getattr(hints["errors"], "__metadata__", ())


def test_coverage_counts_history():
    tasks = [
        TaskEstimate(
            title="a",
            suggested_hours=10,
            reliability=Reliability.HIGH,
            needs_human_input=False,
        ),
        TaskEstimate(title="b", suggested_hours=None, needs_human_input=True),
    ]
    cov = _coverage(tasks)
    assert (cov.with_history, cov.without_history, cov.total) == (1, 1, 2)


def test_consolidate_sums_only_known_hours():
    # Las tareas sin horas (needs_human_input) no rompen el total: cuentan como 0.
    modules = [
        EstimatedModule(
            name="m",
            tasks=[
                TaskEstimate(
                    title="a",
                    suggested_hours=8,
                    reliability=Reliability.HIGH,
                    needs_human_input=False,
                ),
                TaskEstimate(title="b", suggested_hours=None),
            ],
        )
    ]
    estimate = _consolidate(modules)
    assert estimate.total_suggested_hours == 8
    assert estimate.coverage.total == 2
    assert estimate.agent_trace is None


def test_needs_review_true_when_flagged():
    est = StructuredEstimate(
        modules=[
            EstimatedModule(
                name="m",
                tasks=[
                    TaskEstimate(title="a", suggested_hours=None, needs_human_input=True),
                ],
            )
        ],
        coverage=Coverage(with_history=0, without_history=1, total=1),
        total_suggested_hours=0.0,
    )
    assert _needs_review(est) is True


def test_needs_review_false_when_everything_resolved():
    est = StructuredEstimate(
        modules=[
            EstimatedModule(
                name="m",
                tasks=[
                    TaskEstimate(
                        title="a",
                        suggested_hours=12,
                        reliability=Reliability.HIGH,
                        needs_human_input=False,
                    ),
                ],
            )
        ],
        coverage=Coverage(with_history=1, without_history=0, total=1),
        total_suggested_hours=12.0,
    )
    assert _needs_review(est) is False


def test_route_after_validation_defaults_to_validated():
    assert route_after_validation({"status": "needs_review"}) == "needs_review"
    assert route_after_validation({}) == "validated"


# --- Orquestación con dobles: sin LLM, sin BD, sin checkpointer -----------------

_FAKE_SKELETON = EstimateSkeleton(
    modules=[
        SkeletonModule(
            name="Auth", tasks=[SkeletonTask(title="Login"), SkeletonTask(title="SSO")]
        ),
        SkeletonModule(name="Catálogo", tasks=[SkeletonTask(title="Buscador")]),
    ]
)


@pytest.fixture
def stub_deps(monkeypatch):
    """GraphDeps con las funciones envueltas sustituidas por dobles. Recovery apagado:
    el grafo corre el camino determinista puro."""
    from app.generation.graph import nodes

    monkeypatch.setattr(
        nodes,
        "reformulate_transcript",
        lambda *, transcript, wrapper, settings: ReformulatedQuery(
            project_function="plataforma logística",
            technologies=[],
            sector="other",
            scale="medium",
            search_text="plataforma logística multi-módulo",
        ),
    )
    monkeypatch.setattr(
        nodes, "generate_skeleton", lambda *, transcript, wrapper, settings: _FAKE_SKELETON
    )

    async def _fake_estimate(task_title, *, pipeline, settings):
        # "SSO" queda sin historia → fuerza el camino needs_review.
        if task_title == "SSO":
            return TaskEstimate(title=task_title, suggested_hours=None)
        return TaskEstimate(
            title=task_title,
            suggested_hours=10,
            reliability=Reliability.HIGH,
            needs_human_input=False,
        )

    monkeypatch.setattr(nodes, "estimate_task_hours", _fake_estimate)

    settings = get_settings().model_copy(update={"agent_recovery_enabled": False})
    return GraphDeps(settings=settings, wrapper=None, pipeline=None, client=None)


async def test_graph_runs_all_nodes_with_stubs(stub_deps):
    """El grafo corre START → … → END sin checkpointer y consolida la estimación."""
    graph = build_graph(None, stub_deps)
    estimate, status = await run_estimation_graph(
        graph, transcript="una transcripción de prueba", thread_id="unit-stub"
    )

    # Reagrupación fiel al esqueleto (orden plano alineado).
    assert [m.name for m in estimate.modules] == ["Auth", "Catálogo"]
    assert [t.title for t in estimate.modules[0].tasks] == ["Login", "SSO"]
    # Cobertura y total desde el acumulador: 3 tareas, 2 con historia (10h cada una).
    assert estimate.coverage == Coverage(with_history=2, without_history=1, total=3)
    assert estimate.total_suggested_hours == 20
    # SSO quedó sin horas → revisión pendiente. Sin recovery, no hay traza.
    assert status == "needs_review"
    assert estimate.agent_trace is None


async def test_accumulator_collects_one_estimate_per_skeleton_task(stub_deps):
    """El reducer acumulador recoge exactamente una TaskEstimate por tarea del esqueleto."""
    graph = build_graph(None, stub_deps)
    state = await graph.ainvoke(
        {"transcript": "t", "task_estimates": [], "errors": []},
        {"configurable": {"thread_id": "unit-acc"}},
    )
    expected = sum(len(m.tasks) for m in _FAKE_SKELETON.modules)
    assert len(state["task_estimates"]) == expected
    assert [t.title for t in state["task_estimates"]] == ["Login", "SSO", "Buscador"]


async def test_conditional_validation_graph_also_reaches_end(stub_deps):
    """Nivel 3: con arista condicional el grafo termina igual y conserva el status."""
    graph = build_graph(None, stub_deps, conditional_validation=True)
    estimate, status = await run_estimation_graph(
        graph, transcript="t", thread_id="unit-cond"
    )
    assert status == "needs_review"
    assert estimate.coverage.total == 3


# --- Integration: infra real ---------------------------------------------------


@pytest.mark.integration
async def test_graph_end_to_end(sample_transcript_complex):
    """E2E: compila el grafo con checkpointer real y corre sobre la transcripción
    compleja. Requiere Postgres del proyecto levantado (docker-compose) y OPENAI_API_KEY.
    Genera además la traza Logfire (un span por nodo) que exige el ejercicio."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from app.generation.graph import build_deps

    settings = get_settings()
    conninfo = checkpointer_conninfo(settings.database_url)
    async with AsyncPostgresSaver.from_conn_string(conninfo) as cp:
        await cp.setup()
        graph = build_graph(cp, build_deps(settings))
        estimate, status = await run_estimation_graph(
            graph, transcript=sample_transcript_complex, thread_id="test-s13-e2e"
        )
        # El checkpoint de la ejecución quedó persistido bajo su thread_id.
        tuple_ = await cp.aget_tuple({"configurable": {"thread_id": "test-s13-e2e"}})

    assert status in {"validated", "needs_review"}
    assert estimate.coverage.total == sum(len(m.tasks) for m in estimate.modules)
    assert tuple_ is not None
