"""Comparador de estrategias (sin API: solo estrategias mecánicas, sin query)."""

from __future__ import annotations

from app.generation.rag.comparison import StrategyReport, _p, compare_strategies
from app.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata


def _budget() -> Budget:
    return Budget(
        budget_id="BUD-2024-001",
        client_metadata=ClientMetadata(name="Acme", sector="finance", country="ES"),
        project_summary="Mobile banking API",
        main_technology="python",
        year=2024,
        total_estimated_hours=100,
        components=[
            BudgetComponent(
                component_id=f"C{i}",
                name=f"Component {i}",
                description="A self-contained component with a clear scope.",
                tech_stack=["python"],
                estimated_hours=40,
                complexity="medium",
            )
            for i in range(3)
        ],
    )


def test_percentile_helper():
    assert _p([], 0.95) == 0.0
    assert _p([10, 20, 30, 40], 0.5) == 30.0


def test_compare_without_query_reports_counts_and_percentiles():
    reports = compare_strategies(
        [_budget()], ["structural", "hierarchical"], query=None
    )
    assert all(isinstance(r, StrategyReport) for r in reports)
    by_name = {r.name: r for r in reports}

    # structural: un chunk por componente (3), sin query -> sin top_scores
    assert by_name["structural"].num_chunks == 3
    assert by_name["structural"].top_scores == []
    assert by_name["structural"].p50_tokens > 0

    # hierarchical: 1 padre + 3 hijos = 4
    assert by_name["hierarchical"].num_chunks == 4
    assert by_name["hierarchical"].p95_tokens >= by_name["hierarchical"].p50_tokens


def test_compare_counts_orphans():
    # estructural sobre componentes de tamaño normal -> ningún huérfano
    reports = compare_strategies([_budget()], ["structural"], query=None)
    assert reports[0].orphan_count == 0
