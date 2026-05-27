"""Tests de las métricas del stress test.

Cubren, para cada métrica, un caso que pasa, uno que falla y uno límite. Los
snapshots de `MemoryDriftMetric` usan las claves reales que la métrica lee
(`last_summary`/`anchored_facts`/`project_metadata`), que casan 1:1 con el
`SessionDebugResponse` del endpoint debug.
"""

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


def test_latency_budget_passes_under_budget():
    r = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 1200.0})
    assert r.passed and r.score == 1.0


def test_latency_budget_fails_over_budget():
    r = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 9000.0})
    assert not r.passed and r.score == 0.0


def test_latency_budget_boundary_equal():
    # Caso límite: exactamente en el presupuesto → pasa (≤).
    r = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4000.0})
    assert r.passed


def test_cost_budget_passes_under():
    r = CostBudgetMetric(budget_usd=0.05).evaluate({"cost_usd": 0.004})
    assert r.passed and r.score == 1.0


def test_cost_budget_fails_over():
    r = CostBudgetMetric(budget_usd=0.05).evaluate({"cost_usd": 0.12})
    assert not r.passed


def test_cost_budget_boundary_equal():
    # Caso límite: justo en el presupuesto → pasa (≤).
    r = CostBudgetMetric(budget_usd=0.05).evaluate({"cost_usd": 0.05})
    assert r.passed


def test_memory_drift_found_in_metadata():
    # La memoria persistente: project_metadata es donde sobrevive el nombre.
    snapshot = {
        "last_summary": None,
        "anchored_facts": [],
        "project_metadata": {"project_name": "Nimbus"},
    }
    assert MemoryDriftMetric(fact="Nimbus").evaluate(snapshot).passed


def test_memory_drift_found_in_summary_and_anchors():
    snap_summary = {
        "last_summary": "Atlas inventory tool with a 30000 EUR cap",
        "anchored_facts": [],
        "project_metadata": {},
    }
    assert MemoryDriftMetric(fact="30000").evaluate(snap_summary).passed

    snap_anchors = {
        "last_summary": None,
        "anchored_facts": ["SAP integration required"],
        "project_metadata": {},
    }
    assert MemoryDriftMetric(fact="SAP").evaluate(snap_anchors).passed


def test_memory_drift_not_found():
    snapshot = {
        "last_summary": "Generic project",
        "anchored_facts": [],
        "project_metadata": {},
    }
    assert not MemoryDriftMetric(fact="Flutter").evaluate(snapshot).passed


def test_memory_drift_requires_short_searchable_fact():
    # Lección de diseño: una frase no aparece literal; un término corto sí.
    snapshot = {
        "last_summary": "The Nimbus project will ship in Q3",
        "anchored_facts": [],
        "project_metadata": {},
    }
    assert not MemoryDriftMetric(fact="project name: Nimbus").evaluate(snapshot).passed
    assert MemoryDriftMetric(fact="Nimbus").evaluate(snapshot).passed


def test_memory_drift_where_restricts_search():
    # where=("metadata",) no encuentra algo que solo vive en el summary.
    snapshot = {
        "last_summary": "budget 30000",
        "anchored_facts": [],
        "project_metadata": {},
    }
    result = MemoryDriftMetric(fact="30000", where=("metadata",)).evaluate(snapshot)
    assert not result.passed
