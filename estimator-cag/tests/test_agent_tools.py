"""Tests deterministas de las tools del agente (S12): sin LLM, sin BD."""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.generation.agentic.agent_tools import (
    build_tool_registry,
    build_tools,
    execute_tool,
)


@pytest.fixture
def settings():
    return get_settings()


async def test_calculate_estimate_mediana_y_contingencia(settings):
    registry = build_tool_registry(pipeline=None, settings=settings)  # calculate no usa pipeline
    out = await execute_tool(
        registry,
        "calculate_estimate",
        {"components": [{"name": "Auth", "reference_amounts": [420.0, 380.0]}]},
    )
    # mediana(420,380)=400; *1.15 = 460.0
    assert out["components"][0]["estimated_hours"] == 460.0
    assert out["components"][0]["unbudgeted"] is False
    assert out["total_hours"] == 460.0


async def test_calculate_estimate_sin_referencias_marca_unbudgeted(settings):
    registry = build_tool_registry(pipeline=None, settings=settings)
    out = await execute_tool(
        registry,
        "calculate_estimate",
        {"components": [{"name": "Panel", "reference_amounts": []}]},
    )
    assert out["components"][0]["unbudgeted"] is True
    assert out["components"][0]["estimated_hours"] == 0.0


async def test_validate_estimate_detecta_incoherencias(settings):
    registry = build_tool_registry(pipeline=None, settings=settings)
    out = await execute_tool(
        registry,
        "validate_estimate",
        {
            "components": [
                {"name": "A", "estimated_hours": 100.0, "reference_budget_ids": ["BUD-1"]},
                {"name": "B", "estimated_hours": 50.0, "reference_budget_ids": []},
            ],
            "total_hours": 999.0,  # no cuadra
        },
    )
    assert out["ok"] is False
    assert any("no cuadra" in i for i in out["issues"])
    assert any("sin presupuesto" in i for i in out["issues"])


async def test_search_budgets_stub_envelope(settings):
    registry = build_tool_registry(pipeline=None, settings=settings, stub=True)
    out = await execute_tool(
        registry,
        "search_budgets",
        {
            "query": "OAuth JWT authentication backend",
            "component_type": "backend",
            "year_min": None,
            "year_max": None,
        },
    )
    assert out["matches"] >= 1
    assert out["median_hours"] is not None
    assert all("estimated_hours" in i for i in out["items"])


async def test_tool_desconocida_devuelve_error_como_observacion(settings):
    """Un nombre de tool inexistente NO revienta el bucle: vuelve como observación."""
    registry = build_tool_registry(pipeline=None, settings=settings)
    out = await execute_tool(registry, "tool_que_no_existe", {})
    assert "error" in out


def test_build_tools_incluye_validate_por_defecto(settings):
    names = {t["name"] for t in build_tools(settings)}
    assert {"search_budgets", "calculate_estimate"} <= names
    assert ("validate_estimate" in names) == settings.agent_validate_enabled
    # schema plano (Responses API), no anidado bajo "function"
    for t in build_tools(settings):
        assert t["type"] == "function" and "name" in t and t["strict"] is True
