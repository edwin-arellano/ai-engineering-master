"""Tools del agente de estimación (S12) para la Responses API de OpenAI.

- search_budgets: envuelve el retrieval híbrido+rerank de S9–S11 (RetrievalPipeline). Es
  la ÚNICA tool con efectos de I/O (solo lectura sobre pgvector). Devuelve una observación
  de alto valor: nº de matches + mediana de horas + confianza + items compactos.
- calculate_estimate: coste determinista por mediana + buffer de contingencia. Sin LLM.
- validate_estimate (extensión opcional): guardrails deterministas (coherencia de total,
  componentes sin referencia, rango razonable). Reutiliza el ESPÍRITU de S4/S11, no el gate
  de citación de S11 (que exige evidencia por línea que la salida del agente no lleva).

Los schemas son PLANOS (type/name/description/parameters al mismo nivel) — Responses API,
NO Chat Completions. `filters` se aplana a campos top-level anulables por strict-mode.
El agente solo ve estos schemas: la calidad de las descripciones gobierna su comportamiento.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from statistics import median
from typing import Any

import structlog

from app.foundations.config import Settings
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.schemas import MetadataFilters, ReformulatedQuery, SearchTarget

logger = structlog.get_logger(__name__)

ToolFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Schemas (Responses API, plano + strict). Descripciones EN INGLÉS a propósito:
# son lo único que el modelo lee para decidir cuándo y cómo usar cada tool.
# ---------------------------------------------------------------------------


def _search_budgets_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "search_budgets",
        "description": (
            "Search historical project budgets for items comparable to ONE software "
            "component. Call this once per component; keep unrelated components (for "
            "example, an ERP integration and a mobile app) in separate calls. Returns "
            "historical items with recorded engineer-hours; use their hours as "
            "reference_amounts for calculate_estimate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused, self-contained description of ONE component to price "
                        "(what it does, key technologies). One component only."
                    ),
                },
                "component_type": {
                    "type": "string",
                    "enum": ["backend", "integration", "mobile", "analytics", "frontend"],
                    "description": (
                        "Category of the component. Steers ranking; it is not a hard filter."
                    ),
                },
                "year_min": {
                    "type": ["integer", "null"],
                    "description": "Oldest budget year to consider (inclusive), or null.",
                },
                "year_max": {
                    "type": ["integer", "null"],
                    "description": "Newest budget year to consider (inclusive), or null.",
                },
            },
            "required": ["query", "component_type", "year_min", "year_max"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _calculate_estimate_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "calculate_estimate",
        "description": (
            "Deterministically cost each component from its historical reference amounts "
            "(engineer-hours) and return the per-component breakdown and the total. Call "
            "once, after gathering references for every component. It does not call a model "
            "and does not invent numbers: a component with no references is costed at 0 and "
            "flagged as unbudgeted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "description": "One entry per component to cost.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Component name."},
                            "reference_amounts": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": (
                                    "Historical engineer-hour amounts from search_budgets "
                                    "for this component. Empty list if none were found."
                                ),
                            },
                        },
                        "required": ["name", "reference_amounts"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["components"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _validate_estimate_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "validate_estimate",
        "description": (
            "Run deterministic sanity checks on a candidate estimate BEFORE finalizing: the "
            "total must match the sum of components, components without any historical "
            "reference are flagged, and per-component hours must fall in a plausible range. "
            "Call this as the LAST step before producing the final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "estimated_hours": {"type": "number"},
                            "reference_budget_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["name", "estimated_hours", "reference_budget_ids"],
                        "additionalProperties": False,
                    },
                },
                "total_hours": {"type": "number"},
            },
            "required": ["components", "total_hours"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def build_tools(settings: Settings) -> list[dict[str, Any]]:
    """Lista de schemas para la Responses API. validate_estimate es opt-in (toggle)."""
    tools = [_search_budgets_schema(), _calculate_estimate_schema()]
    if settings.agent_validate_enabled:
        tools.append(_validate_estimate_schema())
    return tools


# ---------------------------------------------------------------------------
# Implementaciones. calculate/validate son deterministas (async por uniformidad
# del registro; sin await interno).
# ---------------------------------------------------------------------------


def _confidence(items: list[dict[str, Any]], settings: Settings) -> str:
    """Confianza por nº de vecinos cercanos (misma heurística que per_task._reliability)."""
    close = [i for i in items if i["distance"] <= settings.per_task_close_distance]
    if len(close) >= settings.per_task_min_neighbors_high:
        return "high"
    if close:
        return "medium"
    return "low"


def _make_search_budgets(pipeline: RetrievalPipeline, settings: Settings) -> ToolFn:
    async def search_budgets(args: dict[str, Any]) -> dict[str, Any]:
        query: str = args["query"]
        component_type = args.get("component_type")
        year_min = args.get("year_min")
        year_max = args.get("year_max")
        # component_type NO es filtro duro (no hay eje en el corpus): pista de ranking.
        search_text = query if not component_type else f"{query} ({component_type})"
        reformulated = ReformulatedQuery(
            project_function=query[:200],
            technologies=[],
            sector="other",
            scale="small",
            countries=[],
            constraints=[],
            search_text=search_text,
        )
        # sector="other" es placeholder: NO se usa como filtro porque pasamos `filters`
        # explícitos (que sobrescriben el default budget-céntrico del pipeline).
        filters = MetadataFilters(
            year_min=year_min,
            year_max=year_max,
            chunk_types=["budget_component", "historical_task"],
        )
        retrieval = await pipeline.retrieve(
            reformulated=reformulated,
            settings=settings,
            search_mode=settings.agent_search_mode,
            reranking=settings.agent_reranking,
            routing=False,
            query_transform=False,
            temporal_decay=settings.temporal_decay_enabled,
            explicit_targets=[SearchTarget.BUDGETS],
            filters=filters,
        )
        items: list[dict[str, Any]] = []
        for chunk in retrieval.chunks[: settings.agent_search_top_k]:
            hours = chunk.metadata.get("estimated_hours")
            if hours is None:
                continue
            items.append(
                {
                    "budget_id": str(chunk.metadata.get("budget_id", "")),
                    "chunk_ref": chunk.chunk_ref,
                    "content_preview": chunk.content[:180],
                    "sector": str(chunk.metadata.get("client_sector", "")),
                    "year": chunk.metadata.get("year"),
                    "estimated_hours": float(hours),
                    "distance": chunk.distance,
                }
            )
        if not items:
            # Error informativo COMO observación: el modelo puede reformular y reintentar.
            return {
                "matches": 0,
                "median_hours": None,
                "confidence": "none",
                "note": "no historical matches; reformulate the query with other terms",
                "items": [],
            }
        med = round(median([i["estimated_hours"] for i in items]), 1)
        return {
            "matches": len(items),
            "median_hours": med,
            "confidence": _confidence(items, settings),
            "items": items,
        }

    return search_budgets


def _make_calculate_estimate(settings: Settings) -> ToolFn:
    contingency = settings.agent_contingency_factor

    async def calculate_estimate(args: dict[str, Any]) -> dict[str, Any]:
        """Coste por mediana (robusta a un outlier) + buffer plano; sin inventar cifras."""
        breakdown: list[dict[str, Any]] = []
        total = 0.0
        for component in args["components"]:
            name = component["name"]
            refs = component.get("reference_amounts", [])
            if refs:
                central = median(refs)
                hours = round(central * (1 + contingency), 1)
                unbudgeted = False
            else:
                hours = 0.0
                unbudgeted = True
            total += hours
            breakdown.append(
                {
                    "name": name,
                    "reference_count": len(refs),
                    "estimated_hours": hours,
                    "unbudgeted": unbudgeted,
                }
            )
        total = round(total, 1)
        return {
            "components": breakdown,
            "total_hours": total,
            "summary": f"total={total}h across {len(breakdown)} components",
        }

    return calculate_estimate


def _make_validate_estimate(settings: Settings) -> ToolFn:
    tol = settings.agent_validate_tolerance_hours
    ceiling = settings.agent_validate_max_component_hours

    async def validate_estimate(args: dict[str, Any]) -> dict[str, Any]:
        """Guardrails deterministas de S4/S11 sobre la estimación candidata."""
        components = args["components"]
        total = round(float(args["total_hours"]), 1)
        issues: list[str] = []
        summed = round(sum(float(c["estimated_hours"]) for c in components), 1)
        if abs(summed - total) > tol:
            issues.append(
                f"total_hours ({total}) no cuadra con la suma de componentes ({summed})"
            )
        for c in components:
            hours = float(c["estimated_hours"])
            if not c.get("reference_budget_ids"):
                issues.append(f"'{c['name']}' sin presupuesto de referencia (unbudgeted)")
            if hours < 0:
                issues.append(f"'{c['name']}' con horas negativas ({hours})")
            elif hours > ceiling:
                issues.append(
                    f"'{c['name']}' fuera de rango razonable ({hours}h > {ceiling}h)"
                )
        return {"ok": not issues, "checked": len(components), "issues": issues}

    return validate_estimate


# ---------------------------------------------------------------------------
# Registro + despacho. Desacopla "qué tools existen" de "cómo funciona el bucle".
# ---------------------------------------------------------------------------


def build_tool_registry(
    *, pipeline: RetrievalPipeline | None, settings: Settings, stub: bool = False
) -> dict[str, ToolFn]:
    """Registro nombre→función. Con stub=True, search_budgets usa la red de seguridad
    (reference_retrieval) en vez del retrieval real — para depurar el BUCLE sin BD."""
    if stub:
        from app.generation.agentic.reference_retrieval import search_budgets_stub

        async def search_budgets(args: dict[str, Any]) -> dict[str, Any]:
            hits = search_budgets_stub(
                args["query"],
                {"sectors": None},  # el stub filtra por sectors si se le pasan
            )
            items = [
                {
                    "budget_id": h["budget_id"],
                    "chunk_ref": h["budget_id"],
                    "content_preview": h["content_preview"],
                    "sector": h["sector"],
                    "year": None,
                    "estimated_hours": float(h["estimated_hours"]),
                    "distance": h["distance"],
                }
                for h in hits
            ]
            if not items:
                return {
                    "matches": 0,
                    "median_hours": None,
                    "confidence": "none",
                    "note": "no stub matches; reformulate",
                    "items": [],
                }
            med = round(median([i["estimated_hours"] for i in items]), 1)
            return {
                "matches": len(items),
                "median_hours": med,
                "confidence": _confidence(items, settings),
                "items": items,
            }

        search = search_budgets
    else:
        # Construcción lazy: `pipeline=None` solo revienta si se LLAMA a search_budgets,
        # y ahí execute_tool lo convierte en observación de error. Así los tests de las
        # tools deterministas (calculate/validate) no necesitan BD ni pipeline.
        search = _make_search_budgets(pipeline, settings)

    registry: dict[str, ToolFn] = {
        "search_budgets": search,
        "calculate_estimate": _make_calculate_estimate(settings),
    }
    if settings.agent_validate_enabled:
        registry["validate_estimate"] = _make_validate_estimate(settings)
    return registry


async def execute_tool(
    registry: dict[str, ToolFn], name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """Un fallo de tool NO revienta el bucle: se devuelve como observación de error."""
    fn = registry.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return await fn(args)
    except Exception as exc:  # noqa: BLE001 — el error es una observación recuperable
        logger.warning("agent.tool_failed", tool=name, error=str(exc))
        return {"error": str(exc)}
