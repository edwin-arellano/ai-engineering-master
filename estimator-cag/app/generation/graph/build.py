"""Cablea y compila el grafo secuencial de estimación (S13). Aristas directas por
defecto; con `conditional_validation=True` (Nivel 3) la salida de validación pasa por
una arista condicional (validated | needs_review)."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.generation.graph.deps import GraphDeps
from app.generation.graph.nodes import make_nodes, route_after_validation
from app.generation.graph.state import EstimationState


def build_graph(checkpointer, deps: GraphDeps, *, conditional_validation: bool = False):
    nodes = make_nodes(deps)
    builder = StateGraph(EstimationState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")

    if conditional_validation:  # Nivel 3 (opcional)
        builder.add_conditional_edges(
            "validate_and_consolidate",
            route_after_validation,
            {"validated": END, "needs_review": END},
        )
    else:
        builder.add_edge("validate_and_consolidate", END)

    return builder.compile(checkpointer=checkpointer)
