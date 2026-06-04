"""Estrategias que pegan al LLM/API (semantic, propositional, contextual_retrieval).

Fijan el patrón real del loader de prompts y del wrapper. Marcadas integration:
no corren en la suite por defecto (coste y red).
"""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.chunking.strategies.contextual_retrieval import (
    ContextualRetrievalChunker,
)
from app.generation.rag.chunking.strategies.propositional import PropositionalChunker
from app.generation.rag.chunking.strategies.semantic import SemanticChunker
from app.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata


def _budget() -> Budget:
    return Budget(
        budget_id="BUD-2024-001",
        client_metadata=ClientMetadata(name="Acme", sector="finance", country="ES"),
        project_summary="Mobile banking API",
        main_technology="python",
        year=2024,
        total_estimated_hours=120,
        components=[
            BudgetComponent(
                component_id="AUTH-001",
                name="OAuth backend",
                description=(
                    "Implements OAuth 2.0 flows. Validates JWT tokens. "
                    "Handles refresh and rotation of signing keys."
                ),
                tech_stack=["python", "redis"],
                estimated_hours=120,
                complexity="high",
            )
        ],
    )


@pytest.mark.integration
def test_semantic_produces_chunks():
    chunks = SemanticChunker().chunk([_budget()])
    assert chunks
    assert all(c.metadata["strategy"] == "semantic" for c in chunks)


@pytest.mark.integration
def test_propositional_generates_propositions():
    wrapper = LLMWrapper(get_settings())
    chunks = PropositionalChunker(wrapper=wrapper).chunk([_budget()])
    assert len(chunks) >= 1
    assert all(c.metadata["strategy"] == "propositional" for c in chunks)


@pytest.mark.integration
def test_contextual_retrieval_prepends_context():
    wrapper = LLMWrapper(get_settings())
    chunks = ContextualRetrievalChunker(wrapper=wrapper).chunk([_budget()])
    assert len(chunks) == 1  # un chunk por componente
    # el contexto generado se guarda en metadata y se antepone al texto base
    assert "generated_context" in chunks[0].metadata
    assert chunks[0].metadata["generated_context"] in chunks[0].text
