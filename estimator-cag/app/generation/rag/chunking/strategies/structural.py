"""Chunking estructural de presupuestos JSON. Granularidad: un componente = un
chunk. El texto combina los detalles del componente con CONTEXTUAL CHUNK HEADERS
del presupuesto padre (sector, año, tecnología, summary) — la palanca de mayor
ROI conocida en RAG (versión estática y barata de Contextual Retrieval, sin LLM).
La metadata filtrable va FUERA del texto embebido.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog
import tiktoken

from app.generation.rag.schemas import Budget, BudgetComponent, Chunk

logger = structlog.get_logger(__name__)

# Umbral de aviso: un chunk por encima de esto es candidato a discutir en directo.
# NO se parte (el ejercicio pide no hacer splitting de descripciones largas).
LONG_CHUNK_WARN_TOKENS = 512


def _get_tokenizer(model: str) -> "tiktoken.Encoding":
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # los modelos de embeddings de OpenAI usan cl100k_base
        return tiktoken.get_encoding("cl100k_base")


class Chunker(ABC):
    """Interfaz común para cualquier estrategia de chunking del pipeline.

    El ejercicio pre-sesión implementa JSONStructuralChunker; el
    TopicSegmentationChunker (transcripciones) y otros entran en el directo
    sobre esta misma base.
    """

    @abstractmethod
    def chunk(self, budgets: list[Budget]) -> list[Chunk]: ...


class JSONStructuralChunker(Chunker):
    def __init__(self, model_for_token_count: str = "text-embedding-3-small") -> None:
        self._tokenizer = _get_tokenizer(model_for_token_count)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            produced = self._chunk_one_budget(budget)
            chunks.extend(produced)
            logger.info(
                "chunk.budget_done", budget_id=budget.budget_id, chunks=len(produced)
            )
        return chunks

    def _chunk_one_budget(self, budget: Budget) -> list[Chunk]:
        parent_context = self._build_parent_context(budget)
        return [
            self._build_chunk(c, budget, parent_context) for c in budget.components
        ]

    def _build_parent_context(self, budget: Budget) -> str:
        c = budget.client_metadata
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {c.sector} | Year: {budget.year} | "
            f"Main tech: {budget.main_technology}]"
        )

    def _render_component_text(
        self, component: BudgetComponent, parent_context: str
    ) -> str:
        return (
            f"{parent_context}\n\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )

    def _build_metadata(
        self, component: BudgetComponent, budget: Budget
    ) -> dict[str, Any]:
        return {
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "client_sector": budget.client_metadata.sector,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity,
            "estimated_hours": component.estimated_hours,
        }

    def _build_chunk(
        self, component: BudgetComponent, budget: Budget, parent_context: str
    ) -> Chunk:
        text = self._render_component_text(component, parent_context)
        token_count = len(self._tokenizer.encode(text))
        if token_count > LONG_CHUNK_WARN_TOKENS:
            logger.warning(
                "chunk.unusually_large",
                chunk_id=f"{budget.budget_id}::{component.component_id}",
                token_count=token_count,
            )
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=self._build_metadata(component, budget),
            token_count=token_count,
        )
