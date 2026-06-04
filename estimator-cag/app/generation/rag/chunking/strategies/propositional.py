"""Descompone cada componente en proposiciones atómicas vía LLM. Precisa pero cara
y propensa a huérfanos (Antonio no la recomienda; se incluye para la comparación).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.foundations.config import get_settings
from app.foundations.prompts.loader import render_propositional_prompt
from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk


class PropositionList(BaseModel):
    propositions: list[str]


class PropositionalChunker(Chunker):
    name = "propositional"

    def __init__(self, wrapper) -> None:  # wrapper LLM inyectado (mismo patrón que cag)
        self._wrapper = wrapper

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        version = get_settings().propositional_prompt_version
        system = render_propositional_prompt(version)
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                text = common.render_component_text(c, parent)
                result = self._wrapper.complete_structured(
                    system_prompt=system,
                    user_message=text,
                    response_model=PropositionList,
                    max_tokens=1500,
                    temperature=0.0,
                )
                for i, prop in enumerate(result.propositions):
                    tokens = common.count_tokens(prop)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{c.component_id}::prop::{i}",
                            text=prop,
                            metadata=common.build_metadata(
                                c, budget, strategy=self.name
                            ),
                            token_count=tokens,
                            is_orphan=common.is_orphan(tokens),
                        )
                    )
        return chunks
