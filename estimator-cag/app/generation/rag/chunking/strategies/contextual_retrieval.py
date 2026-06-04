"""Contextual Retrieval (técnica de Anthropic): a cada chunk estructural se le
antepone contexto del documento padre generado por LLM. La favorita de Antonio:
muy efectiva en retrieval, pero LLM-por-chunk (cara y lenta).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.foundations.config import get_settings
from app.foundations.prompts.loader import render_contextual_retrieval_prompt
from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk


class ChunkContext(BaseModel):
    context: str


class ContextualRetrievalChunker(Chunker):
    name = "contextual_retrieval"

    def __init__(self, wrapper) -> None:
        self._wrapper = wrapper

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        version = get_settings().contextual_retrieval_prompt_version
        system = render_contextual_retrieval_prompt(version)
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            doc = budget.model_dump_json()
            for c in budget.components:
                base_text = common.render_component_text(c, parent)
                ctx = self._wrapper.complete_structured(
                    system_prompt=system,
                    user_message=f"<document>{doc}</document>\n<chunk>{base_text}</chunk>",
                    response_model=ChunkContext,
                    max_tokens=300,
                    temperature=0.0,
                )
                text = f"{ctx.context}\n\n{base_text}"
                tokens = common.count_tokens(text)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{c.component_id}::ctx",
                        text=text,
                        metadata=common.build_metadata(
                            c, budget, strategy=self.name, generated_context=ctx.context
                        ),
                        token_count=tokens,
                        is_orphan=common.is_orphan(tokens),
                    )
                )
        return chunks
