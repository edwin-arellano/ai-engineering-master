"""Parent-child: indexa el presupuesto completo (padre) y cada componente (hijo),
con el hijo referenciando al padre vía parent_chunk_id. "Contextual retrieval barato"
sin LLM. Útil para textos donde el contexto del padre importa.
"""

from __future__ import annotations

from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk


class HierarchicalChunker(Chunker):
    name = "hierarchical"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            parent_id = f"{budget.budget_id}::parent"
            parent_text = parent + "\n\n" + "\n\n".join(
                common.render_component_text(c, "") for c in budget.components
            )
            ptok = common.count_tokens(parent_text)
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    metadata={
                        "budget_id": budget.budget_id,
                        "level": "parent",
                        "strategy": self.name,
                    },
                    token_count=ptok,
                    is_orphan=False,
                )
            )
            for c in budget.components:
                text = common.render_component_text(c, parent)
                tokens = common.count_tokens(text)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{c.component_id}::child",
                        text=text,
                        metadata=common.build_metadata(
                            c,
                            budget,
                            strategy=self.name,
                            level="child",
                            parent_chunk_id=parent_id,
                        ),
                        token_count=tokens,
                        is_orphan=common.is_orphan(tokens),
                    )
                )
        return chunks
