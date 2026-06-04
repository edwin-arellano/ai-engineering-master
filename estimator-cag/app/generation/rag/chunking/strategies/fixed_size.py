from __future__ import annotations

from app.foundations.config import get_settings
from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk


class FixedSizeChunker(Chunker):
    """Trozos de tamaño fijo con overlap. Mecánica, ruidosa: solo como baseline."""

    name = "fixed_size"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        s = get_settings()
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                text = common.render_component_text(c, parent)
                for i, piece in enumerate(
                    common.split_by_tokens(
                        text, s.chunk_max_tokens, s.chunk_overlap_tokens
                    )
                ):
                    tokens = common.count_tokens(piece)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{c.component_id}::fixed::{i}",
                            text=piece,
                            metadata=common.build_metadata(
                                c, budget, strategy=self.name
                            ),
                            token_count=tokens,
                            is_orphan=common.is_orphan(tokens),
                        )
                    )
        return chunks
