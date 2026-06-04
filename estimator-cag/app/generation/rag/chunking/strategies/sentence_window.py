from __future__ import annotations

from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk

_WINDOW = 2  # oraciones por chunk


class SentenceWindowChunker(Chunker):
    """Ventanas de N oraciones. Alto recall, pero genera muchos huérfanos."""

    name = "sentence_window"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                sentences = common.split_sentences(
                    common.render_component_text(c, parent)
                )
                for i in range(0, len(sentences), _WINDOW):
                    piece = " ".join(sentences[i : i + _WINDOW])
                    tokens = common.count_tokens(piece)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{budget.budget_id}::{c.component_id}::sw::{i // _WINDOW}",
                            text=piece,
                            metadata=common.build_metadata(
                                c, budget, strategy=self.name
                            ),
                            token_count=tokens,
                            is_orphan=common.is_orphan(tokens),
                        )
                    )
        return chunks
