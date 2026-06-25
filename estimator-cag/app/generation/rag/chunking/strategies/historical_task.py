"""Chunking de tareas atómicas (historical_task). Granularidad fina: una tarea
por componente del presupuesto, con contextual headers del padre. Complementa
al StructuralChunker (un chunk por componente completo): el mismo dato indexado
de dos formas distintas mejora el retrieval (overview vs detalle)."""

from __future__ import annotations

import structlog

from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, Chunk

logger = structlog.get_logger(__name__)


class TaskChunker(Chunker):
    name = "historical_task"

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                text = common.render_task_text(c, parent)
                tokens = common.count_tokens(text)
                metadata = common.build_metadata(c, budget, strategy=self.name)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{c.component_id}::task",
                        text=text,
                        metadata=metadata,
                        token_count=tokens,
                        is_orphan=common.is_orphan(tokens),
                    )
                )
        return chunks
