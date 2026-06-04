"""Chunking semántico: embebe oraciones consecutivas y corta donde la similitud
coseno cae por debajo del umbral (breakpoint). Reutiliza LiteLLMEmbedder; no añade
sentence-transformers (el directo usó la API; modelo local = optimización futura).
"""

from __future__ import annotations

import math

from app.foundations.config import get_settings
from app.generation.rag.chunking import common
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.schemas import Budget, Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, embedder: LiteLLMEmbedder | None = None) -> None:
        self._embedder = embedder or LiteLLMEmbedder()

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        threshold = get_settings().semantic_breakpoint_threshold
        chunks: list[Chunk] = []
        for budget in budgets:
            parent = common.build_parent_context(budget)
            for c in budget.components:
                sentences = common.split_sentences(
                    common.render_component_text(c, parent)
                )
                if len(sentences) <= 1:
                    chunks.append(self._mk(budget, c, parent, sentences, 0))
                    continue
                vectors = [self._embedder.embed_one(s) for s in sentences]
                groups, current = [], [0]
                for i in range(1, len(sentences)):
                    if _cosine(vectors[i], vectors[i - 1]) < threshold:
                        groups.append(current)
                        current = [i]
                    else:
                        current.append(i)
                groups.append(current)
                for gi, idxs in enumerate(groups):
                    chunks.append(
                        self._mk(budget, c, parent, [sentences[i] for i in idxs], gi)
                    )
        return chunks

    def _mk(self, budget, component, parent, sents, gi) -> Chunk:
        text = " ".join(sents) if sents else parent
        tokens = common.count_tokens(text)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}::sem::{gi}",
            text=text,
            metadata=common.build_metadata(component, budget, strategy=self.name),
            token_count=tokens,
            is_orphan=common.is_orphan(tokens),
        )
