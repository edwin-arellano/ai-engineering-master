from __future__ import annotations

from app.generation.rag.chunking.strategies.contextual_retrieval import (
    ContextualRetrievalChunker,
)
from app.generation.rag.chunking.strategies.fixed_size import FixedSizeChunker
from app.generation.rag.chunking.strategies.hierarchical import HierarchicalChunker
from app.generation.rag.chunking.strategies.historical_task import TaskChunker
from app.generation.rag.chunking.strategies.propositional import PropositionalChunker
from app.generation.rag.chunking.strategies.recursive import RecursiveChunker
from app.generation.rag.chunking.strategies.semantic import SemanticChunker
from app.generation.rag.chunking.strategies.sentence_window import SentenceWindowChunker
from app.generation.rag.chunking.strategies.structural import StructuralChunker

# estrategias sin LLM (instanciables sin dependencias)
MECHANICAL = {
    c.name: c
    for c in (
        StructuralChunker,
        TaskChunker,
        FixedSizeChunker,
        RecursiveChunker,
        SentenceWindowChunker,
        HierarchicalChunker,
    )
}
# estrategias que requieren embedder o wrapper (se construyen con inyección)
SEMANTIC = {SemanticChunker.name: SemanticChunker}
LLM_BASED = {
    PropositionalChunker.name: PropositionalChunker,
    ContextualRetrievalChunker.name: ContextualRetrievalChunker,
}

ALL_STRATEGY_NAMES = sorted({*MECHANICAL, *SEMANTIC, *LLM_BASED})


def build_chunker(name: str, *, embedder=None, wrapper=None):
    if name in MECHANICAL:
        return MECHANICAL[name]()
    if name in SEMANTIC:
        return SemanticChunker(embedder=embedder)
    if name in LLM_BASED:
        return LLM_BASED[name](wrapper=wrapper)
    raise ValueError(f"Estrategia de chunking desconocida: {name}")
