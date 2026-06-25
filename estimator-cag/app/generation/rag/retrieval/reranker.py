"""Reranker cross-encoder (BAAI/bge-reranker-v2-m3, multilingüe). Lee consulta y
candidato JUNTOS y puntúa relevancia del par — corrige el mal ordenamiento del
bi-encoder. Modelo cargado UNA vez (singleton de ciclo de vida); la inferencia es
cómputo local y debe despacharse a un thread fuera del event loop."""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from app.foundations.config import get_settings
from app.generation.rag.schemas import RetrievedChunk

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder reranker para candidatos recuperados."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().reranker_model_name
        # Carga costosa (segundos + memoria permanente): hacerla una vez, en construcción.
        self._model = CrossEncoder(self._model_name)
        logger.info("reranker_loaded", model=self._model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Puntúa cada par (query, candidate.content) conjuntamente y devuelve los top_k.
        Entra y sale el MISMO tipo → etapa opcional y componible del pipeline."""
        if not candidates:
            return []
        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        logger.info(
            "rerank_completed",
            candidates_in=len(candidates),
            candidates_out=min(top_k, len(ranked)),
            model=self._model_name,
        )
        return [candidate for candidate, _ in ranked[:top_k]]


# Singleton de ciclo de vida: una instancia compartida entre peticiones.
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
