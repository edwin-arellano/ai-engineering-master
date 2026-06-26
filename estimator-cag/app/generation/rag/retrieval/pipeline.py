"""RetrievalPipeline: compone las etapas de recuperación según configuración, en el
orden canónico 'barato-y-excluyente primero, caro-y-fino al final, blando al cierre':

  reformulación (aguas arriba) → routing → transformación de consulta →
  (por sub-consulta × colección: filtros duros → vector + léxica → fusión RRF de ramas) →
  fusión entre sub-consultas (RRF si expansión, interleave si descomposición) →
  rerank → ponderación blanda.

Cada etapa es activable por toggle. Con TODOS los toggles a False y una sola colección
(budgets) el comportamiento replica el de pre-session-10 (vector|hybrid × rerank).

NOTA sobre ids: el chunk_id es la PK por-tabla; NO es único entre colecciones. Para
fusionar sin colisiones se asigna un id sintético por (colección, chunk_id) dentro del
alcance de cada retrieve(); el reranker y la salida usan los objetos RetrievedChunk, no
esos ids, así que la sustitución es interna.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.foundations.config import Settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.collections import COLLECTION_MODELS
from app.generation.rag.persistence.models import BudgetChunkRow, ChunkRow
from app.generation.rag.persistence.repository import search_chunks
from app.generation.rag.retrieval.fulltext_search import FullTextSearcher
from app.generation.rag.retrieval.fusion import (
    interleave_rankings,
    reciprocal_rank_fusion,
)
from app.generation.rag.retrieval.query_transform import transform_query
from app.generation.rag.retrieval.reranker import get_reranker
from app.generation.rag.retrieval.retriever import filters_from_reformulation
from app.generation.rag.retrieval.router import QueryRouter
from app.generation.rag.retrieval.weighting import apply_soft_weighting
from app.generation.rag.schemas import (
    MetadataFilters,
    ReformulatedQuery,
    RetrievalResult,
    RetrievedChunk,
    SearchTarget,
)

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        embedder: LiteLLMEmbedder,
        session_factory: async_sessionmaker,
        wrapper: LLMWrapper | None = None,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        # Solo necesario si se activa routing o query_transform (etapas con LLM).
        self._wrapper = wrapper
        self._fulltext = FullTextSearcher(session_factory)

    async def _vector_search(
        self,
        *,
        model: type[ChunkRow],
        search_text: str,
        k: int,
        filters: MetadataFilters,
        ef_search: int,
    ) -> list[RetrievedChunk]:
        query_vector = await asyncio.to_thread(self._embedder.embed_one, search_text)
        async with self._session_factory() as session:
            rows = await search_chunks(
                session, model=model, query_vector=query_vector, k=k,
                ef_search=ef_search, filters=filters,
            )
        return [
            RetrievedChunk(
                chunk_id=r._mapping["chunk_id"], document_id=r._mapping["document_id"],
                chunk_type=r._mapping["chunk_type"], content=r._mapping["content"],
                distance=round(float(r._mapping["distance"]), 4),
                metadata=r._mapping["metadata"],
            )
            for r in rows
        ]

    def _resolve_targets(
        self,
        *,
        reformulated: ReformulatedQuery,
        settings: Settings,
        routing: bool,
        explicit_targets: list[SearchTarget] | None,
    ) -> tuple[list[SearchTarget], str]:
        """Decide las colecciones destino. Con routing activo usa el router en cascada;
        si no, respeta explicit_targets o cae a BUDGETS (back-compat)."""
        if routing:
            if self._wrapper is None:
                raise ValueError("routing=True requiere un LLMWrapper en el pipeline")
            decision, level = QueryRouter(self._wrapper, settings).route(
                reformulated.search_text, explicit=explicit_targets
            )
            return decision.targets, level
        if explicit_targets:
            return explicit_targets, "explicit"
        return [SearchTarget.BUDGETS], "default"

    def _resolve_subqueries(
        self, *, reformulated: ReformulatedQuery, settings: Settings, query_transform: bool
    ) -> tuple[list[str], str]:
        """Devuelve (sub_queries, technique). Sin transformación → la search_text tal cual."""
        if not query_transform:
            return [reformulated.search_text], "direct"
        if self._wrapper is None:
            raise ValueError("query_transform=True requiere un LLMWrapper en el pipeline")
        result = transform_query(
            reformulated.search_text, wrapper=self._wrapper, settings=settings,
            strategy=settings.query_transform_strategy,
        )
        return [s.query for s in result.sub_queries], result.technique

    async def retrieve(
        self,
        *,
        reformulated: ReformulatedQuery,
        settings: Settings,
        search_mode: str,
        reranking: bool,
        routing: bool = False,
        query_transform: bool = False,
        temporal_decay: bool = False,
        explicit_targets: list[SearchTarget] | None = None,
        filters: MetadataFilters | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        # Recall amplio si vamos a rerankear; si no, directamente el top_k final.
        recall_k = settings.retrieval_candidate_pool_size if reranking else settings.rag_top_k

        # 1. Transformación de consulta (técnica → tipo de fusión entre sub-consultas).
        sub_queries, technique = self._resolve_subqueries(
            reformulated=reformulated, settings=settings, query_transform=query_transform
        )
        # 2. Routing → colecciones destino.
        targets, level = self._resolve_targets(
            reformulated=reformulated, settings=settings,
            routing=routing, explicit_targets=explicit_targets,
        )
        models = [COLLECTION_MODELS[t] for t in targets]

        # Filtro duro base. El default `filters_from_reformulation` es budget-céntrico
        # (sector + chunk_types de presupuesto) → SOLO se aplica cuando budgets es el
        # único target. En multi-colección NO hay default (cada colección tiene su propio
        # esquema de metadata; aplicar el filtro de budgets vaciaría las otras tablas).
        if filters is not None:
            base_filters = filters
        elif len(targets) == 1 and targets[0] is SearchTarget.BUDGETS:
            base_filters = filters_from_reformulation(reformulated)
        else:
            base_filters = None

        # id sintético por (colección, chunk_id) para fusionar sin colisiones entre tablas.
        synthetic: dict[tuple[int, int], int] = {}
        by_id: dict[int, RetrievedChunk] = {}

        def sid(model_idx: int, chunk: RetrievedChunk) -> int:
            key = (model_idx, chunk.chunk_id)
            if key not in synthetic:
                synthetic[key] = len(synthetic)
                by_id[synthetic[key]] = chunk
            return synthetic[key]

        # 3 + 4. Recall por (sub-consulta × colección) en paralelo; fusión RRF de ramas
        #         (vector + léxica × colecciones) DENTRO de cada sub-consulta.
        per_subquery_rankings: list[list[int]] = []
        for sub in sub_queries:
            branch_specs: list[tuple[int, str]] = []  # (model_idx, branch)
            coros = []
            for model_idx, model in enumerate(models):
                # El filtro budget-céntrico solo se aplica a la colección de budgets;
                # las demás buscan sin filtro duro (su metadata no tiene esos ejes).
                model_filters = base_filters if model is BudgetChunkRow else None
                coros.append(
                    self._vector_search(
                        model=model, search_text=sub, k=recall_k,
                        filters=model_filters, ef_search=settings.hnsw_ef_search,
                    )
                )
                branch_specs.append((model_idx, "vector"))
                if search_mode == "hybrid":
                    coros.append(
                        self._fulltext.search(
                            model=model, query_text=sub, k=recall_k, filters=model_filters
                        )
                    )
                    branch_specs.append((model_idx, "lexical"))
            results = await asyncio.gather(*coros)
            branch_rankings: list[list[int]] = []
            for (model_idx, branch), res in zip(branch_specs, results):
                if not res:
                    # Verifica cardinalidad tras filtrar: rama vacía = silencio del HNSW.
                    logger.info(
                        "rag.branch_dropped_to_zero",
                        collection=targets[model_idx].value, branch=branch, sub=sub,
                    )
                branch_rankings.append([sid(model_idx, c) for c in res])
            per_subquery_rankings.append(
                reciprocal_rank_fusion(branch_rankings, k=settings.rrf_smoothing_k)
            )

        # 5. Fusión entre sub-consultas/colecciones.
        if len(per_subquery_rankings) == 1:
            fused_ids = per_subquery_rankings[0]
        elif technique == "decomposition":
            fused_ids = interleave_rankings(per_subquery_rankings, top_k=recall_k)
        else:  # expansion (consenso) o varias listas directas
            fused_ids = reciprocal_rank_fusion(
                per_subquery_rankings, k=settings.rrf_smoothing_k
            )
        candidates = [by_id[i] for i in fused_ids[:recall_k] if i in by_id]

        # 6. Rerank (cross-encoder local → fuera del event loop).
        if reranking:
            candidates = await asyncio.to_thread(
                get_reranker().rerank,
                reformulated.search_text, candidates, settings.rag_top_k,
            )
        else:
            candidates = candidates[: settings.rag_top_k]

        # 7. Ponderación blanda (último ajuste, solo desempate de finalistas).
        if temporal_decay or settings.contextual_weighting_enabled:
            candidates = apply_soft_weighting(
                candidates, reformulated=reformulated, settings=settings
            )
            candidates = candidates[: settings.rag_top_k]

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info(
            "rag.pipeline_retrieved",
            search_mode=search_mode, reranking=reranking,
            routing=routing, routing_level=level, targets=[t.value for t in targets],
            query_transform=query_transform, technique=technique,
            temporal_decay=temporal_decay,
            recall_k=recall_k, final=len(candidates), search_time_ms=elapsed_ms,
        )
        return RetrievalResult(
            reformulated=reformulated, filters=base_filters or MetadataFilters(),
            top_k=settings.rag_top_k, distance_threshold=settings.rag_distance_threshold,
            chunks=candidates, search_time_ms=elapsed_ms,
            targets=[t.value for t in targets], routing_level=level, technique=technique,
        )
