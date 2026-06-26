"""Transformación de consulta previa a la búsqueda. Tres caminos NO intercambiables:
- direct: consulta corta y nítida → no se llama al LLM (1 sub-consulta = la original).
- expansion: una intención, varias formulaciones (fusión por CONSENSO/RRF aguas abajo).
- decomposition: varias intenciones → sub-consultas independientes (fusión por
  COBERTURA/interleave aguas abajo).

Heurística humilde para elegir 'direct' sin pagar latencia; LLM con la correa corta
para el resto (máx sub-consultas, términos exactos del dominio preservados).
"""

from __future__ import annotations

import structlog

from app.foundations.config import Settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_query_transform_prompt
from app.generation.rag.schemas import QueryTransformResult, SubQuery

logger = structlog.get_logger(__name__)

# Umbral heurístico: consultas cortas y de un solo tema pasan directas (sin LLM).
_DIRECT_MAX_WORDS = 12


def transform_query(
    query: str,
    *,
    wrapper: LLMWrapper,
    settings: Settings,
    strategy: str = "auto",
) -> QueryTransformResult:
    """Decide la técnica y devuelve las sub-consultas. `strategy`: auto|expand|decompose|off.

    - off, o auto con consulta corta → 'direct' (sin LLM).
    - resto → clasificador LLM (alias barato) que elige expansion|decomposition."""
    if strategy == "off" or (
        strategy == "auto" and len(query.split()) <= _DIRECT_MAX_WORDS
    ):
        logger.info("query_transform.direct", words=len(query.split()))
        return QueryTransformResult(
            technique="direct", sub_queries=[SubQuery(topic="original", query=query)]
        )
    result = wrapper.complete_structured(
        system_prompt=render_query_transform_prompt(settings.query_transform_prompt_version),
        user_message=query,
        response_model=QueryTransformResult,
        alias=REFORMULATOR_ALIAS,
        temperature=0.0,
        max_tokens=400,
    )
    # Correa corta: respeta el tope de sub-consultas aunque el modelo se exceda.
    if len(result.sub_queries) > settings.query_transform_max_subqueries:
        result.sub_queries = result.sub_queries[: settings.query_transform_max_subqueries]
    logger.info(
        "query_transform.llm",
        technique=result.technique,
        sub_queries=len(result.sub_queries),
        topics=[s.topic for s in result.sub_queries],
    )
    return result
