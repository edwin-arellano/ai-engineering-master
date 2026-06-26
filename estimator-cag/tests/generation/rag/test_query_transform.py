"""Transformación de consulta + fusión por cobertura. Unit (sin LLM): el camino directo
no llama al modelo; interleave_rankings garantiza cobertura y dedup."""

from __future__ import annotations

from app.foundations.config import get_settings
from app.generation.rag.retrieval.fusion import interleave_rankings
from app.generation.rag.retrieval.query_transform import transform_query


class _ExplodingWrapper:
    """Si el camino directo llamara al LLM, esto reventaría el test."""

    def complete_structured(self, **_kwargs):
        raise AssertionError("el camino directo NO debe llamar al LLM")


def test_direct_path_does_not_call_llm():
    result = transform_query(
        "API banca OAuth", wrapper=_ExplodingWrapper(), settings=get_settings(),
        strategy="auto",
    )
    assert result.technique == "direct"
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0].query == "API banca OAuth"


def test_off_strategy_is_direct_without_llm():
    result = transform_query(
        "una consulta larga con muchas palabras que normalmente iría al modelo de transformación",
        wrapper=_ExplodingWrapper(), settings=get_settings(), strategy="off",
    )
    assert result.technique == "direct"


def test_interleave_round_robin_with_dedup():
    # A=[1,2,3], B=[2,4,5]: round-robin con dedup → 1,2(de A),4,3 (2 de B se salta)
    fused = interleave_rankings([[1, 2, 3], [2, 4, 5]], top_k=4)
    assert fused == [1, 2, 4, 3]
    assert len(fused) == len(set(fused))  # sin duplicados


def test_interleave_covers_every_ranking():
    # Cobertura: el primer elemento de CADA ranking entra antes que el segundo de otro.
    fused = interleave_rankings([[10, 11], [20, 21], [30, 31]], top_k=3)
    assert fused == [10, 20, 30]  # un representante por tema antes de profundizar


def test_interleave_empty():
    assert interleave_rankings([], top_k=5) == []
