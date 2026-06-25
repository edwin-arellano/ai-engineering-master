"""Reciprocal Rank Fusion — función pura, determinista (sin DB ni modelo)."""

from __future__ import annotations

from app.generation.rag.retrieval.fusion import RRF_SMOOTHING_K, reciprocal_rank_fusion


def test_consensus_beats_single_champion():
    """Replica el ejemplo del artículo: un id 2º+5º supera al campeón de un solo ranking.
    id=2 está 2º en r1 (1/62) y 5º en r2 (1/65) → 0.0315; id=1 solo 1º en r1 (1/61) → 0.0164."""
    r1 = [1, 2, 3, 4, 5]
    r2 = [6, 7, 8, 9, 2]
    fused = reciprocal_rank_fusion([r1, r2])
    assert fused[0] == 2
    assert fused.index(2) < fused.index(1)


def test_default_k_is_60():
    assert RRF_SMOOTHING_K == 60


def test_fuses_n_rankings_not_only_two():
    # id=7 aparece en los tres rankings → debe ganar al que solo aparece en uno.
    fused = reciprocal_rank_fusion([[7, 1], [7, 2], [7, 3]])
    assert fused[0] == 7


def test_single_ranking_is_preserved():
    assert reciprocal_rank_fusion([[10, 20, 30]]) == [10, 20, 30]


def test_empty_input_yields_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_smaller_k_sharpens_top_positions():
    """k pequeño → dominan las primeras posiciones. Con k=1, el 1º de un ranking
    pesa 1/2; el consenso en posiciones bajas no lo alcanza tan fácil."""
    r1 = [1, 2, 3]
    r2 = [4, 5, 1]
    fused = reciprocal_rank_fusion([r1, r2], k=1)
    assert fused[0] == 1  # 1º en r1 (1/2) + 3º en r2 (1/4) = 0.75, claro ganador
