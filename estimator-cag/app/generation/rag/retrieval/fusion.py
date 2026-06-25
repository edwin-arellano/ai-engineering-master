"""Reciprocal Rank Fusion: fusiona varios rankings por posición (no por puntuación).
Esquiva el problema de combinar distancias coseno con ts_rank, que viven en escalas
incomparables. Función pura: recibe listas de ids ordenadas, devuelve un ranking único."""

from __future__ import annotations

from collections import defaultdict

# Constante de suavizado del paper original; robusta entre dominios. Cambiarla es
# optimización prematura (k pequeño → dominan las primeras posiciones; k grande → se aplana).
RRF_SMOOTHING_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = RRF_SMOOTHING_K,
) -> list[int]:
    """Fusiona múltiples rankings de chunk_id en un ranking único.

    rrf_score(d) = Σ_i 1 / (k + rank_i(d)), con rank empezando en 1. El consenso gana:
    aparecer razonablemente arriba en varios rankings vale más que arrasar en uno.
    Recibe una LISTA de rankings (no exactamente dos) para fusionar N fuentes sin cambios.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return [
        chunk_id
        for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
