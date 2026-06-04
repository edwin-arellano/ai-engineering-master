"""Compara el embedding del corpus en 1536 vs 768 dimensiones: latencia y acuerdo
de coseno (replica el demo de Antonio). text-embedding-3-small soporta dimensions=.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import litellm

from app.generation.rag.chunking.registry import build_chunker
from app.generation.rag.schemas import Budget

ROOT = Path(__file__).resolve().parents[1]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed(texts, dims):
    t = time.perf_counter()
    r = litellm.embedding(
        model="text-embedding-3-small", input=texts, dimensions=dims
    )
    vectors = [
        item["embedding"] if isinstance(item, dict) else item.embedding
        for item in r.data
    ]
    return vectors, (time.perf_counter() - t) * 1000


def main() -> None:
    budgets = [
        Budget(**b)
        for b in json.loads((ROOT / "data" / "budgets_sample.json").read_text())
    ]
    texts = [c.text for c in build_chunker("structural").chunk(budgets)][:20]
    v1536, l1536 = _embed(texts, 1536)
    v768, l768 = _embed(texts, 768)
    diffs = [
        abs(_cos(v1536[i], v1536[j]) - _cos(v768[i], v768[j]))
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    print(f"1536 dims: {l1536:.0f} ms | 768 dims: {l768:.0f} ms")
    print(
        f"diferencia media de coseno entre pares (1536 vs 768): "
        f"{sum(diffs) / len(diffs):.5f}"
    )


if __name__ == "__main__":
    main()
