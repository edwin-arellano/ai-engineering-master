"""Compara la similitud coseno entre dos textos usando LiteLLMEmbedder.
Coseno calculado a mano con la biblioteca estándar (sin numpy/scikit-learn).

Uso:
  uv run python -m scripts.compare --text-a "..." --text-b "..."
  uv run python scripts/compare.py --text-a "..." --text-b "..."
  docker compose exec servicio_ia python scripts/compare.py --text-a "..." --text-b "..."
"""

from __future__ import annotations

import argparse
import math

from app.embedding_pipeline.embedder import LiteLLMEmbedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Similitud coseno entre dos textos")
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    args = parser.parse_args()

    embedder = LiteLLMEmbedder()
    vec_a = embedder.embed_one(args.text_a)
    vec_b = embedder.embed_one(args.text_b)
    sim = cosine_similarity(vec_a, vec_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
