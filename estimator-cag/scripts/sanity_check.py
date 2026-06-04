"""Ejecuta las tres parejas de validación del ejercicio y escribe
app/embedding_pipeline/SANITY_CHECK.md con los valores y un comentario editable.
"""

from __future__ import annotations

from pathlib import Path

from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from scripts.compare import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "embedding_pipeline" / "SANITY_CHECK.md"

PAIRS = [
    ("A — cercanos (esperado > 0.6)",
     "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app",
     "Authorization service using JSON Web Tokens for a banking application"),
    ("B — no relacionados (esperado < 0.4)",
     "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app",
     "Database migration from MySQL to PostgreSQL with zero downtime"),
    ("C — genéricos/ambiguos (sin expectativa fija)",
     "Backend services",
     "API development"),
]


def main() -> None:
    embedder = LiteLLMEmbedder()
    lines = [
        "# Sanity check de embeddings",
        "",
        "| Pareja | Texto A | Texto B | Coseno |",
        "|---|---|---|---|",
    ]
    results = []
    for label, a, b in PAIRS:
        sim = cosine_similarity(embedder.embed_one(a), embedder.embed_one(b))
        results.append((label, sim))
        lines.append(f"| {label} | {a} | {b} | {sim:.4f} |")
    lines += [
        "",
        "## Comentario",
        "",
        "<!-- 3-5 líneas: ¿encajan los resultados con la intuición? ¿algún valor sorprendente? -->",
        "- Pareja A:",
        "- Pareja B:",
        "- Pareja C:",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for label, sim in results:
        print(f"{label}: {sim:.4f}")
    print(f"\nEscrito {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
