"""Ejercita /search con cinco queries representativas y vuelca el resultado a
output_examples.txt (y a stdout). Reemplaza, para el caso de búsqueda end-to-end,
al compare.py de S07 (que medía coseno entre pares de textos sueltos).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

QUERIES = [
    # 1. Componente directo conocido (sanity check, match casi perfecto esperado)
    "REST API development with JWT authentication for financial sector",
    # 2. Reformulación semántica (mismo concepto, otro vocabulario)
    "secure backend service with token-based access control for banking applications",
    # 3. Dominio distinto (debería dar distancias altas / irrelevante)
    "mobile application for restaurant reservations",
    # 4. Consulta ambigua (corta y genérica)
    "integration with external system",
    # 5. Consulta muy específica (vocabulario técnico preciso)
    "migration from monolith to microservices architecture using Kubernetes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="output_examples.txt")
    args = parser.parse_args()

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    with httpx.Client(base_url=args.backend, timeout=60.0) as client:
        for i, query in enumerate(QUERIES, start=1):
            resp = client.post("/search", json={"query": query, "k": args.k})
            resp.raise_for_status()
            body = resp.json()
            emit("=" * 88)
            emit(f"Q{i}: {query}")
            emit(f"    (k={body['k']}, search_time_ms={body['search_time_ms']})")
            emit("-" * 88)
            for r in body["results"]:
                snippet = r["content"][:120].replace("\n", " ")
                emit(
                    f"  dist={r['distance']:.4f}  chunk_id={r['chunk_id']:<5} "
                    f"type={r['chunk_type']:<18} {snippet}"
                )
            emit()

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nEscrito {args.out}")


if __name__ == "__main__":
    main()
