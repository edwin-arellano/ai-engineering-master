"""Medición artesanal de recuperación contra un golden set anotado a mano.
Ejecuta las 4 configuraciones (A/B/C/D) contra el endpoint de debug de retrieval
(/api/v1/retrieve-debug, sin generación LLM) y reporta precision@5 y latencia
mediana. NO es infraestructura: herramienta de decisión puntual.

Latencia medida = `search_time_ms` del pipeline (retrieval puro: excluye reformulación
y HTTP), que es justo lo que cambia entre las 4 configs. Medición en caliente
(descarta la 1ª run de cada query), mediana de las runs restantes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

import httpx

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
BACKEND = "http://localhost:8000"
ENDPOINT = "/api/v1/retrieve-debug"
RUNS_PER_QUERY = 4  # 1 de calentamiento (se descarta) + 3 medidas

CONFIGS = {
    "A_vector_norerank": {"search_mode": "vector", "reranking": False},
    "B_hybrid_norerank": {"search_mode": "hybrid", "reranking": False},
    "C_vector_rerank":   {"search_mode": "vector", "reranking": True},
    "D_hybrid_rerank":   {"search_mode": "hybrid", "reranking": True},
}


def precision_at_k(retrieved_budget_ids: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_budget_ids[:k]
    if not top:
        return 0.0
    return sum(1 for b in top if b in relevant) / len(top)


def _run_table(client: httpx.Client, golden: dict, top_k: int, *, apply_filters: bool):
    rows = []
    for config_name, params in CONFIGS.items():
        precisions: list[float] = []
        latencies: list[float] = []  # search_time_ms (retrieval puro)
        wall_latencies: list[float] = []  # wall-clock HTTP (referencia)
        for entry in golden["queries"]:
            relevant = set(entry["relevant_budget_ids"])
            retrieved_ids: list[str] = []
            for run in range(RUNS_PER_QUERY):
                start = time.perf_counter()
                resp = client.post(
                    ENDPOINT,
                    json={
                        "transcript": entry["query"],
                        "apply_metadata_filters": apply_filters,
                        **params,
                    },
                )
                wall = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                body = resp.json()
                retrieved_ids = body["retrieved_budget_ids"]
                if run > 0:  # descarta la run de calentamiento
                    latencies.append(float(body["search_time_ms"]))
                    wall_latencies.append(wall)
            p = precision_at_k(retrieved_ids, relevant, top_k)
            precisions.append(p)
            print(f"  [{config_name}] {entry['id']}: precision@{top_k}={p:.2f} "
                  f"retrieved={retrieved_ids[:top_k]}")
        rows.append((
            config_name,
            round(sum(precisions) / len(precisions), 3),
            round(median(latencies)),
            round(median(wall_latencies)),
        ))
        print(f"=> {config_name}: precision@{top_k}={rows[-1][1]} "
              f"retrieval_mediana={rows[-1][2]}ms wall_mediana={rows[-1][3]}ms\n")

    print(f"\n| Config | Búsqueda | Reranking | precision@{top_k} | "
          f"Latencia retrieval (ms) | Wall-clock (ms) |")
    print("|---|---|---|---|---|---|")
    for name, p, lat, wall in rows:
        sm = "Híbrida" if "hybrid" in name else "Vectorial"
        rr = "Sí" if name.endswith("_rerank") else "No"
        print(f"| {name[0]} | {sm} | {rr} | {p} | {lat} | {wall} |")


def main() -> None:
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    top_k = golden.get("top_k", 5)
    with httpx.Client(base_url=BACKEND, timeout=300.0) as client:
        print("### Tabla 1 — con filtro de metadata (sector) de S09 [producción]\n")
        _run_table(client, golden, top_k, apply_filters=True)
        print("\n### Tabla 2 — sin filtro de sector (ranking sobre todo el corpus)\n")
        _run_table(client, golden, top_k, apply_filters=False)


if __name__ == "__main__":
    main()
