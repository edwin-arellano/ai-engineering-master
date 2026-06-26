"""Medición por etapa del pipeline avanzado (S10): además de vector/hybrid × rerank,
activa/desactiva routing, query_transform y temporal_decay, y reporta:

- Tabla de ABLACIÓN por etapa sobre las queries de budgets: precision@5 + latencia
  mediana (search_time_ms del pipeline, retrieval puro). Cada fila flipa UNA etapa
  sobre la baseline para aislar su aporte.
- Tabla de ACCURACY DE ROUTING sobre routing_queries: ¿los targets resueltos coinciden
  con expected_targets? + nivel del router (deterministic/llm/fallback) y técnica.

Medición en caliente (descarta la 1ª run de cada query), mediana de las restantes.
NO es infraestructura: herramienta de decisión puntual. Requiere el backend arriba y
las 3 colecciones ingestadas.

    uv run python -m scripts.measure_pipeline
"""

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

# Ablación por etapa: baseline + una etapa activada por fila (sobre todo el corpus,
# sin filtro de sector, para que el ranking sea sensible al orden).
ABLATION_CONFIGS = {
    "baseline_vector":     {"search_mode": "vector", "reranking": False},
    "+hybrid":             {"search_mode": "hybrid", "reranking": False},
    "+rerank":             {"search_mode": "vector", "reranking": True},
    "+routing":            {"search_mode": "vector", "reranking": False, "routing": True},
    "+query_transform":    {"search_mode": "vector", "reranking": False, "query_transform": True},
    "+temporal":           {"search_mode": "vector", "reranking": False, "temporal_decay": True},
    "all_stages":          {"search_mode": "hybrid", "reranking": True, "routing": True,
                            "query_transform": True, "temporal_decay": True},
}


def precision_at_k(retrieved_budget_ids: list[str], relevant: set[str], k: int) -> float:
    top = retrieved_budget_ids[:k]
    if not top:
        return 0.0
    return sum(1 for b in top if b in relevant) / len(top)


def _post(client: httpx.Client, query: str, params: dict) -> tuple[dict, float]:
    start = time.perf_counter()
    resp = client.post(
        ENDPOINT,
        json={"transcript": query, "apply_metadata_filters": False, **params},
    )
    wall = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return resp.json(), wall


def _ablation_table(client: httpx.Client, golden: dict, top_k: int) -> None:
    print("### Ablación por etapa (queries de budgets, ranking sobre todo el corpus)\n")
    rows = []
    for config_name, params in ABLATION_CONFIGS.items():
        precisions: list[float] = []
        latencies: list[float] = []
        for entry in golden["queries"]:
            relevant = set(entry["relevant_budget_ids"])
            retrieved: list[str] = []
            for run in range(RUNS_PER_QUERY):
                body, _ = _post(client, entry["query"], params)
                retrieved = body["retrieved_budget_ids"]
                if run > 0:
                    latencies.append(float(body["search_time_ms"]))
            precisions.append(precision_at_k(retrieved, relevant, top_k))
        rows.append(
            (
                config_name,
                round(sum(precisions) / len(precisions), 3),
                round(median(latencies)) if latencies else 0,
            )
        )
        print(f"=> {config_name}: precision@{top_k}={rows[-1][1]} retrieval_mediana={rows[-1][2]}ms")

    print(f"\n| Config | precision@{top_k} | Latencia retrieval (ms) |")
    print("|---|---|---|")
    for name, p, lat in rows:
        print(f"| {name} | {p} | {lat} |")
    print()


def _routing_table(client: httpx.Client, golden: dict) -> None:
    routing_queries = golden.get("routing_queries", [])
    if not routing_queries:
        return
    print("### Accuracy de routing (routing_queries)\n")
    print("| Caso | Esperado | Resuelto | Nivel | Técnica | Acierto |")
    print("|---|---|---|---|---|---|")
    hits = 0
    for entry in routing_queries:
        expected = entry.get("expected_targets", [])
        # routing + query_transform activos: medimos targets y técnica a la vez.
        body, _ = _post(
            client, entry["query"], {"routing": True, "query_transform": True}
        )
        resolved = body.get("targets", [])
        ok = set(resolved) == set(expected)
        hits += int(ok)
        print(
            f"| {entry['id']} | {expected} | {resolved} | "
            f"{body.get('routing_level', '')} | {body.get('technique', '')} | "
            f"{'✅' if ok else '❌'} |"
        )
    print(f"\n=> routing accuracy: {hits}/{len(routing_queries)} "
          f"({round(100 * hits / len(routing_queries))}%)\n")


def main() -> None:
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    top_k = golden.get("top_k", 5)
    with httpx.Client(base_url=BACKEND, timeout=300.0) as client:
        _ablation_table(client, golden, top_k)
        _routing_table(client, golden)


if __name__ == "__main__":
    main()
