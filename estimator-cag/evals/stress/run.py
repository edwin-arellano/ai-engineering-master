"""Runner del stress test. Golpea el endpoint HTTP real y vuelca un CSV.

uso:
  uv run python -m evals.stress.run --http http://localhost:8000 \\
      --scenarios growing,pivot,contradiction \\
      --attachment-sizes 0,5,20,50,100 \\
      --repeats 3 \\
      --output evals/stress/results.csv

Usa `--http` (no in-process) a propósito: medir la latencia P95 realista exige
incluir el overhead del endpoint. El snapshot de cada turno se lee vía
`GET /sessions/{id}`, que ya embebe el `turn_observed` y la memoria persistente
(`last_summary`/`anchored_facts`/`project_metadata`), de modo que la
`MemoryDriftMetric` se evalúa sobre el snapshot tal cual, sin inyectar nada.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import httpx

from evals.stress.fixtures.build_pdfs import FIXTURES_DIR, build_all
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import SCENARIOS

CSV_FIELDS = [
    "scenario", "attachment_kb", "repeat", "turn_index", "session_id",
    "enriched_transcript_chars", "attachments_total_chars", "messages_in_window",
    "anchors_count", "summary_chars", "tokens_in", "tokens_out", "cost_usd",
    "latency_ms", "cache_hit_kind", "last_resolved_tier",
    "latency_budget_passed", "cost_budget_passed", "memory_drift_passed",
]


def _turns_for(scenario, n_cap: int):
    return [t for t in scenario.turns if t.turn_index <= n_cap]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", required=True)
    parser.add_argument("--scenarios", default="growing,pivot,contradiction")
    parser.add_argument("--attachment-sizes", default="0,5,20,50,100")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-cap", type=int, default=20)
    parser.add_argument("--latency-budget-ms", type=int, default=4000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.05)
    parser.add_argument("--output", default="evals/stress/results.csv")
    args = parser.parse_args()

    build_all()  # regenera PDFs determinísticos
    sizes = [int(x) for x in args.attachment_sizes.split(",")]
    scenario_names = args.scenarios.split(",")

    rows: list[dict] = []
    with httpx.Client(base_url=args.http, timeout=180.0) as client:
        for scen_name in scenario_names:
            scenario = SCENARIOS[scen_name]
            for kb in sizes:
                for repeat in range(args.repeats):
                    session_id = client.post(
                        "/api/v1/sessions", json={"estimation_mode": "actor"}
                    ).json()["session_id"]

                    files = None
                    if kb > 0:
                        pdf = FIXTURES_DIR / f"attach_{kb}kb.pdf"
                        files = {
                            "attachments": (pdf.name, pdf.read_bytes(), "application/pdf")
                        }

                    for turn in _turns_for(scenario, args.n_cap):
                        resp = client.post(
                            f"/api/v1/sessions/{session_id}/estimate",
                            data={
                                "transcript": turn.transcript,
                                "project_type": "web_saas",
                                "detail_level": "medium",
                                "output_format": "phases_table",
                            },
                            files=files,
                        )
                        resp.raise_for_status()

                        # El snapshot del endpoint es autosuficiente: trae el
                        # turn_observed (latencia/coste) y la memoria persistente.
                        snapshot = client.get(
                            f"/api/v1/sessions/{session_id}"
                        ).json()
                        observed = snapshot["last_turn_observed"] or {}

                        lat = LatencyBudgetMetric(args.latency_budget_ms).evaluate(
                            observed
                        )
                        cost = CostBudgetMetric(args.cost_budget_usd).evaluate(observed)
                        drift = MemoryDriftMetric(turn.fact_to_remember).evaluate(
                            snapshot
                        )

                        row = {
                            **observed,
                            "scenario": scen_name,
                            "attachment_kb": kb,
                            "repeat": repeat,
                            "latency_budget_passed": lat.passed,
                            "cost_budget_passed": cost.passed,
                            "memory_drift_passed": drift.passed,
                        }
                        rows.append({k: row.get(k) for k in CSV_FIELDS})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Resumen por consola.
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        print(f"Filas: {len(rows)} | P50 latency: {p50:.0f}ms | P95: {p95:.0f}ms")
    print(f"CSV → {out}")


if __name__ == "__main__":
    main()
