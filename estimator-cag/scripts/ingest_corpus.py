"""Ingesta el corpus de ejemplo (data/budgets_sample.json) contra el endpoint HTTP.
Cliente puro: no toca la DB ni importa app.*. Cada budget = un documento; source_path
derivado del budget_id. Tolera 409 (ya ingestado) para ser idempotente.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

SAMPLE = Path("data/budgets_sample.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--sample", default=str(SAMPLE))
    args = parser.parse_args()

    budgets = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    print(f"Ingestando {len(budgets)} presupuestos contra {args.backend} ...\n")

    created = skipped = 0
    with httpx.Client(base_url=args.backend, timeout=120.0) as client:
        for budget in budgets:
            source_path = f"budgets_sample/{budget['budget_id']}"
            resp = client.post(
                "/embeddings/ingest",
                json={
                    "source_path": source_path,
                    "document_type": "historical_budget",
                    "content": budget,
                },
            )
            if resp.status_code == 200:
                body = resp.json()
                created += 1
                print(
                    f"  [OK ] {source_path}: doc={body['document_id']} "
                    f"chunks={body['chunks_created']} {body['ingestion_time_ms']}ms"
                )
            elif resp.status_code == 409:
                skipped += 1
                print(f"  [409] {source_path}: ya ingestado (doc={resp.json()['document_id']})")
            else:
                print(f"  [ERR] {source_path}: {resp.status_code} {resp.text[:200]}")

    print(f"\nResumen: {created} creados, {skipped} ya existentes.")


if __name__ == "__main__":
    main()
