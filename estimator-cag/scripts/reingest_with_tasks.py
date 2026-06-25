"""Re-ingesta el corpus poblando los DOS chunk_types: budget_component (overview)
e historical_task (detalle). Idempotente (409). Cliente HTTP puro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

SAMPLE = Path("data/budgets_sample.json")
STRATEGIES = [("structural", "components"), ("historical_task", "tasks")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--sample", default=str(SAMPLE))
    args = parser.parse_args()

    budgets = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.backend, timeout=180.0) as client:
        for strategy, suffix in STRATEGIES:
            for budget in budgets:
                source_path = f"budgets_sample/{budget['budget_id']}/{suffix}"
                resp = client.post(
                    "/embeddings/ingest",
                    json={
                        "source_path": source_path,
                        "document_type": "historical_budget",
                        "content": budget,
                        "chunk_strategy": strategy,
                    },
                )
                tag = {200: "OK ", 409: "409"}.get(resp.status_code, "ERR")
                print(f"  [{tag}] {source_path} ({strategy})")


if __name__ == "__main__":
    main()
