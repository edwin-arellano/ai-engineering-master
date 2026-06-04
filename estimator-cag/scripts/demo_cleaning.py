"""CLI demo de limpieza+validación (estilo directo S06).

Carga los presupuestos JSON del seed, los limpia de forma determinista y los
valida con Pandera, mostrando cómo se reparten en válidos / cuarentena / descarte.

    uv run python -m scripts.demo_cleaning
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.foundations.config import get_settings
from app.ingest.cleaning.budgets import clean_budget_records
from app.ingest.cleaning.policy import validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord

ROOT = Path(__file__).resolve().parents[1]


def _load_budget_records(budgets_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(budgets_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(records)


def _show(title: str, df: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n## {title} ({len(df)})")
    if df.empty:
        print("  (vacío)")
        return
    present = [c for c in columns if c in df.columns]
    for _, row in df.iterrows():
        print("  - " + ", ".join(f"{c}={row[c]}" for c in present))


def main() -> None:
    settings = get_settings()
    budgets_dir = ROOT / settings.ingest_seed_dir / "budgets"
    if not budgets_dir.exists():
        raise SystemExit(
            f"No existe {budgets_dir}. Genera el seed: uv run python -m scripts.build_seed"
        )

    raw = _load_budget_records(budgets_dir)
    print(f"Registros crudos: {len(raw)}")

    cleaned = clean_budget_records(raw)
    print(f"Tras limpieza+dedup: {len(cleaned)}")

    result = validate_with_policy(cleaned, BudgetRecord)
    cols = ["budget_id", "client_name", "total_amount", "currency", "status"]
    _show("Válidos", result.valid, cols)
    _show("En cuarentena", result.quarantined, cols)
    _show("Descartados", result.discarded, cols)
    print(f"\nReporte: {result.report}")


if __name__ == "__main__":
    main()
