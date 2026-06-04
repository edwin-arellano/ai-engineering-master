"""Genera el seed corpus sintético de ingesta (NO se comitea, reproducible).

Defectos calibrados del directo S06:
- budget_004: total negativo -> discard (check le/ge)
- budget_003: client_name "to be defined" -> quarantine (nullable)
- budget_005 v1/v2: mismo budget_id, totales discordantes, signed_at distinto -> dedup keep-last
- heterogeneidad de moneda (eur / EUR / €)
- budget_id con formato roto -> discard (str_matches)
- transcript legacy sin tags de speaker -> review (catálogo)
- rate_card_2024.xlsx envejecido >365 días -> exclude (catálogo)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_budgets() -> None:
    b = SEED / "budgets"
    _write_json(
        b / "budget_001.json",
        {
            "budget_id": "BUDGET-2024-0001",
            "client_name": "Acme Corp",
            "total_amount": "80000",
            "currency": "eur",
            "signed_at": "2024-03-15",
            "status": "signed",
        },
    )
    _write_json(
        b / "budget_002.json",
        {
            "budget_id": "BUDGET-2024-0002",
            "client_name": "Globex",
            "total_amount": 45000,
            "currency": "€",
            "signed_at": "15/04/2024",
            "status": "signed",
        },
    )
    _write_json(
        b / "budget_003.json",
        {
            "budget_id": "BUDGET-2024-0003",
            "client_name": "to be defined",
            "total_amount": 30000,
            "currency": "EUR",
            "signed_at": "2024-05-01",
            "status": "draft",
        },
    )
    _write_json(
        b / "budget_004.json",
        {
            "budget_id": "BUDGET-2024-0004",
            "client_name": "Initech",
            "total_amount": -50000,
            "currency": "EUR",
            "signed_at": "2024-02-10",
            "status": "signed",
        },
    )
    _write_json(
        b / "budget_005_v1.json",
        {
            "budget_id": "BUDGET-2024-0005",
            "client_name": "Umbrella",
            "total_amount": 30000,
            "currency": "EUR",
            "signed_at": "2024-04-10",
            "status": "signed",
        },
    )
    _write_json(
        b / "budget_005_v2.json",
        {
            "budget_id": "BUDGET-2024-0005",
            "client_name": "Umbrella",
            "total_amount": 32000,
            "currency": "EUR",
            "signed_at": "2024-04-12",
            "status": "signed",
        },
    )
    _write_json(
        b / "budget_006.json",
        {
            "budget_id": "BAD-ID-0006",
            "client_name": "Soylent",
            "total_amount": 21000,
            "currency": "EUR",
            "signed_at": "2024-06-01",
            "status": "signed",
        },
    )


def build_transcripts() -> None:
    t = SEED / "transcripts"
    t.mkdir(parents=True, exist_ok=True)
    (t / "meeting_2024_modern.txt").write_text(
        "[00:00:01] Antonio: Repasamos el alcance del proyecto.\n"
        "[00:00:09] Cliente: Queremos integrar pagos y reporting.\n",
        encoding="utf-8",
    )
    (t / "meeting_legacy.txt").write_text(  # sin tags de speaker -> review
        "Repasamos el alcance del proyecto.\nQueremos integrar pagos y reporting.\n",
        encoding="utf-8",
    )


def build_rate_card() -> None:
    r = SEED / "rates"
    r.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["role", "seniority", "hourly_rate_eur"])
    ws.append(["backend", "senior", 75])
    ws.append(["frontend", "mid", 55])
    path = r / "rate_card_2024.xlsx"
    wb.save(path)
    # envejecer >365 días para forzar exclude por obsolescencia
    old = (datetime.now(timezone.utc) - timedelta(days=480)).timestamp()
    os.utime(path, (old, old))


def main() -> None:
    build_budgets()
    build_transcripts()
    build_rate_card()
    print(f"Seed generado en {SEED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
