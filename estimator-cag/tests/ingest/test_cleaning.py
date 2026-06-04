"""Limpieza determinista + política de validación Pandera.

Aquí se fijan empíricamente los nombres de los checks de la versión instalada de
Pandera; si difieren, ajustar app.ingest.cleaning.policy._DISCARD_CHECKS.
"""

from __future__ import annotations

import pandas as pd

from app.ingest.cleaning.budgets import clean_budget_records
from app.ingest.cleaning.policy import validate_with_policy
from app.ingest.cleaning.schemas import BudgetRecord

_RAW = [
    # válido (moneda en minúsculas -> EUR)
    {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme",
        "total_amount": "80000",
        "currency": "eur",
        "signed_at": "2024-03-15",
        "status": "signed",
    },
    # nulo disfrazado -> cuarentena (not_nullable)
    {
        "budget_id": "BUDGET-2024-0003",
        "client_name": "to be defined",
        "total_amount": 30000,
        "currency": "EUR",
        "signed_at": "2024-05-01",
        "status": "draft",
    },
    # importe negativo -> descarte (ge)
    {
        "budget_id": "BUDGET-2024-0004",
        "client_name": "Initech",
        "total_amount": -50000,
        "currency": "EUR",
        "signed_at": "2024-02-10",
        "status": "signed",
    },
    # budget_id roto -> descarte (str_matches)
    {
        "budget_id": "BAD-ID-0006",
        "client_name": "Soylent",
        "total_amount": 21000,
        "currency": "EUR",
        "signed_at": "2024-06-01",
        "status": "signed",
    },
    # dos versiones del mismo presupuesto -> dedup keep-last (la más reciente)
    {
        "budget_id": "BUDGET-2024-0005",
        "client_name": "Umbrella",
        "total_amount": 30000,
        "currency": "EUR",
        "signed_at": "2024-04-10",
        "status": "signed",
    },
    {
        "budget_id": "BUDGET-2024-0005",
        "client_name": "Umbrella",
        "total_amount": 32000,
        "currency": "EUR",
        "signed_at": "2024-04-12",
        "status": "signed",
    },
]


def test_clean_normalizes_currency_and_nulls_and_dedups():
    cleaned = clean_budget_records(pd.DataFrame(_RAW))
    # dedup: 6 filas crudas -> 5 (budget_005 colapsa a una)
    assert len(cleaned) == 5
    assert set(cleaned["currency"]) == {"EUR"}
    # nulo disfrazado convertido a NA real
    row_003 = cleaned[cleaned["budget_id"] == "BUDGET-2024-0003"].iloc[0]
    assert pd.isna(row_003["client_name"])
    # keep-last: se conserva la versión de 32000
    row_005 = cleaned[cleaned["budget_id"] == "BUDGET-2024-0005"].iloc[0]
    assert row_005["total_amount"] == 32000


def test_policy_routes_by_severity():
    cleaned = clean_budget_records(pd.DataFrame(_RAW))
    result = validate_with_policy(cleaned, BudgetRecord)

    valid_ids = set(result.valid["budget_id"])
    quarantined_ids = set(result.quarantined["budget_id"])
    discarded_ids = set(result.discarded["budget_id"])

    assert "BUDGET-2024-0004" in discarded_ids  # negativo
    assert "BAD-ID-0006" in discarded_ids  # id roto
    assert "BUDGET-2024-0003" in quarantined_ids  # nulo
    assert "BUDGET-2024-0001" in valid_ids
    assert "BUDGET-2024-0005" in valid_ids  # superviviente del dedup
    assert result.report["total"] == 5
