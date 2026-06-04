"""Limpieza de registros de presupuesto. Cada paso es estrecho y sin efectos
secundarios para testear en aislamiento. NO decide: deja NaN/NaT que resolverá
la capa de validación. El orden importa (nulos antes que dedup).
"""

from __future__ import annotations

import hashlib

import pandas as pd

NULL_PLACEHOLDERS = {
    "",
    "n/a",
    "na",
    "-",
    "--",
    "unknown",
    "tbd",
    "pendiente",
    "to be defined",
}
_CURRENCY_MAP = {
    "eur": "EUR",
    "euros": "EUR",
    "€": "EUR",
    "usd": "USD",
    "$": "USD",
    "gbp": "GBP",
}


def clean_budget_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. nulos disfrazados -> NA real
    out["client_name"] = (
        out["client_name"]
        .astype(str)
        .str.strip()
        .where(lambda s: ~s.str.lower().isin(NULL_PLACEHOLDERS), other=pd.NA)
    )

    # 2. normalización de moneda
    out["currency"] = (
        out["currency"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_CURRENCY_MAP)
        .fillna(out["currency"])
    )

    # 3. fechas con coerción permisiva
    out["signed_at"] = pd.to_datetime(out["signed_at"], errors="coerce", utc=True)

    # 4. importes a numérico
    out["total_amount"] = pd.to_numeric(out["total_amount"], errors="coerce")

    # 5. dedup por hash de contenido: conserva la versión más reciente por budget_id
    out["content_hash"] = out.apply(
        lambda r: hashlib.sha256(
            f"{r['budget_id']}|{r['total_amount']}|{r['currency']}".encode()
        ).hexdigest(),
        axis=1,
    )
    out = out.sort_values("signed_at").drop_duplicates(
        subset=["budget_id"], keep="last"
    )
    return out.drop(columns=["content_hash"])
