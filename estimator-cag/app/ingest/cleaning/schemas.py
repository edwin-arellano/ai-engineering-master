from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing.pandas import Series


class BudgetRecord(DataFrameModel):
    """Contrato canónico de presupuestos antes de normalizar a Document.

    strict=True: rechaza columnas no declaradas (defensa ante cambios silenciosos
    del parser). coerce=False: la limpieza ya hizo las coerciones.
    """

    budget_id: Series[str] = Field(str_matches=r"^BUDGET-\d{4}-\d{4}$")
    client_name: Series[str] = Field(
        nullable=False, str_length={"min_value": 2, "max_value": 200}
    )
    total_amount: Series[float] = Field(ge=0, le=10_000_000, nullable=False)
    currency: Series[str] = Field(isin=["EUR", "USD", "GBP"])
    signed_at: Series[pd.Timestamp] = Field(
        nullable=False, le=datetime.now(timezone.utc)
    )
    status: Series[str] = Field(isin=["draft", "signed", "rejected"])

    class Config:
        strict = True
        coerce = False
        ordered = False

    @pa.dataframe_check
    def positive_amount_for_signed(cls, df: pd.DataFrame) -> Series[bool]:
        """Regla cross-column: status='signed' implica total_amount > 0."""
        return ~((df["status"] == "signed") & (df["total_amount"] == 0))
