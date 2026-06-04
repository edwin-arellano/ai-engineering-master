from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
import structlog

logger = structlog.get_logger(__name__)

# checks cuyo fallo indica contaminación clara -> descarte. Los nombres exactos
# dependen de la versión de Pandera; se fijan empíricamente (ver test_cleaning).
_DISCARD_CHECKS = {
    "str_matches",
    "greater_than_or_equal_to",
    "less_than_or_equal_to",
}


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame
    discarded: pd.DataFrame
    report: dict


def validate_with_policy(
    df: pd.DataFrame, schema: type[pa.DataFrameModel]
) -> ValidationResult:
    """Valida con lazy=True y enruta los fallos por severidad.

    lazy=True recoge TODOS los errores (no solo el primero), que es lo que permite
    una política diferenciada por tipo de fallo.
    """
    try:
        valid = schema.validate(df, lazy=True)
        return ValidationResult(
            valid,
            pd.DataFrame(),
            pd.DataFrame(),
            {"total": len(df), "valid": len(df)},
        )
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        failed_indices = failure_cases["index"].dropna().unique()

        is_discard = failure_cases["check"].apply(
            lambda c: any(str(c).startswith(prefix) for prefix in _DISCARD_CHECKS)
        )
        discard_indices = failure_cases.loc[is_discard, "index"].dropna().unique()
        quarantine_indices = [i for i in failed_indices if i not in discard_indices]

        valid_indices = df.index.difference(failed_indices)
        result = ValidationResult(
            valid=df.loc[valid_indices].copy(),
            quarantined=df.loc[quarantine_indices].copy(),
            discarded=df.loc[discard_indices].copy(),
            report={
                "total": len(df),
                "valid": len(valid_indices),
                "quarantined": len(quarantine_indices),
                "discarded": len(discard_indices),
                "failure_breakdown": failure_cases["check"].value_counts().to_dict(),
            },
        )
        logger.warning(
            "ingest.validation",
            valid=result.report["valid"],
            quarantined=result.report["quarantined"],
            discarded=result.report["discarded"],
        )
        return result
