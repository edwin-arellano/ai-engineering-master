"""Reporte de auditoría legible a partir del catálogo (deliverable para no-técnicos)."""

from __future__ import annotations

from app.ingest.catalog.models import DataCatalog, IngestionDecision


def generate_audit_report(catalog: DataCatalog) -> str:
    included = catalog.included_sources()
    excluded = [s for s in catalog.sources if s.decision == IngestionDecision.EXCLUDE]
    review = [s for s in catalog.sources if s.decision == IngestionDecision.REVIEW]

    lines = [
        f"# Reporte de auditoría de datos — {catalog.last_audited}",
        "",
        f"**Fuentes auditadas:** {len(catalog.sources)} | "
        f"**incluidas:** {len(included)} | "
        f"**excluidas:** {len(excluded)} | "
        f"**en revisión:** {len(review)}",
        "",
        "## Fuentes incluidas",
        "",
    ]
    for s in included:
        lines.append(
            f"- **{s.name}** ({s.format}, {s.volume.records} registros) — "
            f"owner: {s.owner_business}, última actualización: {s.refresh.observed_last_update}"
        )
    if review:
        lines += ["", "## En revisión", ""]
        for s in review:
            lines.append(
                f"- **{s.name}** — motivo: {s.decision_reason or 'ver catálogo'}"
            )
    if excluded:
        lines += ["", "## Excluidas", ""]
        for s in excluded:
            lines.append(
                f"- **{s.name}** — motivo: {s.decision_reason or s.notes or 'ver catálogo'}"
            )
    return "\n".join(lines)
