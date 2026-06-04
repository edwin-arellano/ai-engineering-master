"""CLI: carga el catálogo, ejecuta la ingesta y reporta el resultado.

    uv run python -m scripts.run_ingestion
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingest.catalog.loader import load_catalog
from app.ingest.catalog.report import generate_audit_report
from app.ingest.orchestrator import run_ingestion

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()
    catalog_path = ROOT / settings.data_catalog_path
    if not catalog_path.exists():
        raise SystemExit(
            f"No existe el catálogo en {catalog_path}. "
            "Genera primero con: uv run python -m scripts.inspect_sources"
        )

    catalog = load_catalog(catalog_path)
    print(generate_audit_report(catalog))
    print()

    result = run_ingestion(catalog, project_root=ROOT)
    print("## Resultado de la ingesta")
    print(f"- Documentos producidos: {len(result.documents)}")
    print(f"- Fuentes rechazadas: {result.rejected_sources}")
    print(f"- En cuarentena: {result.quarantined_count}")
    print(f"- Descartados: {result.discarded_count}")
    for source_name, report in result.validation_reports.items():
        print(f"- Validación '{source_name}': {report}")


if __name__ == "__main__":
    main()
