from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.foundations.config import get_settings
from app.ingest.catalog.loader import load_catalog
from app.ingest.catalog.models import IngestionDecision
from app.ingest.orchestrator import run_ingestion
from app.domain.ingestion import CatalogResponse, IngestionRunResponse

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


def _project_root() -> Path:
    # raíz de estimator-cag/ (app/routers/ingestion.py -> parents[2])
    return Path(__file__).resolve().parents[2]


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    settings = get_settings()
    path = _project_root() / settings.data_catalog_path
    if not path.exists():
        raise HTTPException(
            404, f"Catálogo no encontrado en {settings.data_catalog_path}"
        )
    catalog = load_catalog(path)
    by = {
        d: [s.name for s in catalog.sources if s.decision == d]
        for d in IngestionDecision
    }
    return CatalogResponse(
        version=catalog.version,
        last_audited=catalog.last_audited,
        included=by[IngestionDecision.INCLUDE],
        review=by[IngestionDecision.REVIEW],
        excluded=by[IngestionDecision.EXCLUDE],
    )


# endpoint SÍNCRONO a propósito: FastAPI lo corre en threadpool. Bloqueante por
# diseño (session-07 lo hará no-bloqueante con BackgroundTasks/pipeline).
@router.post("", response_model=IngestionRunResponse)
def run_ingest() -> IngestionRunResponse:
    settings = get_settings()
    root = _project_root()
    path = root / settings.data_catalog_path
    if not path.exists():
        raise HTTPException(
            404, f"Catálogo no encontrado en {settings.data_catalog_path}"
        )
    catalog = load_catalog(path)
    result = run_ingestion(catalog, project_root=root)
    return IngestionRunResponse(
        documents_produced=len(result.documents),
        rejected_sources=result.rejected_sources,
        quarantined_count=result.quarantined_count,
        discarded_count=result.discarded_count,
        validation_reports=result.validation_reports,
    )
