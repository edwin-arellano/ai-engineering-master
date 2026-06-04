"""CLI: inspecciona las fuentes del seed, pide una pista al evaluador LLM y
vuelca ``data/catalog/data_catalog.yaml``.

Flujo (pasos 1-2 del viaje del dato):
    1. ``inspect_filesystem_source`` -> folder facts + muestreo estructural.
    2. ``evaluate_source`` (LLM, opcional) -> pista de calidad/sensibilidad/decisión
       sobre SOLO esos facts (nunca contenido crudo).
    3. Se combinan facts + juicio LLM + campos curados a mano (owners, lineage,
       cadencia declarada, decisión revisada) en un ``CatalogSource``.
    4. Se serializa el ``DataCatalog`` a YAML.

El catálogo es CÓDIGO revisado por un humano: la `decision` final la fija el spec
curado (no el LLM), que solo aporta una segunda opinión registrada en
`decision_reason`. Con ``--offline`` se omite la llamada al LLM y todo queda
determinista (útil para reproducir el catálogo sin red).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from app.foundations.config import get_settings
from app.ingest.catalog.inspect import inspect_filesystem_source
from app.ingest.catalog.models import (
    CatalogSource,
    DataCatalog,
    IngestionDecision,
    Lineage,
    Quality,
    Refresh,
    Sensitivity,
    Volume,
)

ROOT = Path(__file__).resolve().parents[1]


# Specs curados por un auditor humano: lo que el LLM no puede saber (owners,
# linaje upstream, cadencia declarada) y la decisión revisada por fuente.
_SOURCE_SPECS = [
    {
        "name": "budgets",
        "subdir": "budgets",
        "format": "json",
        "owner_technical": "data-platform@empresa.com",
        "owner_business": "Dirección Comercial",
        "declared_cadence": "on_close",
        "upstream": "CRM export (presupuestos firmados)",
        "transformations": ["export JSON por presupuesto"],
        "decision": IngestionDecision.INCLUDE,
        "fallback_description": "Presupuestos históricos firmados, una entrada JSON por presupuesto.",
        "fallback_quality": Quality(
            completeness=4, consistency=4, actuality=4, reliability=4
        ),
        "fallback_sensitivity": Sensitivity(
            contains_pii=True,
            pii_types=["client_name"],
            access_restrictions="interno",
        ),
    },
    {
        "name": "transcripts",
        "subdir": "transcripts",
        "format": "txt",
        "owner_technical": "data-platform@empresa.com",
        "owner_business": "Oficina de Proyectos",
        "declared_cadence": "per_meeting",
        "upstream": "Notas/transcripciones de reuniones de alcance",
        "transformations": ["volcado a texto plano"],
        "decision": IngestionDecision.REVIEW,
        "fallback_description": "Transcripciones de reuniones; conviven ficheros modernos con tags de speaker y ficheros legacy sin atribuir.",
        "fallback_quality": Quality(
            completeness=3, consistency=2, actuality=4, reliability=3
        ),
        "fallback_sensitivity": Sensitivity(
            contains_pii=True,
            pii_types=["speaker_name"],
            access_restrictions="interno",
        ),
    },
    {
        "name": "rates",
        "subdir": "rates",
        "format": "xlsx",
        "owner_technical": "finanzas@empresa.com",
        "owner_business": "Finanzas",
        "declared_cadence": "yearly",
        "upstream": "Tarifario interno anual",
        "transformations": ["mantenimiento manual en Excel"],
        "decision": IngestionDecision.EXCLUDE,
        "fallback_description": "Tarifario de roles; la copia disponible está obsoleta (>365 días).",
        "fallback_quality": Quality(
            completeness=4, consistency=4, actuality=1, reliability=3
        ),
        "fallback_sensitivity": Sensitivity(
            contains_pii=False,
            pii_types=[],
            access_restrictions="confidencial",
        ),
    },
]


def _build_source(spec: dict, *, use_llm: bool, wrapper, settings) -> CatalogSource:
    root = ROOT / settings.ingest_seed_dir / spec["subdir"]
    facts = inspect_filesystem_source(spec["name"], root, project_root=ROOT)

    quality = spec["fallback_quality"]
    sensitivity = spec["fallback_sensitivity"]
    description = spec["fallback_description"]
    llm_hint = "sin evaluación LLM (modo offline)"

    if use_llm:
        # ejerce el flujo Q1: el LLM solo ve facts + muestreo estructural.
        from app.ingest.catalog.evaluator import evaluate_source

        judgment = evaluate_source(facts, wrapper=wrapper, settings=settings)
        quality = judgment.quality
        sensitivity = judgment.sensitivity
        description = judgment.suggested_description
        llm_hint = f"LLM sugirió '{judgment.decision.value}': {judgment.decision_reason}"

    return CatalogSource(
        name=spec["name"],
        description=description,
        location=str(root.relative_to(ROOT)),
        owner_technical=spec["owner_technical"],
        owner_business=spec["owner_business"],
        format=spec["format"],
        volume=Volume(records=facts.file_count, size_mb=facts.total_size_mb),
        refresh=Refresh(
            declared=spec["declared_cadence"],
            observed_last_update=facts.latest_modified.date().isoformat(),
            observed_lag_days=facts.observed_lag_days,
        ),
        quality=quality,
        sensitivity=sensitivity,
        lineage=Lineage(
            upstream=spec["upstream"], transformations=spec["transformations"]
        ),
        decision=spec["decision"],
        decision_reason=f"Decisión revisada a mano. {llm_hint}",
        notes=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="No consultar el LLM; usar juicios de calidad/sensibilidad por defecto.",
    )
    args = parser.parse_args()
    use_llm = not args.offline

    settings = get_settings()
    wrapper = None
    if use_llm:
        from app.foundations.llm_wrapper import LLMWrapper

        wrapper = LLMWrapper(settings)

    sources = [
        _build_source(spec, use_llm=use_llm, wrapper=wrapper, settings=settings)
        for spec in _SOURCE_SPECS
    ]
    catalog = DataCatalog(
        version=1,
        last_audited=datetime.now().date().isoformat(),
        sources=sources,
    )

    out_path = ROOT / settings.data_catalog_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = catalog.model_dump(mode="json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Catálogo de fuentes de datos (S06). CÓDIGO revisado a mano.\n")
        f.write("# Generado por scripts/inspect_sources.py; editar con criterio.\n")
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    print(f"Catálogo escrito en {out_path.relative_to(ROOT)}")
    for s in catalog.sources:
        print(f"  - {s.name}: {s.decision.value}")


if __name__ == "__main__":
    main()
