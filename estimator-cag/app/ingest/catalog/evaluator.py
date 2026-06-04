"""Evaluación de fuentes vía LLM (Q1).

Recibe SOLO folder facts + muestreo estructural (claves/columnas/flags), nunca
contenido crudo. Devuelve un juicio que el CLI combina con los campos manuales
(owners, lineage) para materializar un CatalogSource.

NOTA DE ARQUITECTURA: en producción este juicio debería correr contra un modelo
pequeño on-prem (open source, en local) para no exponer ni siquiera metadatos
sensibles a un tercero. Aquí se usa el wrapper actual como punto de intercambio;
el contrato (facts -> CatalogSourceJudgment) no cambia si se sustituye el backend.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.config import Settings
from app.core.llm_wrapper import LLMWrapper
from app.core.metrics import TurnMetrics
from app.ingest.catalog.inspect import FilesystemSourceFacts
from app.ingest.catalog.models import IngestionDecision, Quality, Sensitivity
from app.prompts.loader import render_catalog_evaluator_prompt


class CatalogSourceJudgment(BaseModel):
    """Juicio del LLM sobre una fuente, basado solo en hechos."""

    decision: IngestionDecision
    decision_reason: str = Field(min_length=1)
    suggested_description: str = Field(min_length=1)
    quality: Quality
    sensitivity: Sensitivity


def _facts_to_user_message(facts: FilesystemSourceFacts) -> str:
    payload = {
        "name": facts.name,
        "path": facts.path,
        "file_count": facts.file_count,
        "total_size_mb": facts.total_size_mb,
        "latest_modified": facts.latest_modified.isoformat(),
        "observed_lag_days": facts.observed_lag_days,
        "formats_detected": facts.formats_detected,
        "structural_sample": facts.structural_sample,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evaluate_source(
    facts: FilesystemSourceFacts,
    *,
    wrapper: LLMWrapper,
    settings: Settings,
    metrics: TurnMetrics | None = None,
) -> CatalogSourceJudgment:
    """Pide al LLM un juicio sobre la fuente a partir de hechos factuales.

    Reutiliza el wrapper síncrono existente (``complete_structured``) y el
    prompt loader (``render_catalog_evaluator_prompt``); no inventa firmas.
    El ``temperature=0.0`` busca determinismo en una decisión de auditoría.
    """
    system_prompt = render_catalog_evaluator_prompt(
        settings.catalog_evaluator_prompt_version
    )
    return wrapper.complete_structured(
        system_prompt=system_prompt,
        user_message=_facts_to_user_message(facts),
        response_model=CatalogSourceJudgment,
        max_tokens=1500,
        temperature=0.0,
        metrics=metrics,
    )
