"""Test de integración del evaluador LLM (pega al modelo real)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config import get_settings
from app.core.llm_wrapper import LLMWrapper
from app.ingest.catalog.evaluator import evaluate_source
from app.ingest.catalog.inspect import FilesystemSourceFacts
from app.ingest.catalog.models import IngestionDecision


@pytest.mark.integration
def test_obsolete_rate_card_is_excluded():
    facts = FilesystemSourceFacts(
        name="rates",
        path="data/seed/rates",
        file_count=1,
        total_size_mb=0.005,
        latest_modified=datetime(2023, 1, 1, tzinfo=timezone.utc),
        observed_lag_days=480,  # > 365 -> debería excluirse
        formats_detected=["xlsx"],
        structural_sample={"xlsx_columns": ["role", "seniority", "hourly_rate_eur"]},
    )
    settings = get_settings()
    wrapper = LLMWrapper(settings)
    judgment = evaluate_source(facts, wrapper=wrapper, settings=settings)
    assert judgment.decision == IngestionDecision.EXCLUDE
