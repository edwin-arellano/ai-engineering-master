"""Ancla numérica determinista (S11): conversión y desviación sin LLM."""

from __future__ import annotations

from app.generation.rag.quality.numeric_anchor import anchor_line


def test_hours_convert_to_days_number_first():
    result = anchor_line(
        line_engineer_days=15.0, evidence="120 horas", hours_per_day=8.0, tolerance=0.25
    )
    assert result.evidence_days == 15.0  # 120 / 8
    assert result.deviation == 0.0
    assert result.numeric_fail is False


def test_hours_convert_to_days_unit_first():
    # Formato real del corpus: la unidad va antes del número.
    result = anchor_line(
        line_engineer_days=11.25, evidence="Estimated hours: 90", hours_per_day=8.0, tolerance=0.25
    )
    assert result.evidence_days == 11.25  # 90 / 8
    assert result.numeric_fail is False


def test_deviation_above_tolerance_fails():
    # Evidencia 120 horas ≈ 15 días, pero la línea afirma 40 días → desviación alta.
    result = anchor_line(
        line_engineer_days=40.0, evidence="Hours: 120", hours_per_day=8.0, tolerance=0.25
    )
    assert result.evidence_days == 15.0
    assert result.deviation is not None and result.deviation > 0.25
    assert result.numeric_fail is True


def test_evidence_without_number_does_not_block():
    result = anchor_line(
        line_engineer_days=5.0,
        evidence="autenticación OAuth con multi-tenant",
        hours_per_day=8.0,
        tolerance=0.25,
    )
    assert result.evidence_days is None
    assert result.deviation is None
    assert result.numeric_fail is False  # sin cifra parseable → no degrada
