"""Curación del corpus (S11): gate de indexabilidad determinista. Sin LLM ni DB."""

from __future__ import annotations

from app.generation.rag.quality.curation import is_indexable


def test_default_metadata_is_indexable():
    verdict = is_indexable(metadata={"budget_id": "BUD-2024-001", "estimated_hours": 90})
    assert verdict.indexable is True
    assert verdict.reasons == []


def test_explicit_false_flag_rejected():
    verdict = is_indexable(metadata={"indexable": False})
    assert verdict.indexable is False
    assert any("no indexable" in r for r in verdict.reasons)


def test_client_exception_rejected():
    verdict = is_indexable(metadata={"is_exception": True})
    assert verdict.indexable is False
    assert any("excepción" in r for r in verdict.reasons)


def test_client_specific_rejected():
    verdict = is_indexable(metadata={"client_specific": True})
    assert verdict.indexable is False
