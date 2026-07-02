"""Fase 5 (verificación) + invariantes de RagEstimate. Sin LLM ni DB."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.rag_estimation import (
    Citation,
    Confidence,
    RagEstimate,
    RagModule,
    RagTask,
)
from app.generation.rag.retrieval.verification import (
    enforce_confidence_coherence,
    verify_citations,
)
from app.generation.rag.schemas import AugmentedContext


def _context(refs: list[str]) -> AugmentedContext:
    return AugmentedContext(
        context_block="...", token_count=10, included_refs=refs, dropped=0
    )


def _cite(source_id: str) -> Citation:
    return Citation(source_id=source_id, document_id="BUD-2024-001", evidence="5 días")


def _estimate_with_sources(source_ids: list[str]) -> RagEstimate:
    task = RagTask(
        title="Auth backend",
        engineer_days=5.0,
        sources=[_cite(s) for s in source_ids],
    )
    return RagEstimate(
        confidence=Confidence.HIGH,
        reasoning="derivado de los bloques",
        modules=[RagModule(name="Backend", tasks=[task])],
        total_engineer_days=5.0,
    )


# --- verify_citations ----------------------------------------------------------


def test_verify_citations_flags_invented_source():
    estimate = _estimate_with_sources(["BUD::REAL", "BUD::FAKE"])
    report = verify_citations(estimate, _context(["BUD::REAL"]))
    assert report.dangling == ["BUD::FAKE"]
    assert report.grounded_lines == 0  # la tarea tiene una cita colgante


def test_verify_citations_no_false_positives_when_all_present():
    estimate = _estimate_with_sources(["BUD::A", "BUD::B"])
    report = verify_citations(estimate, _context(["BUD::A", "BUD::B", "BUD::C"]))
    assert report.dangling == []
    assert report.grounded_lines == 1
    assert report.total_lines == 1


def test_enforce_confidence_coherence_passes_for_insufficient():
    estimate = RagEstimate(confidence=Confidence.INSUFFICIENT, reasoning="sin contexto")
    enforce_confidence_coherence(estimate)  # no debe lanzar


# --- model_validators de RagEstimate -------------------------------------------


def test_task_without_source_must_be_assumption():
    with pytest.raises(ValidationError):
        RagTask(title="huérfana", engineer_days=2.0)  # sin sources ni is_assumption


def test_assumption_task_is_allowed_without_sources():
    task = RagTask(title="asunción razonable", engineer_days=2.0, is_assumption=True)
    assert task.sources == []


def test_insufficient_with_modules_is_rejected():
    with pytest.raises(ValidationError):
        RagEstimate(
            confidence=Confidence.INSUFFICIENT,
            reasoning="incoherente",
            modules=[
                RagModule(
                    name="X",
                    tasks=[
                        RagTask(
                            title="t",
                            engineer_days=1.0,
                            sources=[_cite("BUD::A")],
                        )
                    ],
                )
            ],
            total_engineer_days=1.0,
        )


def test_totals_must_match_sum_of_tasks():
    with pytest.raises(ValidationError):
        RagEstimate(
            confidence=Confidence.MEDIUM,
            reasoning="totales que no cuadran",
            modules=[
                RagModule(
                    name="X",
                    tasks=[
                        RagTask(
                            title="t",
                            engineer_days=3.0,
                            sources=[_cite("BUD::A")],
                        )
                    ],
                )
            ],
            total_engineer_days=10.0,  # 3.0 != 10.0 (> ±0.5)
        )


def test_totals_within_tolerance_are_accepted():
    estimate = RagEstimate(
        confidence=Confidence.MEDIUM,
        reasoning="dentro de tolerancia",
        modules=[
            RagModule(
                name="X",
                tasks=[
                    RagTask(
                        title="t",
                        engineer_days=3.0,
                        sources=[_cite("BUD::A")],
                    )
                ],
            )
        ],
        total_engineer_days=3.4,  # |3.0 - 3.4| = 0.4 <= 0.5
    )
    assert estimate.confidence == Confidence.MEDIUM
