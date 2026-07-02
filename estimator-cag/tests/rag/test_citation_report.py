"""CitationReport (S11): verificación estructural por línea. Sin LLM ni DB.
Clasifica cada tarea en grounded / insufficient (asunción) / con cita colgante."""

from __future__ import annotations

from app.domain.rag_estimation import (
    Citation,
    Confidence,
    RagEstimate,
    RagModule,
    RagTask,
)
from app.generation.rag.retrieval.verification import verify_citations
from app.generation.rag.schemas import AugmentedContext


def _context(refs: list[str]) -> AugmentedContext:
    return AugmentedContext(context_block="...", token_count=10, included_refs=refs, dropped=0)


def _cite(source_id: str) -> Citation:
    return Citation(source_id=source_id, document_id="BUD-2024-001", evidence="3 días")


def _grounded_task(source_id: str) -> RagTask:
    return RagTask(title=f"t-{source_id}", engineer_days=1.0, sources=[_cite(source_id)])


def _assumption_task() -> RagTask:
    return RagTask(title="asunción", engineer_days=1.0, is_assumption=True)


def _estimate(tasks: list[RagTask]) -> RagEstimate:
    return RagEstimate(
        confidence=Confidence.MEDIUM,
        reasoning="mezcla de fundamentadas y asunciones",
        modules=[RagModule(name="M", tasks=tasks)],
        total_engineer_days=float(len(tasks)),
    )


def test_all_grounded_yields_no_dangling():
    estimate = _estimate([_grounded_task("BUD::A"), _grounded_task("BUD::B")])
    report = verify_citations(estimate, _context(["BUD::A", "BUD::B"]))
    assert report.total_lines == 2
    assert report.grounded_lines == 2
    assert report.insufficient_lines == 0
    assert report.dangling == []


def test_invented_source_id_appears_in_dangling():
    estimate = _estimate([_grounded_task("BUD::REAL"), _grounded_task("BUD::FAKE")])
    report = verify_citations(estimate, _context(["BUD::REAL"]))
    assert report.dangling == ["BUD::FAKE"]
    assert report.grounded_lines == 1  # solo la tarea con cita resuelta


def test_assumption_counts_as_insufficient_not_grounded():
    estimate = _estimate([_grounded_task("BUD::A"), _assumption_task()])
    report = verify_citations(estimate, _context(["BUD::A"]))
    assert report.total_lines == 2
    assert report.grounded_lines == 1
    assert report.insufficient_lines == 1
    assert report.dangling == []
