"""Contrato de EstimateFromTranscriptResult (S11): el campo back-compat
`invalid_citations` debe seguir el `dangling` del citation_report. Sin LLM ni DB."""

from __future__ import annotations

from app.domain.rag_estimation import (
    Citation,
    Confidence,
    RagEstimate,
    RagModule,
    RagTask,
)
from app.generation.rag.quality import DegradationReport
from app.generation.rag.retrieval.service import EstimateFromTranscriptResult
from app.generation.rag.retrieval.verification import CitationReport, verify_citations
from app.generation.rag.schemas import AugmentedContext


def _estimate(source_id: str) -> RagEstimate:
    task = RagTask(
        title="t",
        engineer_days=2.0,
        sources=[Citation(source_id=source_id, document_id="BUD-2024-002", evidence="2 días")],
    )
    return RagEstimate(
        confidence=Confidence.MEDIUM,
        reasoning="x",
        modules=[RagModule(name="M", tasks=[task])],
        total_engineer_days=2.0,
    )


def _result(estimate: RagEstimate, report: CitationReport) -> EstimateFromTranscriptResult:
    return EstimateFromTranscriptResult(
        estimate=estimate,
        retrieved_chunks=1,
        retrieved_budget_ids=["BUD-2024-002"],
        context_tokens=10,
        citation_report=report,
        invalid_citations=report.dangling,  # mismo cableado que el servicio
        degradation_report=DegradationReport(
            total_lines=1, degraded_lines=0, verified_lines=1, gates=[]
        ),
        search_time_ms=1,
        search_mode="hybrid",
        reranking=True,
    )


def test_invalid_citations_mirrors_dangling_when_dangling():
    estimate = _estimate("BUD::FAKE")
    report = verify_citations(
        estimate,
        AugmentedContext(context_block="...", token_count=10, included_refs=["BUD::REAL"], dropped=0),
    )
    result = _result(estimate, report)
    assert result.invalid_citations == result.citation_report.dangling == ["BUD::FAKE"]


def test_invalid_citations_empty_when_all_grounded():
    estimate = _estimate("BUD::REAL")
    report = verify_citations(
        estimate,
        AugmentedContext(context_block="...", token_count=10, included_refs=["BUD::REAL"], dropped=0),
    )
    result = _result(estimate, report)
    assert result.invalid_citations == result.citation_report.dangling == []
