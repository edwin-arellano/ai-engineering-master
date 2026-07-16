"""Gate de alucinaciones (S11): gate_line puro + apply_gate (degradar a cero y
recalcular total sin violar los validadores). Sin LLM."""

from __future__ import annotations

from app.domain.rag_estimation import (
    Citation,
    Confidence,
    RagEstimate,
    RagModule,
    RagTask,
)
from app.generation.rag.quality.gate import apply_gate, gate_line
from app.generation.rag.quality.judge import LineVerdict
from app.generation.rag.quality.numeric_anchor import AnchorResult


def _ok_anchor() -> AnchorResult:
    return AnchorResult(evidence_days=5.0, line_days=5.0, deviation=0.0, numeric_fail=False)


def _fail_anchor() -> AnchorResult:
    return AnchorResult(evidence_days=5.0, line_days=20.0, deviation=3.0, numeric_fail=True)


def test_gate_line_degrades_on_numeric_fail():
    gate = gate_line(index=0, is_assumption=False, anchor=_fail_anchor(), verdict=None)
    assert gate.degraded is True
    assert any("desviación numérica" in r for r in gate.reasons)


def test_gate_line_degrades_when_judge_rejects():
    verdict = LineVerdict(index=0, supported=False, reason="el scope no coincide")
    gate = gate_line(index=0, is_assumption=False, anchor=_ok_anchor(), verdict=verdict)
    assert gate.degraded is True
    assert any("juez" in r for r in gate.reasons)


def test_gate_line_grounded_and_supported_is_not_degraded():
    verdict = LineVerdict(index=0, supported=True, reason="ok")
    gate = gate_line(index=0, is_assumption=False, anchor=_ok_anchor(), verdict=verdict)
    assert gate.degraded is False
    assert gate.reasons == []


def test_assumption_is_annotated_not_degraded():
    # Las asunciones ya van sin horas por diseño: se anotan pero no se "degradan".
    gate = gate_line(index=0, is_assumption=True, anchor=_ok_anchor(), verdict=None)
    assert gate.degraded is False
    assert any("asunción" in r for r in gate.reasons)


def _two_task_estimate() -> RagEstimate:
    def task(title: str) -> RagTask:
        return RagTask(
            title=title,
            engineer_days=5.0,
            sources=[Citation(source_id="BUD::A", document_id="BUD-2024-001", evidence="40 horas")],
        )

    return RagEstimate(
        confidence=Confidence.MEDIUM,
        reasoning="dos tareas",
        modules=[RagModule(name="M", tasks=[task("t0"), task("t1")])],
        total_engineer_days=10.0,
    )


def test_apply_gate_zeroes_degraded_and_recalculates_total():
    estimate = _two_task_estimate()
    gates = {
        0: gate_line(index=0, is_assumption=False, anchor=_fail_anchor(), verdict=None),  # degrada
        1: gate_line(index=1, is_assumption=False, anchor=_ok_anchor(), verdict=None),  # conserva
    }
    degraded, report = apply_gate(estimate, gates)

    tasks = degraded.modules[0].tasks
    assert tasks[0].engineer_days == 0.0
    assert tasks[1].engineer_days == 5.0
    assert degraded.total_engineer_days == 5.0  # recalculado
    assert report.degraded_lines == 1
    assert report.verified_lines == 1
    # La estimación degradada sigue respetando totals_match_sum_of_tasks al re-validar.
    RagEstimate.model_validate(degraded.model_dump())
