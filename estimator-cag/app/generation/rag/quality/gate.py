"""Gate de alucinaciones: combina el ancla numérica y el veredicto del juez en una
decisión PURA de degradación. Las líneas degradadas van a cero horas (aunque traigan
valor), con sus razones. Reconstruye la estimación recalculando totales para no violar
`totals_match_sum_of_tasks`. Se habla en márgenes de fiabilidad, nunca correcto/incorrecto."""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from app.domain.rag_estimation import RagEstimate, RagModule, RagTask
from app.generation.rag.quality.judge import LineVerdict
from app.generation.rag.quality.numeric_anchor import AnchorResult

logger = structlog.get_logger(__name__)


class LineGate(BaseModel):
    index: int
    degraded: bool
    reasons: list[str]


class DegradationReport(BaseModel):
    total_lines: int
    degraded_lines: int
    verified_lines: int
    gates: list[LineGate]


def gate_line(
    *,
    index: int,
    is_assumption: bool,
    anchor: AnchorResult,
    verdict: LineVerdict | None,
) -> LineGate:
    """PURA: degrada si no-grounded, o falla numérico, o el juez rechaza. Las asunciones
    ya van sin horas por diseño: se anotan pero no se 'degradan' (no hay cifra que anular)."""
    reasons: list[str] = []
    if is_assumption:
        reasons.append("asunción sin evidencia")
    if anchor.numeric_fail:
        reasons.append(
            f"desviación numérica {anchor.deviation:.2f}: línea {anchor.line_days}d "
            f"vs evidencia {anchor.evidence_days}d"
        )
    if verdict is not None and not verdict.supported:
        reasons.append(f"juez: {verdict.reason}")
    return LineGate(index=index, degraded=bool(reasons and not is_assumption), reasons=reasons)


def apply_gate(
    estimate: RagEstimate, gates: dict[int, LineGate]
) -> tuple[RagEstimate, DegradationReport]:
    """Devuelve una estimación degradada (líneas con gate.degraded → engineer_days=0) y el
    reporte. Recalcula total_engineer_days para respetar totals_match_sum_of_tasks."""
    idx = 0
    new_modules: list[RagModule] = []
    line_gates: list[LineGate] = []
    for module in estimate.modules:
        new_tasks: list[RagTask] = []
        for task in module.tasks:
            gate = gates.get(idx, LineGate(index=idx, degraded=False, reasons=[]))
            line_gates.append(gate)
            days = 0.0 if gate.degraded else task.engineer_days
            new_tasks.append(task.model_copy(update={"engineer_days": days}))
            idx += 1
        new_modules.append(module.model_copy(update={"tasks": new_tasks}))
    degraded = sum(1 for g in line_gates if g.degraded)
    new_total = (
        sum(t.engineer_days for m in new_modules for t in m.tasks) if new_modules else None
    )
    degraded_estimate = estimate.model_copy(
        update={"modules": new_modules, "total_engineer_days": new_total}
    )
    report = DegradationReport(
        total_lines=len(line_gates),
        degraded_lines=degraded,
        verified_lines=len(line_gates) - degraded,
        gates=line_gates,
    )
    if degraded:
        logger.warning("rag.degraded_lines", count=degraded)  # request_id auto vía contextvars
    return degraded_estimate, report
