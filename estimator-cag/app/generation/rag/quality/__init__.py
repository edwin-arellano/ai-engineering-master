"""Paquete de calidad de generación (S11): capa que convierte el RAG single-pass de
"parece acertada" en un sistema donde cada cifra es trazable y degradable.

- numeric_anchor: ancla numérica DETERMINISTA (cifra de la evidencia ↔ días-ingeniero).
- judge: juez LLM estricto por línea (semántico; lo que la verificación estructural no cubre).
- gate: gate_line puro + apply_gate → degrada líneas a cero + DegradationReport.
- synthesis: synthesize_range → rangos honestos (HourRange) con detección de contradicción.
- curation: gate de indexabilidad del corpus (garbage-in-garbage-out).
"""

from __future__ import annotations

from app.generation.rag.quality.gate import (
    DegradationReport,
    LineGate,
    apply_gate,
    gate_line,
)
from app.generation.rag.quality.judge import JudgeVerdicts, LineVerdict, judge_lines
from app.generation.rag.quality.numeric_anchor import AnchorResult, anchor_line
from app.generation.rag.quality.synthesis import synthesize_range

__all__ = [
    "AnchorResult",
    "anchor_line",
    "JudgeVerdicts",
    "LineVerdict",
    "judge_lines",
    "LineGate",
    "DegradationReport",
    "gate_line",
    "apply_gate",
    "synthesize_range",
]
