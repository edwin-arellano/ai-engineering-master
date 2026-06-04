"""Detección heurística de anclas en el historial.

Una "ancla" es un hecho crítico que NUNCA debe perderse en la compresión:
NDAs, contratos firmados, alcance ya cerrado, presupuesto bloqueado,
requisitos de compliance, deadlines, compromisos explícitos. Son temas que,
si el modelo los olvida, pueden causar problemas serios (dar información que
no debería, ignorar un compromiso legal).

Decisiones de diseño (siguiendo el directo):
- 100% heurístico, sin LLM: queremos una pieza determinista y barata.
- Solo escanea mensajes del USUARIO. Lo que dice el modelo no genera anclas;
  lo que importa es lo que el usuario ha comprometido o declarado.
- Pocas reglas a propósito. Demasiadas anclas confunden al modelo y capturan
  ruido. Añadir una regla nueva es trivial cuando se necesita.
- Solo son anclas los temas CRÍTICOS, no los opinables. "Me interesa la
  opinión de X" no es ancla; "hay un NDA firmado" sí lo es.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.session import ChatMessage


@dataclass(frozen=True)
class AnchorRule:
    """Una regla de detección de ancla."""

    label: str
    pattern: re.Pattern[str]


# Ocho reglas iniciales. Todas en inglés y español porque la conversación
# puede ser bilingüe.
ANCHOR_RULES: tuple[AnchorRule, ...] = (
    AnchorRule(
        "nda",
        re.compile(
            r"\b(nda|non[- ]disclosure|acuerdo de confidencialidad)\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "signed_contract",
        re.compile(
            r"\b(signed contract|contrato firmado|contrato ya firmado)\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "defined_scope",
        re.compile(
            r"\b(scope is (locked|closed|defined)|"
            r"alcance (cerrado|definido|acordado))\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "locked_budget",
        re.compile(
            r"\b(budget is (locked|fixed|capped)|"
            r"presupuesto (bloqueado|cerrado|fijo))\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "compliance",
        re.compile(
            r"\b(gdpr|hipaa|soc ?2|pci[- ]?dss|iso ?27001|"
            r"compliance|cumplimiento normativo)\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "deadline",
        re.compile(
            r"\b(hard deadline|deadline|fecha l[íi]mite|"
            r"entrega antes de|must ship by)\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "explicit_commitment",
        re.compile(
            r"\b(we committed|hemos comprometido|"
            r"compromiso (firme|explícito)|promised to)\b",
            re.IGNORECASE,
        ),
    ),
    AnchorRule(
        "legal_constraint",
        re.compile(
            r"\b(penalty clause|cláusula de penalización|"
            r"liability|responsabilidad legal|sla penalty)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_anchors(messages: list[ChatMessage]) -> list[str]:
    """Devuelve las anclas detectadas SOLO en los mensajes del usuario.

    El valor de cada ancla es una frase corta del tipo
    "[nda] <fragmento del mensaje>" para que sea legible en el contexto y
    auditable en logs. Deduplicado por contenido.
    """
    found: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message.role != "user":
            continue
        for rule in ANCHOR_RULES:
            match = rule.pattern.search(message.content)
            if not match:
                continue
            snippet = _context_snippet(message.content, match.start(), match.end())
            anchor = f"[{rule.label}] {snippet}"
            if anchor not in seen:
                seen.add(anchor)
                found.append(anchor)
    return found


def _context_snippet(text: str, start: int, end: int, window: int = 80) -> str:
    """Devuelve un fragmento de `text` alrededor del match, recortado."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].strip()
    return re.sub(r"\s+", " ", snippet)
