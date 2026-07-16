"""Helpers compartidos por las estrategias de chunking. Centralizan el conteo de
tokens, los contextual chunk headers del presupuesto padre, la metadata filtrable
y la detección de huérfanos.
"""

from __future__ import annotations

import re

import tiktoken

from app.foundations.config import get_settings
from app.generation.rag.schemas import Budget, BudgetComponent


def get_tokenizer(model: str = "text-embedding-3-small") -> "tiktoken.Encoding":
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


_TOKENIZER = get_tokenizer()


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def is_orphan(token_count: int) -> bool:
    return token_count < get_settings().chunk_orphan_min_tokens


def build_parent_context(budget: Budget) -> str:
    c = budget.client_metadata
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {c.sector} | Year: {budget.year} | "
        f"Main tech: {budget.main_technology}]"
    )


def render_component_text(component: BudgetComponent, parent_context: str) -> str:
    return (
        f"{parent_context}\n\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def render_task_text(component: BudgetComponent, parent_context: str) -> str:
    """Texto de una tarea atómica (S09): foco en el qué + complejidad + horas, con
    headers del padre para contexto. Más corto que render_component_text."""
    return (
        f"{parent_context}\n\n"
        f"Task: {component.name}\n"
        f"Detail: {component.description}\n"
        f"Complexity: {component.complexity} | Hours: {component.estimated_hours}"
    )


def build_metadata(
    component: BudgetComponent, budget: Budget, *, strategy: str, **extra
) -> dict:
    return {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
        "strategy": strategy,
        # S11 (curación): indexable por defecto; `extra` puede marcarlo a False para
        # excepciones/casos límite que falsearían los vectores (garbage-in-garbage-out).
        "indexable": True,
        **extra,
    }


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def split_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Trozos de max_tokens con solapamiento (fixed-size mecánico)."""
    tokens = _TOKENIZER.encode(text)
    if not tokens:
        return []
    out, start = [], 0
    step = max(1, max_tokens - overlap)
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        out.append(_TOKENIZER.decode(window))
        start += step
    return out


def recursive_split(
    text: str, max_tokens: int, separators: list[str] | None = None
) -> list[str]:
    """Divide por separadores naturales sin exceder max_tokens (mecánico, sin overlap)."""
    separators = separators or ["\n\n", "\n", ". ", " "]
    if count_tokens(text) <= max_tokens:
        return [text]
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return split_by_tokens(text, max_tokens, overlap=0)
    parts, buffer, out = text.split(sep), "", []
    for part in parts:
        candidate = (buffer + sep + part) if buffer else part
        if count_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                out.append(buffer)
            buffer = part if count_tokens(part) <= max_tokens else ""
            if not buffer:
                out.extend(recursive_split(part, max_tokens, separators[1:]))
    if buffer:
        out.append(buffer)
    return out
