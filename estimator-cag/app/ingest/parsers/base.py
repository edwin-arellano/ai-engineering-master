"""Capa de extracción por formato. Un parser produce una representación
INTERMEDIA (DataFrame para tabular, lista de turnos para transcripción), NO el
Document canónico todavía. La conversión a Document la hace el normalizer. Tres
capas (loader/parser/normalizer) en vez de dos por testabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class ParsedSource:
    """Representación intermedia. Solo uno de los dos campos según el formato."""

    kind: str  # "tabular" | "text_turns"
    dataframe: pd.DataFrame | None = None
    records: list[dict[str, Any]] = field(default_factory=list)


class Parser(Protocol):
    supported_formats: set[str]

    def parse(self, content: bytes, source_hint: str) -> ParsedSource: ...
