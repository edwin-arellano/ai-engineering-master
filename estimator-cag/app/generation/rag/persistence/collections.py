"""Registro SearchTarget → modelo ORM de su tabla. Punto único de resolución de
colección a tabla: el router decide los `SearchTarget`, el pipeline los traduce a
modelos ORM con este mapa y los pasa al repositorio (parametrizado por `model`)."""

from __future__ import annotations

from app.generation.rag.persistence.models import (
    BudgetChunkRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)
from app.generation.rag.schemas import SearchTarget

COLLECTION_MODELS = {
    SearchTarget.BUDGETS: BudgetChunkRow,
    SearchTarget.TRANSCRIPTS: TranscriptChunkRow,
    SearchTarget.TECHNICAL_DOCS: TechnicalDocChunkRow,
}
