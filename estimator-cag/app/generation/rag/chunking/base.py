"""Interfaz común para cualquier estrategia de chunking. Toda estrategia recibe
list[Budget] y devuelve list[Chunk]. Las que parten texto largo respetan
chunk_max_tokens; las que usan LLM reciben el wrapper por inyección.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.generation.rag.schemas import Budget, Chunk


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, budgets: list[Budget]) -> list[Chunk]: ...
