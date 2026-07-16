"""Dependencias del grafo, construidas UNA vez al arranque y capturadas por closures
en los nodos. No viajan por `config` (evita serializar objetos no triviales en el
checkpointer). Reutilizables entre requests: el pipeline abre sesión por-llamada vía
factory y AsyncOpenAI está pensado para reuso."""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.foundations.config import Settings
from app.foundations.llm_wrapper import LLMWrapper
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval.pipeline import RetrievalPipeline


@dataclass(frozen=True)
class GraphDeps:
    settings: Settings
    wrapper: LLMWrapper
    pipeline: RetrievalPipeline
    client: AsyncOpenAI


def build_deps(settings: Settings) -> GraphDeps:
    return GraphDeps(
        settings=settings,
        wrapper=LLMWrapper(settings),
        pipeline=RetrievalPipeline(
            embedder=LiteLLMEmbedder(), session_factory=AsyncSessionLocal
        ),
        client=AsyncOpenAI(api_key=settings.openai_api_key),
    )
