"""Baseline RAGAS sobre el path single-pass RAG. Arnés puntual (no infra de la app).

DOS FASES (a propósito): la app fija `openai` 2.x (LiteLLM/Instructor) y el juez de
RAGAS necesita el stack `langchain` con `openai` 1.x — incompatibles en un mismo
proceso. Por eso separamos:

  1) generate: corre reformulación→retrieval→augmentation→generación por cada
     consulta del golden set (venv de la app, openai 2.x) y vuelca las 4 entradas
     RAGAS a un JSON intermedio (question, answer, contexts, ground_truth). Solo aquí
     se importa la app y se captura `contexts` (los `content` de los chunks, que el
     endpoint HTTP no expone).

  2) evaluate: lee ese JSON y calcula faithfulness, answer_relevancy,
     context_precision, context_recall con juez OpenAI + embeddings
     text-embedding-3-small. NO importa la app. Se corre en entorno aislado:

       uv run --no-project \\
         --with 'ragas>=0.2,<0.3' --with 'langchain-openai<0.3' \\
         --with 'openai<2' --with datasets --with pandas \\
         python scripts/measure_ragas.py evaluate

Fase 1 (en el venv de la app):
       uv run python scripts/measure_ragas.py generate

Requiere OPENAI_API_KEY (juez + embeddings) y la DB con el corpus ingerido.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# La fase generate importa el paquete `app`; aseguramos la raíz del repo en sys.path
# (los scripts no se instalan como paquete; equivalente a PYTHONPATH=.).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS_PATH = Path(__file__).parent / "ragas_rows.json"
GOLDEN = Path(__file__).parent / "golden_set.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# === Fase 1: generación (venv de la app, openai 2.x) =========================


def render_answer(estimate) -> str:
    """RagEstimate → texto legible para RAGAS (módulos, tareas, días, totales)."""
    lines = [f"Confianza: {estimate.confidence.value}. Razonamiento: {estimate.reasoning}"]
    for module in estimate.modules:
        lines.append(f"Módulo {module.name}:")
        for task in module.tasks:
            flag = " (asunción)" if task.is_assumption else ""
            lines.append(f"  - {task.title}: {task.engineer_days} días-ingeniero{flag}")
    lines.append(f"Total: {estimate.total_engineer_days} días-ingeniero.")
    return "\n".join(lines)


async def _build_row(entry: dict, *, wrapper, embedder, settings) -> dict:
    """Corre el pipeline single-pass para una consulta y arma las 4 entradas RAGAS."""
    from app.generation.rag.persistence.database import AsyncSessionLocal
    from app.generation.rag.retrieval.augmentation import assemble_context
    from app.generation.rag.retrieval.generation import generate_rag_estimate
    from app.generation.rag.retrieval.pipeline import RetrievalPipeline
    from app.generation.rag.retrieval.reformulation import reformulate_transcript

    reformulated = reformulate_transcript(
        transcript=entry["query"], wrapper=wrapper, settings=settings
    )
    pipeline = RetrievalPipeline(embedder=embedder, session_factory=AsyncSessionLocal)
    retrieval = await pipeline.retrieve(
        reformulated=reformulated,
        settings=settings,
        search_mode=settings.rag_search_mode,
        reranking=settings.reranking_enabled,
    )
    context = assemble_context(retrieval, max_tokens=settings.rag_max_context_tokens)
    estimate = generate_rag_estimate(
        reformulated=reformulated, context=context, wrapper=wrapper, settings=settings
    )
    return {
        "question": entry["query"],
        "answer": render_answer(estimate),
        "contexts": [chunk.content for chunk in retrieval.chunks],
        "ground_truth": entry["ground_truth"],
    }


async def _generate() -> None:
    from app.foundations.config import get_settings
    from app.foundations.llm_wrapper import LLMWrapper
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder

    settings = get_settings()
    wrapper, embedder = LLMWrapper(settings), LiteLLMEmbedder()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rows = [
        await _build_row(entry, wrapper=wrapper, embedder=embedder, settings=settings)
        for entry in golden["queries"]
    ]
    ROWS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escritas {len(rows)} filas en {ROWS_PATH}")


# === Fase 2: evaluación (entorno aislado, ragas + openai 1.x) ================


def _evaluate() -> None:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    # Nombres de columna de la API clásica de RAGAS 0.2.
    dataset = Dataset.from_dict({key: [r[key] for r in rows] for key in rows[0]})
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    print(df.to_string(index=False))
    present = [m for m in METRIC_NAMES if m in df.columns]
    print("\nPromedios:")
    print(df[present].mean().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RAGAS (single-pass RAG).")
    parser.add_argument("phase", choices=["generate", "evaluate"])
    args = parser.parse_args()
    if args.phase == "generate":
        asyncio.run(_generate())
    else:
        _evaluate()


if __name__ == "__main__":
    main()
