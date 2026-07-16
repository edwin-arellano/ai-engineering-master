"""Script de traza del agente S12 (entregable del ejercicio).

Uso:
  PYTHONPATH=. uv run python scripts/run_agent_s12.py \
      examples/transcripts/sample_transcript_simple.txt --model gpt-5-mini --stub
  PYTHONPATH=. uv run python scripts/run_agent_s12.py \
      examples/transcripts/sample_transcript_complex.txt --model gpt-5

Depura primero el BUCLE con --stub + gpt-5-mini + transcripción simple (barato); luego
gpt-5 (medium) + transcripción compleja para la ejecución real. Coste total < ~$2-3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from app.foundations.config import get_settings
from app.generation.agentic.agent import run_agent
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.database import AsyncSessionLocal
from app.generation.rag.retrieval.pipeline import RetrievalPipeline


def _render_trace(result) -> str:
    lines: list[str] = []
    for step in result.trace:
        obs = json.dumps(step.observation, ensure_ascii=False)
        obs = obs if len(obs) <= 300 else obs[:297] + "..."
        lines.append(
            f"STEP {step.step}\n"
            f"  reasoning: {step.reasoning or '(sin resumen de razonamiento)'}\n"
            f"  action: {step.action}({json.dumps(step.args, ensure_ascii=False)})\n"
            f"  observation: {obs}"
        )
    tail = f"\nstatus: {result.status} | steps: {result.steps}"
    if result.estimate is not None:
        tail += (
            f"\nestimate: total={result.estimate.total_hours}h across "
            f"{len(result.estimate.components)} components"
        )
    return "\n".join(lines) + tail


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", help="Ruta al .txt de transcripción")
    parser.add_argument("--model", default=None, help="Modelo (default: settings.agent_model)")
    parser.add_argument("--stub", action="store_true", help="Usa reference_retrieval (sin BD)")
    parser.add_argument("--out", default=None, help="Ruta de salida de la traza")
    args = parser.parse_args()

    settings = get_settings()
    transcript = Path(args.transcript).read_text(encoding="utf-8")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    pipeline = RetrievalPipeline(embedder=LiteLLMEmbedder(), session_factory=AsyncSessionLocal)
    result = await run_agent(
        transcript,
        client=client,
        pipeline=pipeline,
        settings=settings,
        model=args.model,
        stub=args.stub,
    )
    rendered = _render_trace(result)
    print(rendered)
    out = Path(args.out) if args.out else Path(args.transcript).with_suffix(".agent_trace.out.txt")
    out.write_text(rendered + "\n", encoding="utf-8")
    print(f"\n[traza escrita en {out}]")


if __name__ == "__main__":
    asyncio.run(_main())
