"""Pipeline de evaluación de generación (S11-07). Extiende measure_ragas con tres modos
y con la lectura de los reportes de calidad (degradación/citación). Como measure_ragas,
corre en DOS FASES por el conflicto openai 2.x (app) / openai 1.x (juez RAGAS):

  generate: corre el flujo single-pass completo (augment+gate+synthesis según la variante)
            por cada consulta del golden set y vuelca filas + conteos de calidad a JSON.
            Venv de la app (openai 2.x).
  evaluate: lee ese JSON, calcula las métricas RAGAS y aplica la lógica del modo. Entorno
            aislado (ragas + openai 1.x), sin importar la app.

Modos (CLI --mode):
  gate    (offline/CI): variante 'full' (toda la calidad on); compara contra
          scripts/eval_baseline.json con tolerancia 0.05; floor = baseline − tolerancia;
          FAIL (exit≠0) si alguna métrica cae por debajo del floor.
  monitor (producción, reference-less): sin ground_truth → NO calcula context_recall;
          solo faithfulness + answer_relevancy (+ context_precision sin referencia si está).
  compare: corre variantes por fase (baseline S10 = todo off; +hallucination_gate;
          +augmentation; +synthesis) y muestra los deltas por métrica de cada fase.

CUIDADO: tarda ~30-40 min y cuesta ~$2-3/run (muchas llamadas al LLM). Correr con criterio.

Uso:
  PYTHONPATH=. uv run python scripts/eval_generation.py generate --mode compare
  uv run --no-project --with 'ragas>=0.2,<0.3' --with 'langchain-openai<0.3' \\
    --with 'openai<2' --with datasets --with pandas \\
    python scripts/eval_generation.py evaluate --mode compare
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS_PATH = Path(__file__).parent / "eval_rows.json"
BASELINE_PATH = Path(__file__).parent / "eval_baseline.json"
GOLDEN = Path(__file__).parent / "golden_set.json"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
MONITOR_METRICS = ["faithfulness", "answer_relevancy"]  # reference-less (sin context_recall)
REGRESSION_TOLERANCE = 0.05

# Variantes por fase para el modo compare (aditivas sobre el baseline S10 = todo off).
_ALL_OFF = {
    "context_compression_enabled": False,
    "reorder_by_edges_enabled": False,
    "hallucination_gate_enabled": False,
    "synthesis_enabled": False,
}
COMPARE_VARIANTS = {
    "baseline": {**_ALL_OFF},
    "+hallucination_gate": {**_ALL_OFF, "hallucination_gate_enabled": True},
    "+augmentation": {**_ALL_OFF, "context_compression_enabled": True, "reorder_by_edges_enabled": True},
    "+synthesis": {**_ALL_OFF, "synthesis_enabled": True},
}
FULL_ON = {
    "context_compression_enabled": True,
    "reorder_by_edges_enabled": True,
    "hallucination_gate_enabled": True,
    "synthesis_enabled": True,
}


def _variants_for_mode(mode: str) -> dict[str, dict]:
    if mode == "compare":
        return COMPARE_VARIANTS
    return {"full": {**FULL_ON}}  # gate y monitor corren el flujo completo


# === Fase 1: generación (venv de la app, openai 2.x) =========================


async def _run_flow(query: str, *, wrapper, embedder, settings) -> dict:
    """Replica el flujo single-pass (fases 1-7) capturando contexts + reportes de calidad.
    Mira a los mismos helpers que retrieval/service.py (el harness necesita los contexts,
    que el servicio no expone)."""
    from app.generation.rag.persistence.database import AsyncSessionLocal
    from app.generation.rag.quality import (
        anchor_line,
        apply_gate,
        gate_line,
        judge_lines,
        synthesize_range,
    )
    from app.generation.rag.retrieval.augmentation import assemble_context
    from app.generation.rag.retrieval.generation import generate_rag_estimate
    from app.generation.rag.retrieval.pipeline import RetrievalPipeline
    from app.generation.rag.retrieval.reformulation import reformulate_transcript
    from app.generation.rag.retrieval.verification import verify_citations

    from scripts.measure_ragas import render_answer

    reformulated = reformulate_transcript(transcript=query, wrapper=wrapper, settings=settings)
    pipeline = RetrievalPipeline(embedder=embedder, session_factory=AsyncSessionLocal)
    retrieval = await pipeline.retrieve(
        reformulated=reformulated, settings=settings,
        search_mode=settings.rag_search_mode, reranking=settings.reranking_enabled,
    )
    context = assemble_context(
        retrieval, max_tokens=settings.rag_max_context_tokens, settings=settings
    )
    estimate = generate_rag_estimate(
        reformulated=reformulated, context=context, wrapper=wrapper, settings=settings
    )
    citation = verify_citations(estimate, context)

    degraded_lines = 0
    if settings.hallucination_gate_enabled:
        lines, anchors, assumptions, idx = [], {}, {}, 0
        for module in estimate.modules:
            for task in module.tasks:
                evidence = task.sources[0].evidence if task.sources else ""
                anchors[idx] = anchor_line(
                    line_engineer_days=task.engineer_days, evidence=evidence,
                    hours_per_day=settings.hours_per_engineer_day,
                    tolerance=settings.numeric_deviation_tolerance,
                )
                assumptions[idx] = task.is_assumption
                lines.append({"index": idx, "title": task.title,
                              "engineer_days": task.engineer_days, "evidence": evidence})
                idx += 1
        verdicts = (
            await judge_lines(lines=lines, wrapper=wrapper, settings=settings)
            if settings.judge_enabled else {}
        )
        gates = {i: gate_line(index=i, is_assumption=assumptions[i], anchor=anchors[i],
                              verdict=verdicts.get(i)) for i in anchors}
        estimate, degradation = apply_gate(estimate, gates)
        degraded_lines = degradation.degraded_lines

    if settings.synthesis_enabled:
        by_ref = {c.chunk_ref: c for c in retrieval.chunks}
        for module in estimate.modules:
            for task in module.tasks:
                hours = [by_ref[s.source_id].metadata.get("estimated_hours")
                         for s in task.sources if s.source_id in by_ref]
                rng = synthesize_range([float(h) for h in hours if h is not None],
                                       wrapper=wrapper, settings=settings, context=task.title)
                if rng is not None:
                    task.hour_range = rng

    return {
        "question": query,
        "answer": render_answer(estimate),
        "contexts": [c.content for c in retrieval.chunks],
        "degraded_lines": degraded_lines,
        "total_lines": citation.total_lines,
        "dangling": citation.dangling,
    }


async def _generate(mode: str) -> None:
    from app.foundations.config import get_settings
    from app.foundations.llm_wrapper import LLMWrapper
    from app.generation.rag.embedding.embedder import LiteLLMEmbedder

    base = get_settings()
    wrapper, embedder = LLMWrapper(base), LiteLLMEmbedder()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    variants = _variants_for_mode(mode)

    out: dict = {"mode": mode, "variants": {}}
    for name, overrides in variants.items():
        settings = base.model_copy(update=overrides)
        rows = []
        for entry in golden["queries"]:
            row = await _run_flow(entry["query"], wrapper=wrapper, embedder=embedder, settings=settings)
            row["ground_truth"] = entry["ground_truth"]
            rows.append(row)
        out["variants"][name] = rows
        deg = sum(r["degraded_lines"] for r in rows)
        print(f"[generate] variante '{name}': {len(rows)} filas, {deg} líneas degradadas")
    ROWS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {ROWS_PATH}")


# === Fase 2: evaluación (entorno aislado, ragas + openai 1.x) ================


def _ragas_metrics(rows: list[dict], *, metric_names: list[str]):
    """Calcula las métricas RAGAS pedidas sobre unas filas. Devuelve dict metric→media."""
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

    by_name = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [by_name[n] for n in metric_names]
    cols = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    }
    if "context_recall" in metric_names or "context_precision" in metric_names or "faithfulness" in metric_names:
        cols["ground_truth"] = [r.get("ground_truth", "") for r in rows]
    dataset = Dataset.from_dict(cols)
    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    result = evaluate(dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    df = result.to_pandas()
    return {n: float(df[n].mean()) for n in metric_names if n in df.columns}


def _evaluate(mode: str) -> None:
    data = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    variants = data["variants"]

    if mode == "monitor":
        rows = variants["full"]
        means = _ragas_metrics(rows, metric_names=MONITOR_METRICS)
        print("=== MONITOR (reference-less; sin context_recall) ===")
        for name, value in means.items():
            print(f"  {name}: {value:.3f}")
        return

    if mode == "gate":
        rows = variants["full"]
        means = _ragas_metrics(rows, metric_names=METRIC_NAMES)
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if BASELINE_PATH.exists() else {}
        print("=== GATE (regresión offline; tolerancia 0.05) ===")
        failed = False
        for name in METRIC_NAMES:
            current = means.get(name)
            if current is None:
                continue
            base = baseline.get(name)
            if base is None:
                print(f"  {name}: {current:.3f} (sin baseline; se registra)")
                continue
            floor = base - REGRESSION_TOLERANCE
            ok = current >= floor
            failed = failed or not ok
            print(f"  {name}: {current:.3f} vs baseline {base:.3f} (floor {floor:.3f}) → {'PASS' if ok else 'FAIL'}")
        deg = sum(r["degraded_lines"] for r in rows)
        print(f"  líneas degradadas: {deg}")
        print("GLOBAL:", "FAIL" if failed else "PASS")
        if failed:
            sys.exit(1)
        return

    # compare
    print("=== COMPARE (deltas por fase vs baseline) ===")
    per_variant = {name: _ragas_metrics(rows, metric_names=METRIC_NAMES) for name, rows in variants.items()}
    base = per_variant.get("baseline", {})
    header = "variante".ljust(22) + "".join(n[:14].ljust(16) for n in METRIC_NAMES) + "degradadas"
    print(header)
    for name, means in per_variant.items():
        row = name.ljust(22)
        for metric in METRIC_NAMES:
            value = means.get(metric)
            if value is None:
                row += "—".ljust(16)
                continue
            delta = value - base.get(metric, value)
            sign = "+" if delta >= 0 else ""
            row += f"{value:.3f} ({sign}{delta:.3f})".ljust(16)
        deg = sum(r["degraded_lines"] for r in variants[name])
        row += str(deg)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de evaluación de generación (S11).")
    parser.add_argument("phase", choices=["generate", "evaluate"])
    parser.add_argument("--mode", choices=["gate", "monitor", "compare"], default="gate")
    args = parser.parse_args()
    if args.phase == "generate":
        asyncio.run(_generate(args.mode))
    else:
        _evaluate(args.mode)


if __name__ == "__main__":
    main()
