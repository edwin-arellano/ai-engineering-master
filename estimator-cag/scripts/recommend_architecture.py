"""CLI: recomienda arquitectura (CAG/RAG/híbrido) a partir del perfil del corpus,
el modelo y el baseline empírico pre-S06 (evals/stress/results.csv).

    uv run python -m scripts.recommend_architecture
"""

from __future__ import annotations

from pathlib import Path

from app.foundations.config import get_settings
from app.ingest.architecture import (
    CorpusProfile,
    IngestionArchitecture,
    ModelProfile,
    summarize_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()
    results_csv = ROOT / "evals" / "stress" / "results.csv"
    if not results_csv.exists():
        raise SystemExit(f"No existe el baseline en {results_csv}")

    baseline = summarize_baseline(results_csv)

    # Perfil del corpus de estimación (Proyecto 2): catálogo de presupuestos +
    # transcripciones + tarifario. Requiere citar fuentes en la respuesta final.
    corpus = CorpusProfile(
        total_tokens=2_000_000,
        update_frequency_days=30,
        requires_source_attribution=True,
        requires_per_user_access_control=False,
    )
    # Claude Haiku 4.5 como modelo primario del servicio.
    model = ModelProfile(
        context_window=200_000,
        cost_per_million_input_tokens=0.80,
    )

    arch = IngestionArchitecture(
        corpus=corpus,
        model=model,
        baseline=baseline,
        latency_sla_seconds=settings.cag_latency_sla_seconds,
        cost_per_turn_budget_usd=settings.cag_cost_per_turn_budget_usd,
        usable_ratio=settings.cag_usable_context_ratio,
    )

    viability = arch.viability()
    print("## Baseline empírico pre-S06 (evals/stress/results.csv)")
    print(f"- Turnos medidos: {baseline.turns}")
    print(f"- Latencia P50: {baseline.latency_p50:.2f} s")
    print(f"- Latencia P95: {baseline.latency_p95:.2f} s")
    print(f"- Coste medio/turno: ${baseline.cost_per_turn_mean:.6f}")

    print("\n## Viabilidad CAG")
    print(f"- Cabe en ventana de contexto: {viability.fits_in_context_window}")
    print(f"- Coste aceptable: {viability.cost_per_query_acceptable}")
    print(
        f"- Latencia aceptable (P95<={settings.cag_latency_sla_seconds}s): "
        f"{viability.latency_acceptable}"
    )
    print(f"- Calidad se sostiene bajo carga: {viability.quality_holds_with_load}")
    print(f"- ¿CAG viable?: {viability.is_viable()}")

    print(f"\n## Recomendación: {arch.recommend().value.upper()}")


if __name__ == "__main__":
    main()
