"""Script de evaluación contra el golden dataset.

Uso:
    uv run python -m evals.run                 # todos los casos, modo actor
    uv run python -m evals.run --max-cases 8   # primeros 8
    uv run python -m evals.run --mode actor_critic_boss

NO forma parte de la aplicación ni de la suite de pytest. Es una herramienta
de evaluación al estilo test de integración: input conocido, criterios de
salida conocidos, pass/fail por caso.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import LLMWrapper
from app.domain.session import EstimationMode, Session
from app.generation.cag.llm_service import generate_estimation_in_session
from app.generation.cag.sessions import get_session_store
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    Metric,
    PhaseCountMetric,
    SchemaAdherenceMetric,
    run_all_metrics,
)

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def _metrics_for_case(expected: dict) -> list[Metric]:
    """Construye las métricas aplicables a un caso a partir de su `expected`.

    Para casos out-of-scope solo se evalúa la adherencia de schema (igual que el
    `_check_case` original, que no validaba nada más en esos casos).
    """
    metrics: list[Metric] = [
        SchemaAdherenceMetric(expected_out_of_scope=expected.get("out_of_scope", False))
    ]
    if not expected.get("out_of_scope"):
        if "cost_range_eur" in expected:
            low, high = expected["cost_range_eur"]
            metrics.append(CostBoundsMetric(low=low, high=high))
        if "phase_count_range" in expected:
            low, high = expected["phase_count_range"]
            metrics.append(PhaseCountMetric(low=low, high=high))
        # `project_name_contains` es un término obligatorio único → require_all.
        if expected.get("project_name_contains"):
            metrics.append(
                ContentRecallMetric(
                    expected_terms=[expected["project_name_contains"]],
                    require_all=True,
                )
            )
        # `technologies_any_of` son alternativas → basta con que aparezca una.
        if expected.get("technologies_any_of"):
            metrics.append(
                ContentRecallMetric(
                    expected_terms=expected["technologies_any_of"],
                    require_all=False,
                )
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--mode", default="actor", choices=["actor", "actor_critic_boss"]
    )
    args = parser.parse_args()

    settings = get_settings()
    wrapper = LLMWrapper(settings)
    store = get_session_store()
    mode = EstimationMode(args.mode)

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.max_cases:
        cases = cases[: args.max_cases]

    passed = 0
    for index, case in enumerate(cases, start=1):
        session = Session(estimation_mode=mode)
        store.save(session)
        try:
            response = generate_estimation_in_session(
                session=session,
                transcript=case["transcript"],
                project_type=case["project_type"],
                detail_level=case["detail_level"],
                output_format=case["output_format"],
                attachments=[],
                wrapper=wrapper,
                session_store=store,
                settings=settings,
            )
            result_dict = response.result.model_dump()
            results = run_all_metrics(_metrics_for_case(case["expected"]), result_dict)
            failures = [r.name for r in results if not r.passed]
        except Exception as exc:  # noqa: BLE001
            failures = [f"excepción: {exc}"]

        status = "PASS" if not failures else "FAIL"
        if not failures:
            passed += 1
        print(f"[{index:02d}/{len(cases)}] {status} — {case['id']}")
        for failure in failures:
            print(f"        ↳ {failure}")

    print(f"\nResultado: {passed}/{len(cases)} casos pasan ({args.mode}).")


if __name__ == "__main__":
    main()
