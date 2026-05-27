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

from app.config import get_settings
from app.core.llm_wrapper import LLMWrapper
from app.schemas.session import EstimationMode, Session
from app.services.llm_service import generate_estimation_in_session
from app.services.sessions import get_session_store

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def _check_case(result_dict: dict, expected: dict) -> list[str]:
    """Devuelve la lista de fallos (vacía si el caso pasa)."""
    failures: list[str] = []
    summary = result_dict.get("summary", "")
    is_out = summary.startswith("Out of scope:")

    if expected.get("out_of_scope"):
        if not is_out:
            failures.append("esperaba out_of_scope pero no lo es")
        return failures  # para out-of-scope no validamos lo demás

    if is_out:
        failures.append("out_of_scope inesperado")
        return failures

    phases = result_dict.get("phases", [])
    if "phase_count_range" in expected:
        low, high = expected["phase_count_range"]
        if not (low <= len(phases) <= high):
            failures.append(f"phase_count {len(phases)} fuera de [{low}, {high}]")

    if "cost_range_eur" in expected:
        low, high = expected["cost_range_eur"]
        cost = result_dict.get("total_cost_eur", 0)
        # Tolerancia generosa en primera pasada (50%).
        if not (low * 0.5 <= cost <= high * 1.5):
            failures.append(f"cost {cost} fuera de [{low}, {high}] (con tolerancia)")

    return failures


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
            failures = _check_case(result_dict, case["expected"])
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
