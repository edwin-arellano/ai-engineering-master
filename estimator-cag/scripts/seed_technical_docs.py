"""Siembra corpus de documentación técnica de apoyo (S10) a partir de
data/budgets_sample.json: un `.md` por `main_technology` distinto del corpus, con
secciones de referencia (visión general, integraciones típicas, riesgos de estimación).

CORPUS SINTÉTICO / SEED: material de referencia DESCRIPTIVO. NO inventa cifras de
esfuerzo (horas) — esas viven solo en los presupuestos. Describe stacks, integraciones
observadas y riesgos cualitativos de estimación derivados del propio corpus.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SAMPLE = Path("data/budgets_sample.json")
OUTPUT_DIR = Path("data/technical_docs")

# Nombre legible por slug de tecnología (los slugs salen del corpus).
_TECH_LABELS = {
    "java": "Java",
    "node_js": "Node.js",
    "python": "Python",
    "ruby_on_rails": "Ruby on Rails",
}


def _label(tech: str) -> str:
    return _TECH_LABELS.get(tech, tech.replace("_", " ").title())


def _render_doc(tech: str, budgets: list[dict]) -> str:
    label = _label(tech)
    # Integraciones/stack observadas en los componentes (sin horas, solo descriptivo).
    tech_stacks: set[str] = set()
    component_names: set[str] = set()
    sectors: set[str] = set()
    for budget in budgets:
        sectors.add(budget["client_metadata"]["sector"])
        for component in budget["components"]:
            component_names.add(component["name"])
            tech_stacks.update(component.get("tech_stack", []))

    stack_lines = "\n".join(f"- {item}" for item in sorted(tech_stacks)) or "- (sin datos)"
    capability_lines = (
        "\n".join(f"- {name}" for name in sorted(component_names)) or "- (sin datos)"
    )
    sector_list = ", ".join(sorted(sectors)) or "varios"

    return f"""# Referencia técnica — {label}

> Documento de referencia interno (corpus sembrado, S10). Material DESCRIPTIVO para
> apoyar el routing y la recuperación; **no contiene cifras de esfuerzo**. Las horas
> de estimación viven exclusivamente en los presupuestos históricos.

## Visión general

{label} es la tecnología principal de varios proyectos del corpus histórico, en los
sectores: {sector_list}. Esta referencia resume el stack y las integraciones técnicas
observadas para esa base tecnológica, como contexto de apoyo a la estimación.

## Integraciones típicas

Componentes técnicos de stack frecuentes en proyectos {label} del corpus:

{stack_lines}

Capacidades funcionales recurrentes construidas sobre {label}:

{capability_lines}

## Riesgos de estimación

Consideraciones cualitativas al estimar proyectos {label} (sin cuantificar esfuerzo):

- La complejidad real depende del grado de integración con sistemas externos y del
  cumplimiento regulatorio del sector ({sector_list}).
- Las capacidades transversales (autenticación, idempotencia, observabilidad) tienden
  a subestimarse cuando no aparecen como componentes explícitos.
- La cercanía de un proyecto nuevo a los históricos de esta tecnología es el mejor
  indicador de fiabilidad de la estimación por analogía.
"""


def seed_technical_docs(*, sample: Path = SAMPLE, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Genera un `.md` por main_technology distinta. Devuelve las rutas escritas."""
    budgets = json.loads(sample.read_text(encoding="utf-8"))
    by_tech: dict[str, list[dict]] = defaultdict(list)
    for budget in budgets:
        by_tech[budget["main_technology"]].append(budget)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tech, tech_budgets in sorted(by_tech.items()):
        path = output_dir / f"{tech}.md"
        path.write_text(_render_doc(tech, tech_budgets), encoding="utf-8")
        written.append(path)
        print(f"  [OK] {path} ({len(tech_budgets)} budgets)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(SAMPLE))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    seed_technical_docs(sample=Path(args.sample), output_dir=Path(args.output))


if __name__ == "__main__":
    main()
