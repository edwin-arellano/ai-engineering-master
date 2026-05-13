"""Loader de templates Jinja2 para los prompts versionados.

Punto único donde el código Python toca los templates. Cualquier consumidor
del prompt (endpoint, test, script) pasa por aquí.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

# Directorio raíz de los templates: app/prompts/
PROMPTS_DIR = Path(__file__).parent

# Environment compartido. Configuración crítica:
# - StrictUndefined: variables no definidas rompen el render con error claro,
#   en lugar de renderizarse como cadena vacía y producir un prompt malformado.
# - trim_blocks + lstrip_blocks: los bloques {% %} no introducen saltos
#   de línea espurios.
# - keep_trailing_newline=False: el render no añade \n al final.
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza el par (system, user) para el endpoint /estimate.

    El parámetro `version` apunta al subdirectorio bajo `estimation/`.
    Cambiar a `v2` sin tocar el resto del código permite A/B y rollback.
    """
    system_template = _env.get_template(f"estimation/{version}/system.j2")
    user_template = _env.get_template(f"estimation/{version}/user.j2")

    context = {
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
    }

    system = system_template.render(**context)
    user = user_template.render(**context)
    return system, user
