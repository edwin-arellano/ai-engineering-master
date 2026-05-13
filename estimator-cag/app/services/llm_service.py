"""Servicio LLM.

Tras pre-session-04 el módulo expone dos rutas distintas:

- `generate_estimation(...)`: flujo nuevo del endpoint /api/v1/estimate.
  Renderiza el prompt vía el loader Jinja2 (`render_estimation_prompt`)
  y delega en `LLMWrapper.complete(...)`.

- `build_legacy_system_prompt(...)` + constantes: flujo legacy de
  session-02/03 que sigue usando el endpoint /api/v1/estimate/stream.
  Eliminar cuando ese endpoint se migre o desaparezca.
"""

from __future__ import annotations

import random

import structlog

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES  # legacy, sigue alimentando el stream
from app.core.llm_wrapper import LLMWrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.schemas.legacy_estimation import (
    LegacyExampleFormat,
    LegacyOutputFormat,
    LegacyPreprocessingType,
)

logger = structlog.get_logger()


# === Wrapper singleton (compartido entre los dos flujos) ===

_wrapper: LLMWrapper | None = None


def get_wrapper() -> LLMWrapper:
    """Devuelve el singleton del wrapper, instanciándolo en la primera llamada."""
    global _wrapper
    if _wrapper is None:
        _wrapper = LLMWrapper()
    return _wrapper


# === Flujo nuevo (pre-session-04): generate_estimation con loader Jinja2 ===

async def generate_estimation(request: EstimationRequest) -> EstimationResponse:
    """Genera una estimación a partir del nuevo schema tipado.

    Flujo:
    1. Renderiza system y user del template Jinja2 versionado (v1 por defecto).
    2. Llama al wrapper (LiteLLM Router con fallback y cache).
    3. Devuelve el texto del LLM con la versión del prompt.

    Todas las features de session-02 (preprocessing, evaluation,
    thinking_budget, etc.) han desaparecido. Si en session-04 reaparecen,
    será con un diseño nuevo basado en structured outputs y guardrails.
    """
    wrapper = get_wrapper()
    settings = get_settings()
    prompt_version = "v1"

    system_prompt, user_message = render_estimation_prompt(
        request, version=prompt_version
    )

    llm_result = wrapper.complete(
        system_prompt=system_prompt,
        user_message=user_message,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    return EstimationResponse(
        text=llm_result["text"],
        prompt_version=prompt_version,
    )


# === Flujo legacy (session-02/03): build_legacy_system_prompt + constantes ===
# Sigue usado por /api/v1/estimate/stream. NO eliminar hasta migrar el stream.

_INLINE_CLEANING_BLOCK = """\
=== INPUT PREPROCESSING INSTRUCTION ===

The transcription you receive below may contain informal conversations,
divagations, off-topic comments, interruptions, and noise typical of
real meeting transcripts. Before producing your estimate:

1. Mentally extract only the functional and technical requirements
   relevant to the estimation.
2. Ignore personal divagations, off-topic conversations, technical
   asides, jokes, and interruptions.
3. Consolidate scattered mentions of the same requirement into a
   single coherent requirement.

Do NOT include the cleaned requirements in your output — only the
final estimate.
"""


_BASE_SYSTEM_PROMPT = """\
You are a senior software estimation consultant with 15+ years of
experience estimating custom development projects. Your job is to
analyze a meeting transcription with a client and produce a
structured, defensible estimation.

Do not invent technologies or features that are not mentioned in
the transcription. If the transcription is too vague or out of
scope (not a software project), say so explicitly and do not
invent numbers.
"""


_MARKDOWN_OUTPUT_INSTRUCTIONS = """\
=== OUTPUT FORMAT ===

Respond in Markdown with the following structure:

## <Project title>

### Desglose de tareas
1. <Task name> (<N> h)
2. <Task name> (<N> h)
...

**Total: <N> h**
**Equipo recomendado: <team composition>**
**Duración estimada: <N>-<M> semanas**

The sum of hours in the breakdown MUST equal the declared total.
"""


_JSON_OUTPUT_INSTRUCTIONS = """\
=== OUTPUT FORMAT ===

Respond with a single valid JSON object (no Markdown code fences,
no commentary before or after) with EXACTLY this structure:

{
  "title": "string — project title",
  "breakdown": [
    {"task": "string — task name", "hours": <integer>}
  ],
  "total_hours": <integer>,
  "team": "string — recommended team composition",
  "duration_weeks": "string — e.g. '6-8' or '10'"
}

Rules:
- The 'breakdown' array MUST contain at least 3 tasks.
- 'total_hours' MUST equal the sum of hours in 'breakdown'.
- All integer fields are whole numbers (no decimals).
"""


def _format_legacy_examples(
    num_examples: int,
    example_format: LegacyExampleFormat,
    deterministic: bool = False,
) -> str:
    """Formatea N ejemplos few-shot. Random por defecto, determinista si se pide.

    `deterministic=True` selecciona los primeros N ejemplos en orden, necesario
    para que el endpoint /estimate/stream produzca cache hits estables.
    """
    if example_format != LegacyExampleFormat.MARKDOWN:
        raise NotImplementedError(
            f"example_format={example_format!r} no soportado en el flujo legacy"
        )

    if num_examples <= 0:
        return ""

    total = len(ESTIMATION_EXAMPLES)
    n = min(num_examples, total)

    if deterministic:
        selected = ESTIMATION_EXAMPLES[:n]
    else:
        selected = random.sample(ESTIMATION_EXAMPLES, n)

    blocks: list[str] = []
    for index, example in enumerate(selected, start=1):
        blocks.append(
            f"===== REFERENCE ESTIMATION {index} =====\n\n"
            f"Meeting summary: {example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}"
        )
    blocks.append("===== END OF REFERENCE ESTIMATIONS =====")
    return "\n\n".join(blocks)


def build_legacy_system_prompt(
    num_examples: int,
    example_format: LegacyExampleFormat,
    output_format: LegacyOutputFormat,
    preprocessing: LegacyPreprocessingType,
    deterministic: bool = False,
) -> str:
    """Compone el system prompt del flujo legacy (sin cambios funcionales).

    Solo lo invoca el endpoint /api/v1/estimate/stream. El endpoint
    /api/v1/estimate ya no pasa por aquí: usa el loader Jinja2.
    """
    parts: list[str] = []

    if preprocessing == LegacyPreprocessingType.INLINE_CLEANING:
        parts.append(_INLINE_CLEANING_BLOCK)

    parts.append(_BASE_SYSTEM_PROMPT)

    if output_format == LegacyOutputFormat.JSON:
        parts.append(_JSON_OUTPUT_INSTRUCTIONS)
    else:
        parts.append(_MARKDOWN_OUTPUT_INSTRUCTIONS)

    examples_block = _format_legacy_examples(num_examples, example_format, deterministic)
    if examples_block:
        parts.append(examples_block)

    return "\n\n".join(parts)
