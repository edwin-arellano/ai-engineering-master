"""Servicio de generación de estimaciones usando el LLM wrapper."""

from __future__ import annotations

import random
import time

import structlog

from app.context.examples import ESTIMATION_EXAMPLES
from app.core.llm_wrapper import LLMWrapper
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    ExampleFormat,
    OutputFormat,
    PreprocessingType,
    TokenUsage,
)
from app.services.evaluation_service import evaluate_estimation

logger = structlog.get_logger()


# === System prompt building blocks ===

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


_REQUIREMENTS_EXTRACTION_SYSTEM = """\
You are a senior business analyst with 15+ years of experience
reading software development meeting transcripts. Your job is to
read the transcript provided by the user and produce a clean,
deduplicated list of:

1. Functional requirements — what the system must DO.
2. Non-functional requirements — performance, security, accessibility,
   technology constraints, deployment, etc.
3. Project constraints — timeline, budget, team composition, integrations.

Output as a Markdown document with three sections (one per category),
each with a bulleted list. Be concise, do not include personal
divagations, off-topic comments, jokes, or interruptions. Do NOT
estimate anything — only extract the requirements as stated.
"""


def _format_examples(
    num_examples: int,
    example_format: ExampleFormat,
    deterministic: bool = False,
) -> str:
    """Formatea N ejemplos. Selección aleatoria por defecto, determinista si se pide.

    `deterministic=True` selecciona los primeros N ejemplos en orden. Esto es
    necesario para el endpoint /estimate/stream: sin determinismo, dos llamadas
    idénticas producirían system prompts distintos (por random.sample) y la
    cache exact-match nunca haría hit.
    """
    if example_format != ExampleFormat.MARKDOWN:
        raise NotImplementedError(
            f"example_format={example_format!r} no soportado en session-03"
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


def build_system_prompt(
    num_examples: int,
    example_format: ExampleFormat,
    output_format: OutputFormat,
    preprocessing: PreprocessingType,
    deterministic: bool = False,
) -> str:
    """Compone el system prompt según las opciones del request.

    `deterministic=True` se usa desde el endpoint /estimate/stream para que la
    cache exact-match funcione (mismo input → misma respuesta cacheable).
    """
    parts: list[str] = []

    if preprocessing == PreprocessingType.INLINE_CLEANING:
        parts.append(_INLINE_CLEANING_BLOCK)

    parts.append(_BASE_SYSTEM_PROMPT)

    if output_format == OutputFormat.JSON:
        parts.append(_JSON_OUTPUT_INSTRUCTIONS)
    else:
        parts.append(_MARKDOWN_OUTPUT_INSTRUCTIONS)

    examples_block = _format_examples(num_examples, example_format, deterministic)
    if examples_block:
        parts.append(examples_block)

    return "\n\n".join(parts)


# === Wrapper singleton ===

_wrapper: LLMWrapper | None = None


def get_wrapper() -> LLMWrapper:
    """Devuelve el singleton del wrapper, instanciándolo en la primera llamada."""
    global _wrapper
    if _wrapper is None:
        _wrapper = LLMWrapper()
    return _wrapper


# === Public API ===

async def generate_estimation(request: EstimationRequest) -> EstimationResponse:
    """Orquesta el flujo completo del endpoint /estimate (no-stream).

    Mantiene todas las features de session-02: preprocessing inline/two-phase,
    evaluation estructural, num_examples random, thinking_budget,
    output_format markdown/json.

    Lo único que cambia respecto a session-02 es que la llamada al LLM pasa
    por el wrapper LiteLLM en lugar de _call_anthropic/_call_openai directos.
    """
    wrapper = get_wrapper()
    started_at = time.time()

    # === Fase 1: preprocesado opcional ===
    extracted_requirements: str | None = None
    preprocessing_input_tokens = 0
    preprocessing_output_tokens = 0

    if request.preprocessing == PreprocessingType.TWO_PHASE:
        extraction = wrapper.complete(
            system_prompt=_REQUIREMENTS_EXTRACTION_SYSTEM,
            user_message=request.transcription,
            max_tokens=2000,
            temperature=0.2,
            thinking_budget=0,
        )
        extracted_requirements = extraction["text"]
        preprocessing_input_tokens = extraction["input_tokens"]
        preprocessing_output_tokens = extraction["output_tokens"]

    # === Fase 2: construcción del system prompt ===
    # NOTA: deterministic=False (default) mantiene el comportamiento random
    # de session-02 para preservar el demo del punto de saturación.
    system_prompt = build_system_prompt(
        num_examples=request.num_examples,
        example_format=request.example_format,
        output_format=request.output_format,
        preprocessing=request.preprocessing,
        deterministic=False,
    )

    # === Fase 3: llamada al LLM vía wrapper ===
    user_input = extracted_requirements or request.transcription
    llm_result = wrapper.complete(
        system_prompt=system_prompt,
        user_message=user_input,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        thinking_budget=request.thinking_budget,
    )

    # === Fase 4: evaluación opcional ===
    evaluation = None
    if request.evaluation:
        evaluation = evaluate_estimation(
            estimation_text=llm_result["text"],
            output_format=request.output_format,
            finish_reason=llm_result["finish_reason"],
        )

    # === Fase 5: construcción de la respuesta ===
    latency_ms = int((time.time() - started_at) * 1000)

    token_usage = None
    if request.usage:
        token_usage = TokenUsage(
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            total_tokens=llm_result["input_tokens"] + llm_result["output_tokens"],
            preprocessing_input_tokens=preprocessing_input_tokens,
            preprocessing_output_tokens=preprocessing_output_tokens,
        )

    return EstimationResponse(
        estimation=llm_result["text"],
        model=llm_result["model"],
        provider=llm_result["provider"],
        finish_reason=llm_result["finish_reason"],
        preprocessing_type=request.preprocessing,
        output_format=request.output_format,
        latency_ms=latency_ms,
        token_usage=token_usage,
        extracted_requirements=extracted_requirements,
        evaluation=evaluation,
        cache_hit=llm_result.get("cache_hit", False),
    )
