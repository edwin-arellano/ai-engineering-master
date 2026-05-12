"""Servicio de generación de estimaciones vía LLM con arquitectura CAG."""

from __future__ import annotations

import random
import time

from anthropic import Anthropic
from fastapi import HTTPException
from openai import OpenAI

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    ExampleFormat,
    OutputFormat,
    PreprocessingType,
    TokenUsage,
)
from app.services.evaluation_service import evaluate_estimation


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


def _format_examples(num_examples: int, example_format: ExampleFormat) -> str:
    """Selecciona aleatoriamente N ejemplos y los formatea con separadores claros.

    La selección aleatoria es deliberada: en la sesión en vivo se observó que
    rotar los ejemplos entre llamadas mejora la robustez del modelo frente a
    ligeras variaciones en la transcripción.
    """
    if example_format != ExampleFormat.MARKDOWN:
        raise NotImplementedError(
            f"example_format={example_format!r} no está soportado todavía en session-02"
        )

    if num_examples <= 0:
        return ""

    total = len(ESTIMATION_EXAMPLES)
    n = min(num_examples, total)
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
) -> str:
    """Compone el system prompt completo según las opciones del request."""
    parts: list[str] = []

    if preprocessing == PreprocessingType.INLINE_CLEANING:
        parts.append(_INLINE_CLEANING_BLOCK)

    parts.append(_BASE_SYSTEM_PROMPT)

    if output_format == OutputFormat.JSON:
        parts.append(_JSON_OUTPUT_INSTRUCTIONS)
    else:
        parts.append(_MARKDOWN_OUTPUT_INSTRUCTIONS)

    examples_block = _format_examples(num_examples, example_format)
    if examples_block:
        parts.append(examples_block)

    return "\n\n".join(parts)


# === LLM call layer ===

def _call_anthropic(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    thinking_budget: int,
) -> dict:
    """Llama a Anthropic y normaliza la respuesta a un dict uniforme.

    Cuando thinking_budget > 0, se activa extended thinking. En ese caso,
    temperature no se envía (Claude 4.5+ no permite combinar temperature
    con thinking habilitado).
    """
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY no está configurada",
        )

    client = Anthropic(api_key=settings.anthropic_api_key)

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    if thinking_budget > 0:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # Cuando thinking está activado, temperature debe omitirse en Claude 4.5+
    else:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)

    # Extraer solo los bloques de tipo "text" (descartando "thinking")
    text_blocks = [block.text for block in response.content if block.type == "text"]
    estimation_text = "".join(text_blocks)

    return {
        "text": estimation_text,
        "model": response.model,
        "finish_reason": response.stop_reason or "end_turn",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _call_openai(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    thinking_budget: int,
) -> dict:
    """Llama a OpenAI Chat Completions y normaliza la respuesta.

    `thinking_budget` se ignora silenciosamente porque el modelo por defecto
    del curso (gpt-4o-mini) no es un modelo de razonamiento. El parámetro
    permanece en el request por uniformidad de API.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY no está configurada",
        )

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    choice = response.choices[0]
    return {
        "text": choice.message.content or "",
        "model": response.model,
        "finish_reason": choice.finish_reason or "stop",
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def _call_llm(
    settings: Settings,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    thinking_budget: int,
) -> dict:
    """Dispatch entre proveedores. Devuelve la respuesta normalizada."""
    if settings.llm_provider == "anthropic":
        return _call_anthropic(
            settings, model, system, user, max_tokens, temperature, thinking_budget
        )
    elif settings.llm_provider == "openai":
        return _call_openai(
            settings, model, system, user, max_tokens, temperature, thinking_budget
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"LLM_PROVIDER desconocido: {settings.llm_provider!r}",
        )


# === Two-phase preprocessing ===

def _extract_requirements(settings: Settings, model: str, transcription: str) -> dict:
    """Primera fase del preprocesado two_phase: extrae requisitos limpios."""
    return _call_llm(
        settings=settings,
        model=model,
        system=_REQUIREMENTS_EXTRACTION_SYSTEM,
        user=transcription,
        max_tokens=2000,
        temperature=0.2,
        thinking_budget=0,
    )


# === Public API ===

async def generate_estimation(request: EstimationRequest) -> EstimationResponse:
    """Orquesta el flujo completo: preprocesado → prompt → LLM → evaluación.

    El orden de las fases es deliberado y cada fase es opt-in según el request:

    1. Si preprocessing == two_phase, ejecutar extract_requirements primero.
    2. Construir el system prompt con las opciones del request.
    3. Llamar al LLM con el prompt construido y la transcripción (o los
       requisitos extraídos si two_phase).
    4. Ejecutar la evaluación estructural si request.evaluation == True.
    5. Construir la respuesta con metadatos (latencia, tokens, evaluación).
    """
    settings = get_settings()
    model = request.model or settings.llm_model

    started_at = time.time()

    # === Fase 1: preprocesado opcional ===
    extracted_requirements: str | None = None
    preprocessing_input_tokens = 0
    preprocessing_output_tokens = 0

    if request.preprocessing == PreprocessingType.TWO_PHASE:
        extraction = _extract_requirements(settings, model, request.transcription)
        extracted_requirements = extraction["text"]
        preprocessing_input_tokens = extraction["input_tokens"]
        preprocessing_output_tokens = extraction["output_tokens"]

    # === Fase 2: construcción del system prompt ===
    system_prompt = build_system_prompt(
        num_examples=request.num_examples,
        example_format=request.example_format,
        output_format=request.output_format,
        preprocessing=request.preprocessing,
    )

    # === Fase 3: llamada al LLM ===
    user_input = extracted_requirements or request.transcription

    llm_result = _call_llm(
        settings=settings,
        model=model,
        system=system_prompt,
        user=user_input,
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
        provider=settings.llm_provider,
        finish_reason=llm_result["finish_reason"],
        preprocessing_type=request.preprocessing,
        output_format=request.output_format,
        latency_ms=latency_ms,
        token_usage=token_usage,
        extracted_requirements=extracted_requirements,
        evaluation=evaluation,
    )
