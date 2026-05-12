"""Servicio de generación de estimaciones vía LLM con arquitectura CAG."""

from anthropic import Anthropic
from fastapi import HTTPException
from openai import OpenAI

from app.config import Settings, get_settings
from app.context.examples import ESTIMATION_EXAMPLES

# === System prompt construction ===

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior software estimation consultant with 15+ years of \
experience estimating custom development projects. Your job is to \
analyze a meeting transcription with a client and produce a \
structured, defensible estimation.

Use the historical estimations below as reference for scale, format, \
breakdown granularity and confidence level. Do not invent technologies \
or features that are not mentioned in the transcription.

Output format requirements:
- Respond in Markdown.
- Include a numbered task breakdown with hours per task.
- Provide a clear total in hours, a recommended team composition, and \
  an estimated duration in weeks.
- If the transcription is too vague or out of scope (not a software \
  project), say so explicitly and do not invent numbers.

===== REFERENCE ESTIMATIONS =====

{examples_block}

===== END OF REFERENCE ESTIMATIONS =====

Now estimate the project described by the user, applying the same \
format and rigor."""


def _format_examples(examples: list[dict[str, str]]) -> str:
    """Formatea los ejemplos CAG con separadores claros entre ellos."""
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"--- Reference estimation {index} ---\n"
            f"Meeting summary: {example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}"
        )
    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """Construye el system prompt completo con los ejemplos CAG inyectados."""
    examples_block = _format_examples(ESTIMATION_EXAMPLES)
    return _SYSTEM_PROMPT_TEMPLATE.format(examples_block=examples_block)


# === Provider dispatch ===


async def generate_estimation(transcription: str) -> dict[str, str]:
    """Llama al LLM configurado y devuelve la estimación normalizada."""
    settings = get_settings()
    system_prompt = build_system_prompt()

    if settings.llm_provider == "anthropic":
        content = _call_anthropic(settings, system_prompt, transcription)
    elif settings.llm_provider == "openai":
        content = _call_openai(settings, system_prompt, transcription)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown LLM_PROVIDER: {settings.llm_provider!r}",
        )

    return {
        "estimation": content,
        "model": settings.llm_model,
        "provider": settings.llm_provider,
    }


def _call_anthropic(settings: Settings, system: str, user: str) -> str:
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500, detail="ANTHROPIC_API_KEY is not configured"
        )
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _call_openai(settings: Settings, system: str, user: str) -> str:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY is not configured"
        )
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""
