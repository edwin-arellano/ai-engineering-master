"""Juez LLM: para cada línea de la estimación, decide si la evidencia citada soporta
GENUINAMENTE el effort/scope. Semántico (lo que la verificación estructural NO cubre).
Alias barato (reformulator); batched en una sola llamada. El cálculo lo hace el ancla
numérica determinista; el juez solo aporta el veredicto semántico."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from app.foundations.config import Settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_judge_prompt


class LineVerdict(BaseModel):
    index: int
    supported: bool
    reason: str = Field(..., min_length=1)


class JudgeVerdicts(BaseModel):
    verdicts: list[LineVerdict] = Field(default_factory=list)


async def judge_lines(
    *, lines: list[dict], wrapper: LLMWrapper, settings: Settings
) -> dict[int, LineVerdict]:
    """`lines` = [{index, title, engineer_days, evidence}]. Devuelve {index: verdict}."""
    if not lines:
        return {}
    payload = "\n".join(
        f"[{ln['index']}] tarea='{ln['title']}' dias={ln['engineer_days']} evidencia='{ln['evidence']}'"
        for ln in lines
    )
    result = await asyncio.to_thread(
        wrapper.complete_structured,
        system_prompt=render_judge_prompt(settings.judge_prompt_version),
        user_message=payload,
        response_model=JudgeVerdicts,
        alias=REFORMULATOR_ALIAS,
        temperature=0.0,
        max_tokens=1200,
    )
    return {v.index: v for v in result.verdicts}
