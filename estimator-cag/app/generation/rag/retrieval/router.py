"""Router multi-índice en cascada: decide en qué colección(es) buscar una consulta,
por coste creciente. Devuelve RoutingDecision + el nivel que decidió (trazabilidad).

Cascada (barato → caro):
- explicit: el llamante YA sabe el destino (p.ej. el flujo por-tarea siempre apunta a
  BUDGETS) → se salta el clasificador.
- deterministic: patrones de vocabulario inequívocos → colección (coste cero).
- llm: clasificador con salida estructurada (alias barato) para los casos ambiguos.
- fallback: si el LLM falla, busca en TODAS las colecciones (degradación elegante).

Patrón embrión de delegación agéntica (la clasificación acotada que precede a un router
de herramientas). El tratamiento agéntico completo se difiere al módulo de agentes.
"""

from __future__ import annotations

import re

import structlog

from app.foundations.config import Settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS, LLMWrapper
from app.foundations.prompts.loader import render_routing_prompt
from app.generation.rag.schemas import RoutingDecision, SearchTarget

logger = structlog.get_logger(__name__)

# Nivel 1 — reglas deterministas: patrones de vocabulario inequívocos → colección.
_DETERMINISTIC_RULES: list[tuple[re.Pattern, SearchTarget]] = [
    (
        re.compile(
            r"\b(cu[aá]nto cost|presupuest|estimaci[oó]n|horas|esfuerzo|partida)", re.I
        ),
        SearchTarget.BUDGETS,
    ),
    (
        re.compile(
            r"\b(dijo el cliente|en la reuni[oó]n|se acord[oó]|pidi[oó] el cliente|transcripci)",
            re.I,
        ),
        SearchTarget.TRANSCRIPTS,
    ),
    (
        re.compile(
            r"\b(oauth|pkce|protocolo|arquitectura t[eé]cnica|integraci[oó]n t[eé]cnica|est[aá]ndar)",
            re.I,
        ),
        SearchTarget.TECHNICAL_DOCS,
    ),
]


class QueryRouter:
    def __init__(self, wrapper: LLMWrapper, settings: Settings) -> None:
        self._wrapper = wrapper
        self._settings = settings

    def route(
        self, query: str, *, explicit: list[SearchTarget] | None = None
    ) -> tuple[RoutingDecision, str]:
        """Resuelve los targets por cascada. Devuelve (RoutingDecision, nivel)."""
        # Nivel 0 — explícito: el llamante nombra la colección.
        if explicit:
            return (
                RoutingDecision(targets=explicit, reason="explicit target from caller"),
                "explicit",
            )
        # Nivel 1 — reglas deterministas (coste cero).
        hits = [target for pattern, target in _DETERMINISTIC_RULES if pattern.search(query)]
        if hits:
            uniq = list(dict.fromkeys(hits))[:3]
            logger.info("router.deterministic", targets=[t.value for t in uniq])
            return (
                RoutingDecision(targets=uniq, reason="deterministic vocabulary match"),
                "deterministic",
            )
        # Nivel 2 — clasificador LLM (salida estructurada, alias barato).
        try:
            decision = self._wrapper.complete_structured(
                system_prompt=render_routing_prompt(self._settings.routing_prompt_version),
                user_message=query,
                response_model=RoutingDecision,
                alias=REFORMULATOR_ALIAS,
                temperature=0.0,
                max_tokens=200,
            )
            logger.info(
                "router.llm",
                targets=[t.value for t in decision.targets],
                reason=decision.reason,
            )
            return decision, "llm"
        except Exception:  # noqa: BLE001 — degradación elegante: si el clasificador falla, busca en todo
            logger.warning("router.llm_failed_fallback_all")
        # Nivel 3 — fallback honesto: todas las colecciones.
        return (
            RoutingDecision(
                targets=list(SearchTarget), reason="fallback: search all collections"
            ),
            "fallback",
        )
