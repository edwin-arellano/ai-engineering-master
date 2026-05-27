"""Resolución heurística del tier de usuario.

El resolver evalúa una lista ordenada de reglas. Cada regla inspecciona el
contexto (transcript, project_metadata) y devuelve un tier o None. La primera
regla que devuelve un tier gana. Si ninguna aplica, se usa el tier por defecto.

Cada regla se ejecuta dentro de un try/except: una regla que rompe (por
ejemplo, una futura regla que llame al LLM y falle) no tumba el resolver, solo
se loguea y se pasa a la siguiente. Es la disciplina de "flujos deterministas
con piezas tolerantes a fallos" que Antonio defiende en el directo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import structlog

from app.config import Settings, get_settings
from app.schemas.session import ProjectMetadata
from app.schemas.tier import UserTier

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TierContext:
    """Información disponible para las reglas de resolución de tier."""

    transcript: str
    project_metadata: ProjectMetadata


# Una regla recibe el contexto y devuelve (tier, rule_name) o None.
TierRule = Callable[[TierContext], tuple[UserTier, str] | None]


# ---------------------------------------------------------------------------
# Reglas heurísticas
# ---------------------------------------------------------------------------

_EXECUTIVE_PATTERNS = re.compile(
    r"\b(board|investor|c-level|cto|ceo|cfo|go/no-go|business case|"
    r"comit[ée]|presupuesto ejecutivo|propuesta comercial)\b",
    flags=re.IGNORECASE,
)
_PM_PATTERNS = re.compile(
    r"\b(roadmap|milestone|hito|sprint|gantt|cronograma|fase|"
    r"project plan|planificaci[óo]n|entregable)\b",
    flags=re.IGNORECASE,
)
_DEVELOPER_PATTERNS = re.compile(
    r"\b(api|endpoint|database|backend|frontend|deploy|stack|"
    r"refactor|arquitectura t[ée]cnica|integraci[óo]n|infra)\b",
    flags=re.IGNORECASE,
)


def _rule_executive(ctx: TierContext) -> tuple[UserTier, str] | None:
    if _EXECUTIVE_PATTERNS.search(ctx.transcript):
        return UserTier.EXECUTIVE, "executive_keywords"
    return None


def _rule_pm(ctx: TierContext) -> tuple[UserTier, str] | None:
    if _PM_PATTERNS.search(ctx.transcript):
        return UserTier.PM, "pm_keywords"
    return None


def _rule_developer(ctx: TierContext) -> tuple[UserTier, str] | None:
    if _DEVELOPER_PATTERNS.search(ctx.transcript):
        return UserTier.DEVELOPER, "developer_keywords"
    return None


# Orden importa: lo más específico/escaso primero (executive), lo más común
# al final (developer). Si nada aplica, cae al default.
DEFAULT_RULES: tuple[TierRule, ...] = (
    _rule_executive,
    _rule_pm,
    _rule_developer,
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TierResolver:
    """Evalúa reglas en orden con fallback graceful."""

    def __init__(
        self,
        rules: tuple[TierRule, ...] = DEFAULT_RULES,
        default_tier: UserTier = UserTier.DEVELOPER,
    ) -> None:
        self.rules = rules
        self.default_tier = default_tier

    def resolve(self, ctx: TierContext) -> UserTier:
        """Devuelve el primer tier que una regla resuelva, o el default."""
        for rule in self.rules:
            try:
                outcome = rule(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "tier_rule_failed",
                    rule=getattr(rule, "__name__", "anonymous"),
                    error=str(exc),
                )
                continue
            if outcome is not None:
                tier, rule_name = outcome
                logger.info("tier_resolved", tier=tier.value, rule=rule_name)
                return tier
        logger.info("tier_resolved", tier=self.default_tier.value, rule="default")
        return self.default_tier


def get_tier_resolver(settings: Settings | None = None) -> TierResolver:
    """Construye el resolver con el tier por defecto de la config."""
    s = settings or get_settings()
    return TierResolver(default_tier=UserTier(s.default_tier))
