"""Schemas del patrón Actor-Critic-Boss.

El crítico devuelve `CriticFeedback`: un veredicto, una lista de issues
estructurados (categoría, severidad, field_path, descripción, fix sugerido) y
su confianza en la review. Nunca reescribe la estimación.

El boss devuelve `BossDecision`: aceptar, iterar o sintetizar. Su trabajo es
gobernanza del proceso, no calidad técnica. Acumula `BossIteration` por cada
ciclo para trazabilidad y entrega un `ActorCriticBossResult` con el resultado
final más el historial de iteraciones.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.domain.estimation import EstimationResult


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class IssueCategory(StrEnum):
    """Tipos de error que el crítico busca en la estimación del actor."""

    ARITHMETIC_ERROR = "arithmetic_error"
    HALLUCINATION = "hallucination"
    SCOPE_MISMATCH = "scope_mismatch"
    PHASE_IMBALANCE = "phase_imbalance"
    LOST_ASSUMPTIONS = "lost_assumptions"
    UNREALISTIC_ESTIMATE = "unrealistic_estimate"


class IssueSeverity(StrEnum):
    """Severidad de un issue detectado por el crítico."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class CriticVerdict(StrEnum):
    """Veredicto global del crítico sobre la estimación del actor."""

    ACCEPT = "accept"
    NEEDS_ITERATION = "needs_iteration"
    REJECT = "reject"


class CriticIssue(BaseModel):
    """Un problema concreto encontrado por el crítico.

    `field_path` apunta a un campo concreto del EstimationResult para que el
    actor y el boss sepan exactamente qué corregir. Ejemplos: "summary",
    "total_cost_eur", "phases", "phases[2].duration_weeks".
    """

    category: IssueCategory
    severity: IssueSeverity
    field_path: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1, max_length=1000)
    suggested_fix: str = Field(..., min_length=1, max_length=1000)


class CriticFeedback(BaseModel):
    """Feedback estructurado del crítico. Nunca contiene una nueva estimación."""

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list)
    review_confidence_pct: int = Field(..., ge=0, le=100)

    def has_critical(self) -> bool:
        return any(i.severity is IssueSeverity.CRITICAL for i in self.issues)

    def has_blocking(self) -> bool:
        """True si hay issues critical o major (no aceptable sin iterar)."""
        return any(
            i.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)
            for i in self.issues
        )

    @classmethod
    def empty_accept(cls) -> "CriticFeedback":
        """Feedback neutro usado como fallback cuando el crítico rompe."""
        return cls(verdict=CriticVerdict.ACCEPT, issues=[], review_confidence_pct=50)


# ---------------------------------------------------------------------------
# Boss
# ---------------------------------------------------------------------------


class BossDecisionType(StrEnum):
    """Decisión de gobernanza del boss para una iteración."""

    ACCEPT = "accept"
    ITERATE = "iterate"
    SYNTHESIZE = "synthesize"


class BossDecision(BaseModel):
    """Decisión simple del boss para una iteración concreta."""

    decision: BossDecisionType
    reasoning: str = Field(..., min_length=1, max_length=1000)


class BossIteration(BaseModel):
    """Registro de una iteración del loop, para trazabilidad."""

    iteration: int = Field(..., ge=0)
    critic_feedback: CriticFeedback
    boss_decision: BossDecision


class ActorCriticBossResult(BaseModel):
    """Resultado final del flujo ACB + el rastro de iteraciones."""

    final_result: EstimationResult
    iterations: list[BossIteration] = Field(default_factory=list)
    converged: bool
    total_iterations: int = Field(..., ge=1)


# Resolución de la forward reference de `EstimationResponse.acb_iterations`.
# Importa al final para evitar circular: `estimation.py` solo tipa `BossIteration`
# bajo TYPE_CHECKING; aquí ya están todas las clases listas.
from app.domain import estimation as _estimation_module  # noqa: E402

_estimation_module.EstimationResponse.model_rebuild()
