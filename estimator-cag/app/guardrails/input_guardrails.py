"""Capa 2 de guardrails: validación semántica del input.

Aplica tres chequeos en orden creciente de coste:

1. Patrones de prompt injection (regex en local, ~microsegundos).
2. Detección de PII (regex sobre email, teléfono e IBAN).
3. OpenAI Moderation API (~50-100 ms, opcional según `OPENAI_API_KEY`).

Política de fallo: **exception**. Cualquier match lanza `InputGuardrailError`
con la categoría correspondiente. El router HTTP traduce a `HTTPException 400`
con un detalle estructurado para que el cliente pueda mostrar un mensaje
específico al usuario.

Moderation es best-effort: si la API falla por red o por falta de credencial,
se loguea un warning y se continúa con las heurísticas regex. No bloquear la
operativa por una dependencia opcional caída.
"""

from __future__ import annotations

import re
from enum import StrEnum

import structlog

from app.config import Settings

logger = structlog.get_logger(__name__)


class InputGuardrailCategory(StrEnum):
    """Etiquetas de la razón por la que el guardrail disparó."""

    PROMPT_INJECTION = "prompt_injection"
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_IBAN = "pii_iban"
    MODERATION = "moderation"


class InputGuardrailError(Exception):
    """Error que indica que la entrada no debe propagarse al modelo."""

    def __init__(self, reason: str, category: InputGuardrailCategory) -> None:
        super().__init__(reason)
        self.reason = reason
        self.category = category


# ---------------------------------------------------------------------------
# Patrones
# ---------------------------------------------------------------------------

# Frases típicas de prompt injection. La lista es deliberadamente conservadora;
# falsos positivos sobre texto técnico legítimo son aceptables a cambio de
# rechazar de raíz los intentos triviales.
PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore (?:all )?previous (?:instructions|context)",
    r"disregard (?:all )?(?:prior|previous) (?:instructions|context)",
    r"you are now",
    r"new instructions\s*:",
    r"system prompt\s*:",
    r"</project_description>",
    r"</?scope>",
    r"</?output_format>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
)

# Email estándar.
PII_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Teléfonos: al menos 9 dígitos para filtrar falsos positivos como "5 weeks"
# o "8 sprints". Acepta espacios, guiones, paréntesis y prefijo internacional.
PII_PHONE_PATTERN = re.compile(
    r"(?:\+?\d[\d\s\-().]{8,}\d)"
)

# IBAN europeo (2 letras de país, 2 dígitos de control, hasta 30 alfanuméricos).
PII_IBAN_PATTERN = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"
)

_PROMPT_INJECTION_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in PROMPT_INJECTION_PATTERNS),
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Funciones internas
# ---------------------------------------------------------------------------


def _check_prompt_injection(description: str) -> None:
    match = _PROMPT_INJECTION_REGEX.search(description)
    if match:
        raise InputGuardrailError(
            f"Patrón de prompt injection detectado: {match.group(0)!r}",
            InputGuardrailCategory.PROMPT_INJECTION,
        )


def _check_pii(description: str) -> None:
    if PII_EMAIL_PATTERN.search(description):
        raise InputGuardrailError(
            "Se detectó un email en la descripción; elimínelo antes de reenviar.",
            InputGuardrailCategory.PII_EMAIL,
        )
    # IBAN antes que teléfono: el formato IBAN (p.ej. ES91 2100 0418 ...) contiene
    # muchos dígitos y dispararía el regex de teléfono si se chequeara primero.
    if PII_IBAN_PATTERN.search(description):
        raise InputGuardrailError(
            "Se detectó un IBAN en la descripción.",
            InputGuardrailCategory.PII_IBAN,
        )
    # Buscamos teléfonos solo si la descripción tiene al menos un dígito; ahorra
    # el regex completo en texto puramente alfabético.
    if any(ch.isdigit() for ch in description):
        match = PII_PHONE_PATTERN.search(description)
        if match:
            digits = sum(ch.isdigit() for ch in match.group(0))
            if digits >= 9:
                raise InputGuardrailError(
                    "Se detectó un número de teléfono en la descripción.",
                    InputGuardrailCategory.PII_PHONE,
                )


def _check_moderation(description: str, settings: Settings) -> None:
    """Llama a la Moderation API de OpenAI si está disponible."""
    if not settings.moderation_enabled or not settings.openai_api_key:
        logger.debug("moderation_skipped", reason="disabled_or_missing_key")
        return
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.moderations.create(input=description)
        result = response.results[0]
        if result.flagged:
            categories = [
                name for name, flag in result.categories.model_dump().items() if flag
            ]
            raise InputGuardrailError(
                f"Moderation API marcó la entrada: {categories}",
                InputGuardrailCategory.MODERATION,
            )
    except InputGuardrailError:
        raise
    except Exception as exc:  # noqa: BLE001
        # No bloqueamos el servicio si Moderation falla por red o credencial.
        logger.warning("moderation_call_failed", error=str(exc))


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def validate_input(description: str, settings: Settings) -> None:
    """Valida la descripción del usuario; lanza `InputGuardrailError` si falla.

    Aplica las tres capas en orden de coste creciente. La función no devuelve
    nada: si retorna, la descripción es válida y se puede pasar al LLM.
    """
    _check_prompt_injection(description)
    _check_pii(description)
    _check_moderation(description, settings)
