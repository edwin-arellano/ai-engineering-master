"""Paquete de guardrails de entrada y salida.

Expone:
- `validate_input`, `InputGuardrailError` y `InputGuardrailCategory` para la
  capa 2 (validación semántica del input).
- `should_cache_result`, `is_out_of_scope`, `is_low_confidence` para la capa 5
  (filtro de salida que decide si la respuesta entra al cache).
"""

from app.generation.cag.guardrails.input_guardrails import (
    InputGuardrailCategory,
    InputGuardrailError,
    validate_input,
)
from app.generation.cag.guardrails.output_guardrails import (
    is_low_confidence,
    is_out_of_scope,
    should_cache_result,
)

__all__ = [
    "InputGuardrailCategory",
    "InputGuardrailError",
    "validate_input",
    "is_low_confidence",
    "is_out_of_scope",
    "should_cache_result",
]
