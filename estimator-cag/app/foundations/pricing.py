"""Tabla de precios y cálculo de coste por llamada.

⚠️ VERIFICAR los precios contra las tarifas vigentes del proveedor antes de
confiar en las cifras absolutas del REPORT. Para el ejercicio lo que importa
es la FORMA de la curva de coste (crecimiento relativo turno a turno), que es
correcta aunque el precio absoluto esté ligeramente desfasado. Ajusta estos
valores con la tabla oficial cuando la tengas a mano.

Precios en USD por 1.000.000 de tokens (input / output).
"""

from __future__ import annotations

# Clave: el `model` tal como lo reporta LiteLLM (sin el prefijo de provider si
# LiteLLM lo normaliza; incluye variantes con y sin prefijo por robustez).
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_usd_per_1m, output_usd_per_1m)
    "claude-haiku-4-5-20251001": (1.00, 5.00),  # ⚠️ verificar
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),  # ⚠️ verificar
    "gpt-4o-mini": (0.15, 0.60),  # ⚠️ verificar
    "openai/gpt-4o-mini": (0.15, 0.60),  # ⚠️ verificar
    # LiteLLM reporta el fallback de OpenAI con sufijo de fecha; sin esta entrada
    # el coste de esas llamadas salía 0 (pricing_model_unknown).
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),  # ⚠️ verificar
}

# Fallback conservador cuando el modelo no está en la tabla: evita romper la
# corrida; deja un coste 0 y se loguea para que se note en el REPORT.
_UNKNOWN_COST: tuple[float, float] = (0.0, 0.0)


def cost_for(model: str, tokens_in: int, tokens_out: int) -> float:
    """Devuelve el coste en USD de una llamada dado el modelo y los tokens."""
    in_per_1m, out_per_1m = MODEL_COSTS.get(model, _UNKNOWN_COST)
    return (tokens_in / 1_000_000) * in_per_1m + (tokens_out / 1_000_000) * out_per_1m


def is_known_model(model: str) -> bool:
    """Indica si el modelo tiene una tarifa explícita en ``MODEL_COSTS``."""
    return model in MODEL_COSTS
