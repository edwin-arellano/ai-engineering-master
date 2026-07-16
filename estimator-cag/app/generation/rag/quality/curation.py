"""Curación del corpus: decide si un documento/estimación DEBE entrar en la BD vectorial.
Garbage-in-garbage-out: los casos límite o excepciones de cliente (aunque correctos)
falsean los vectores. RAGAS NO detecta esta degradación (mide contra el golden set, que
sigue limpio mientras producción se degrada en silencio).

Frontera: aquí solo el gate del lado del SERVICIO IA (flag + señales de excepción). El
puente con la BD de negocio (Rails) y el multitenant son de la capa de negocio."""

from __future__ import annotations

from pydantic import BaseModel


class IndexabilityVerdict(BaseModel):
    indexable: bool
    reasons: list[str]


def is_indexable(*, metadata: dict) -> IndexabilityVerdict:
    """Gate determinista: respeta un flag explícito `indexable` (si viene marcado a False,
    no entra), y rechaza señales de excepción (p.ej. `is_exception`/`client_specific`)."""
    reasons: list[str] = []
    if metadata.get("indexable") is False:
        reasons.append("marcado explícitamente como no indexable")
    if metadata.get("is_exception") or metadata.get("client_specific"):
        reasons.append("caso límite / excepción de cliente (falsea vectores)")
    return IndexabilityVerdict(indexable=not reasons, reasons=reasons)
