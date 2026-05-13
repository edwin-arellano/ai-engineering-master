"""Servicio de evaluación estructural nivel 1 de las respuestas del LLM.

Este módulo es totalmente determinista: ningún chequeo aquí llama a un LLM.
La evaluación nivel 1 se centra en estructura y consistencia interna del
output (¿están todas las secciones? ¿la suma de filas coincide con el total
declarado? ¿la respuesta no se truncó?). Las evaluaciones nivel 2
(coherencia con los ejemplos) y nivel 3 (calidad semántica) se introducen
en sesiones posteriores.
"""

from __future__ import annotations

import json
import re

from app.schemas.legacy_estimation import LegacyOutputFormat, StructureCheck


# Reasons de éxito por proveedor:
# - OpenAI Chat Completions: "stop"
# - Anthropic Messages: "end_turn"
_SUCCESS_FINISH_REASONS = frozenset({"stop", "end_turn"})

# Tolerancia relativa para considerar que dos totales de horas "cuadran"
_HOURS_MATCH_TOLERANCE = 0.05  # 5%


def evaluate_estimation(
    estimation_text: str,
    output_format: LegacyOutputFormat,
    finish_reason: str,
) -> StructureCheck:
    """Evalúa la estructura del output del LLM y devuelve un StructureCheck.

    Score = promedio simple de 7 booleanos:
    has_title, has_breakdown_table, has_total_sections, has_team_sections,
    has_duration_sections, hours_match, finish_reason_ok. Cada uno aporta 1/7.
    """
    issues: list[str] = []

    if output_format == LegacyOutputFormat.JSON:
        result = _evaluate_json(estimation_text, issues)
    else:
        result = _evaluate_markdown(estimation_text, issues)

    finish_reason_ok = finish_reason in _SUCCESS_FINISH_REASONS
    if not finish_reason_ok:
        issues.append(
            f"Respuesta truncada o interrumpida: finish_reason={finish_reason!r}"
        )

    booleans = [
        result["has_title"],
        result["has_breakdown_table"],
        result["has_total_sections"],
        result["has_team_sections"],
        result["has_duration_sections"],
        result["hours_match"],
        finish_reason_ok,
    ]
    score = sum(1 for b in booleans if b) / 7

    return StructureCheck(
        **result,
        finish_reason_ok=finish_reason_ok,
        score=round(score, 3),
        issues=issues,
    )


# === Markdown evaluation ===

_TITLE_RE = re.compile(r"^#{1,2}\s+\S", re.MULTILINE)
_TABLE_RE = re.compile(r"\|[^\n]+\|\s*\n\|\s*-+\s*\|", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s+.+\(\s*\d+\s*h", re.MULTILINE)
_TOTAL_RE = re.compile(r"\*?\*?total(?:\s+estimado)?\*?\*?\s*[:\s]", re.IGNORECASE)
_TEAM_RE = re.compile(r"\*?\*?equipo(?:\s+recomendado)?\*?\*?\s*[:\s]", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\*?\*?duraci[oó]n(?:\s+estimada)?\*?\*?\s*[:\s]",
    re.IGNORECASE,
)
_DECLARED_TOTAL_HOURS_RE = re.compile(
    r"\*?\*?total(?:\s+estimado)?\*?\*?\s*[:\s]+([\d.,]+)\s*h",
    re.IGNORECASE,
)
_ROW_HOURS_RE = re.compile(r"\(?\s*(\d+)\s*h(?:oras)?\b", re.IGNORECASE)


def _evaluate_markdown(text: str, issues: list[str]) -> dict:
    has_title = bool(_TITLE_RE.search(text))
    if not has_title:
        issues.append("Falta el título (heading h1/h2)")

    has_breakdown_table = bool(_TABLE_RE.search(text) or _NUMBERED_LIST_RE.search(text))
    if not has_breakdown_table:
        issues.append("Falta el desglose de tareas (tabla Markdown o lista numerada con horas)")

    has_total_sections = bool(_TOTAL_RE.search(text))
    if not has_total_sections:
        issues.append("Falta la sección de total")

    has_team_sections = bool(_TEAM_RE.search(text))
    if not has_team_sections:
        issues.append("Falta la sección de equipo recomendado")

    has_duration_sections = bool(_DURATION_RE.search(text))
    if not has_duration_sections:
        issues.append("Falta la sección de duración estimada")

    # Parsear el total declarado
    declared_match = _DECLARED_TOTAL_HOURS_RE.search(text)
    declared_total_hours: int | None = None
    if declared_match:
        raw = declared_match.group(1).replace(".", "").replace(",", ".")
        try:
            declared_total_hours = int(float(raw))
        except ValueError:
            declared_total_hours = None

    # Sumar las horas de las filas (excluyendo la mención del total para no duplicar)
    all_hour_mentions = [int(h) for h in _ROW_HOURS_RE.findall(text)]
    sum_row_hours: int | None = None
    if all_hour_mentions:
        if declared_total_hours is not None and declared_total_hours in all_hour_mentions:
            # Eliminamos UNA aparición del total declarado para no contarlo dos veces
            all_hour_mentions.remove(declared_total_hours)
        sum_row_hours = sum(all_hour_mentions) if all_hour_mentions else None

    hours_match = _hours_match(declared_total_hours, sum_row_hours)
    if declared_total_hours is not None and sum_row_hours is not None and not hours_match:
        issues.append(
            f"Las horas no cuadran: total declarado {declared_total_hours}h "
            f"vs suma de filas {sum_row_hours}h"
        )

    return {
        "has_title": has_title,
        "has_breakdown_table": has_breakdown_table,
        "has_total_sections": has_total_sections,
        "has_team_sections": has_team_sections,
        "has_duration_sections": has_duration_sections,
        "declared_total_hours": declared_total_hours,
        "sum_row_hours": sum_row_hours,
        "hours_match": hours_match,
    }


# === JSON evaluation ===

_JSON_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _evaluate_json(text: str, issues: list[str]) -> dict:
    # Quitar code fences si el modelo las añadió pese a las instrucciones
    cleaned = _JSON_CODE_FENCE_RE.sub("", text).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        issues.append(f"Respuesta no es JSON válido: {e.msg} en pos {e.pos}")
        return {
            "has_title": False,
            "has_breakdown_table": False,
            "has_total_sections": False,
            "has_team_sections": False,
            "has_duration_sections": False,
            "declared_total_hours": None,
            "sum_row_hours": None,
            "hours_match": False,
        }

    if not isinstance(data, dict):
        issues.append("El JSON raíz no es un objeto")
        return {
            "has_title": False,
            "has_breakdown_table": False,
            "has_total_sections": False,
            "has_team_sections": False,
            "has_duration_sections": False,
            "declared_total_hours": None,
            "sum_row_hours": None,
            "hours_match": False,
        }

    title = data.get("title")
    has_title = isinstance(title, str) and bool(title.strip())
    if not has_title:
        issues.append("Falta o está vacío el campo 'title'")

    breakdown = data.get("breakdown")
    has_breakdown_table = (
        isinstance(breakdown, list)
        and len(breakdown) > 0
        and all(
            isinstance(item, dict)
            and isinstance(item.get("task"), str)
            and item["task"].strip()
            and isinstance(item.get("hours"), int)
            for item in breakdown
        )
    )
    if not has_breakdown_table:
        issues.append("Falta o está malformado el array 'breakdown'")

    declared_total_hours_raw = data.get("total_hours")
    has_total_sections = isinstance(declared_total_hours_raw, int)
    if not has_total_sections:
        issues.append("Falta o no es entero el campo 'total_hours'")
    declared_total_hours = declared_total_hours_raw if has_total_sections else None

    team = data.get("team")
    has_team_sections = isinstance(team, str) and bool(team.strip())
    if not has_team_sections:
        issues.append("Falta o está vacío el campo 'team'")

    duration = data.get("duration_weeks") or data.get("duration")
    has_duration_sections = isinstance(duration, str) and bool(duration.strip())
    if not has_duration_sections:
        issues.append("Falta o está vacío el campo 'duration_weeks'")

    sum_row_hours: int | None = None
    if has_breakdown_table:
        sum_row_hours = sum(int(item["hours"]) for item in breakdown)

    hours_match = _hours_match(declared_total_hours, sum_row_hours)
    if has_total_sections and sum_row_hours is not None and not hours_match:
        issues.append(
            f"Las horas no cuadran: total declarado {declared_total_hours}h "
            f"vs suma de filas {sum_row_hours}h"
        )

    return {
        "has_title": has_title,
        "has_breakdown_table": has_breakdown_table,
        "has_total_sections": has_total_sections,
        "has_team_sections": has_team_sections,
        "has_duration_sections": has_duration_sections,
        "declared_total_hours": declared_total_hours,
        "sum_row_hours": sum_row_hours,
        "hours_match": hours_match,
    }


# === Shared helpers ===

def _hours_match(declared: int | None, summed: int | None) -> bool:
    if declared is None or summed is None:
        return False
    if declared == 0:
        return summed == 0
    return abs(declared - summed) / declared < _HOURS_MATCH_TOLERANCE
