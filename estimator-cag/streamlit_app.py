"""Cliente Streamlit para el servicio de estimaciones.

Cliente HTTP puro: no importa de `app.*`. Llama al backend FastAPI vía httpx y
renderiza el `EstimationResponse` estructurado:
- `result.summary` con `st.markdown` (o `st.warning` si es out-of-scope).
- `result.phases` con `st.dataframe`.
- Métricas con `st.metric` en tres columnas.
- `st.progress` para la confianza global.
- Caption con `prompt_version` y badge si la respuesta vino de cache.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ENDPOINT = f"{BACKEND_URL.rstrip('/')}/api/v1/estimate"
REQUEST_TIMEOUT = float(os.getenv("STREAMLIT_REQUEST_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------


def _render_form() -> dict[str, Any] | None:
    """Pinta el formulario y devuelve el payload si el usuario lo envía."""
    with st.form("estimation_form", clear_on_submit=False):
        description = st.text_area(
            "Project description",
            height=180,
            placeholder="Describe the project to estimate...",
        )
        col_left, col_right = st.columns(2)
        with col_left:
            project_type = st.selectbox(
                "Project type",
                options=[
                    "mobile_app",
                    "web_saas",
                    "internal_tool",
                    "integration",
                    "other",
                ],
                index=4,
            )
        with col_right:
            detail_level = st.selectbox(
                "Detail level",
                options=["summary", "medium", "detailed"],
                index=1,
            )
        output_format = st.radio(
            "Output format",
            options=["phases_table", "line_items", "narrative"],
            index=0,
            horizontal=True,
        )
        submitted = st.form_submit_button("Generate estimation")
        if not submitted:
            return None
        if not description or len(description.strip()) < 10:
            st.warning("La descripción debe tener al menos 10 caracteres.")
            return None
        return {
            "description": description.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }


# ---------------------------------------------------------------------------
# Llamada al backend
# ---------------------------------------------------------------------------


def _call_backend(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Llama al endpoint y devuelve el JSON crudo, manejando errores comunes."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", {})
        except ValueError:
            detail = {}
        if exc.response.status_code == 400 and isinstance(detail, dict) and detail.get(
            "error"
        ) == "input_guardrail":
            st.error(
                f"La descripción fue rechazada por el guardrail de entrada "
                f"(`{detail.get('category')}`): {detail.get('reason')}"
            )
        else:
            st.error(
                f"El backend respondió con {exc.response.status_code}: "
                f"{exc.response.text}"
            )
        return None
    except httpx.RequestError as exc:
        st.error(f"No se pudo contactar con el backend ({BACKEND_URL}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _render_estimation(response: dict[str, Any]) -> None:
    """Renderiza un `EstimationResponse` estructurado."""
    result = response.get("result", {})
    prompt_version = response.get("prompt_version", "unknown")
    cached = response.get("cached", False)
    cache_level = response.get("cache_level")

    summary = result.get("summary", "")
    is_out_of_scope = summary.startswith("Out of scope:")

    if is_out_of_scope:
        st.warning(summary)
        st.caption(
            "El modelo marcó la petición como fuera del alcance del estimador. "
            "No se generaron fases."
        )
    else:
        st.markdown("### 📝 Summary")
        st.markdown(summary)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            "Duration",
            f"{result.get('total_duration_weeks', 0)} weeks",
        )
        col_b.metric(
            "Cost",
            f"{result.get('total_cost_eur', 0):,} EUR",
        )
        col_c.metric(
            "Confidence",
            f"{result.get('confidence_pct', 0)} %",
        )
        st.progress(min(max(result.get("confidence_pct", 0) / 100, 0.0), 1.0))

        phases = result.get("phases", [])
        if phases:
            st.markdown("### 📊 Breakdown by phase")
            st.dataframe(
                [
                    {
                        "Phase": phase.get("name", ""),
                        "Weeks": phase.get("duration_weeks", 0),
                        "Cost (EUR)": phase.get("cost_eur", 0),
                        "Confidence (%)": phase.get("confidence_pct", 0),
                        "Assumptions": " · ".join(phase.get("assumptions", [])),
                    }
                    for phase in phases
                ],
                use_container_width=True,
                hide_index=True,
            )

    caption_parts = [f"Prompt version: `{prompt_version}`"]
    if cached:
        caption_parts.append(f"📦 From cache ({cache_level})")
    st.caption(" · ".join(caption_parts))


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Estimator", page_icon="📐", layout="centered")
    st.title("📐 Software project estimator")
    st.write(
        "Describe a software project and the backend will return a structured "
        "estimation with phases, totals and confidence."
    )

    payload = _render_form()
    if payload is not None:
        with st.spinner("Generating estimation..."):
            response = _call_backend(payload)
        if response is not None:
            st.session_state.last_response = response

    last_response = st.session_state.get("last_response")
    if last_response is not None:
        st.divider()
        _render_estimation(last_response)


if __name__ == "__main__":
    main()
