"""Cliente Streamlit (formulario) para el servicio estimator-cag.

A partir de pre-session-04, el cliente es un formulario con parámetros
tipados que produce un POST /api/v1/estimate al backend. Ya no es un
chat conversacional ni consume el endpoint SSE.

Sigue siendo cliente HTTP puro: no importa nada de `app.*`.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# === Configuración ===

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{BACKEND_URL}/api/v1/estimate"
REQUEST_TIMEOUT = float(os.environ.get("STREAMLIT_TIMEOUT", "180"))


# === Mapeos enum → label legible para el formulario ===

PROJECT_TYPE_OPTIONS: dict[str, str] = {
    "mobile_app": "Mobile app",
    "web_saas": "Web SaaS",
    "internal_tool": "Internal tool",
    "data_pipeline": "Data pipeline",
}

DETAIL_LEVEL_OPTIONS: dict[str, str] = {
    "summary": "Summary",
    "medium": "Medium",
    "detailed": "Detailed",
}

OUTPUT_FORMAT_OPTIONS: dict[str, str] = {
    "phases_table": "Phases table",
    "line_items": "Line items",
    "narrative": "Narrative",
}


# === HTTP client ===

def submit_estimation(payload: dict) -> dict:
    """POST al backend y devuelve el JSON parseado."""
    response = httpx.post(
        ESTIMATE_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# === App ===

st.set_page_config(
    page_title="Estimator CAG",
    page_icon="🧮",
    layout="centered",
)

st.title("🧮 Estimator CAG")
st.caption(f"Backend: `{BACKEND_URL}` · Endpoint: `/api/v1/estimate`")

with st.form("estimation_form", clear_on_submit=False):
    description = st.text_area(
        "Project description",
        height=180,
        placeholder=(
            "Describe the project you want to estimate. "
            "Include the main features, target platforms, integrations, "
            "and any constraints you know about (20-2000 chars)."
        ),
        max_chars=2000,
    )

    col1, col2 = st.columns(2)
    with col1:
        project_type = st.selectbox(
            "Project type",
            options=list(PROJECT_TYPE_OPTIONS.keys()),
            format_func=lambda key: PROJECT_TYPE_OPTIONS[key],
        )
    with col2:
        output_format = st.selectbox(
            "Output format",
            options=list(OUTPUT_FORMAT_OPTIONS.keys()),
            format_func=lambda key: OUTPUT_FORMAT_OPTIONS[key],
        )

    detail_level = st.radio(
        "Detail level",
        options=list(DETAIL_LEVEL_OPTIONS.keys()),
        format_func=lambda key: DETAIL_LEVEL_OPTIONS[key],
        horizontal=True,
    )

    submitted = st.form_submit_button("Generate estimation", type="primary")


if submitted:
    cleaned_description = description.strip()
    if len(cleaned_description) < 20:
        st.error("La descripción debe tener al menos 20 caracteres.")
        st.stop()

    payload = {
        "description": cleaned_description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }

    with st.spinner("Generating estimation..."):
        try:
            result = submit_estimation(payload)
        except httpx.HTTPStatusError as exc:
            st.error(
                f"El backend respondió con {exc.response.status_code}: "
                f"{exc.response.text}"
            )
            st.stop()
        except httpx.RequestError as exc:
            st.error(
                f"No se pudo conectar al backend en `{BACKEND_URL}`. "
                f"¿Está levantado el contenedor? Detalle: {exc}"
            )
            st.stop()

    # Persistir la última estimación en session_state para que sobreviva a
    # reruns parciales de Streamlit.
    st.session_state.last_result = result


if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.markdown("### Estimation")
    st.markdown(result["text"])
    st.caption(f"Prompt version: `{result['prompt_version']}`")
