"""Cliente Streamlit del servicio conversacional.

Cliente HTTP puro: no importa de ``app.*``. Mantiene ``session_id`` en
``st.session_state``. Cada estimación es un turno dentro de la sesión activa;
el botón "Nueva conversación" del sidebar crea otra sesión y resetea el
estado local.

Renderiza el ``EstimationResult`` estructurado igual que en S04 (st.metric x3,
st.progress, st.dataframe), añade un panel lateral con el ``project_metadata``
optimista que vamos acumulando localmente y muestra el historial de turnos
enviados desde el cliente.

Nota: el backend de pre-S05 no expone ``GET /sessions/{id}``, así que el
panel de metadata se mantiene desde el lado cliente con los datos del último
turno (heurística "lo último que pinté"). El directo de S05 puede enriquecer
``EstimationResponse`` con el metadata consolidado para refrescar el panel.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SESSIONS_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/api/v1/sessions"
REQUEST_TIMEOUT = float(os.getenv("STREAMLIT_REQUEST_TIMEOUT", "180"))

PROJECT_TYPES = ["mobile_app", "web_saas", "internal_tool", "integration", "other"]
DETAIL_LEVELS = ["summary", "medium", "detailed"]
OUTPUT_FORMATS = ["phases_table", "line_items", "narrative"]
ESTIMATION_MODES = ["actor", "actor_critic_boss"]


# ---------------------------------------------------------------------------
# Gestión de sesión
# ---------------------------------------------------------------------------


def _create_session(estimation_mode: str) -> str | None:
    """Crea una sesión nueva en el backend y devuelve su id."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                SESSIONS_ENDPOINT, json={"estimation_mode": estimation_mode}
            )
        response.raise_for_status()
        return response.json()["session_id"]
    except (httpx.HTTPError, KeyError) as exc:
        st.error(f"No se pudo crear la sesión: {exc}")
        return None


def _ensure_session() -> str | None:
    """Garantiza que ``session_state.session_id`` exista, creándolo si hace falta."""
    if "estimation_mode" not in st.session_state:
        st.session_state.estimation_mode = "actor"
    if "session_id" not in st.session_state:
        session_id = _create_session(st.session_state.estimation_mode)
        if session_id is None:
            return None
        st.session_state.session_id = session_id
        st.session_state.turns = []
        st.session_state.project_metadata = {}
    return st.session_state.session_id


def _reset_session() -> None:
    """Limpia el estado local para forzar la creación de una sesión nueva."""
    for key in ("session_id", "turns", "project_metadata", "last_response"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Llamada al backend con multipart
# ---------------------------------------------------------------------------


def _call_estimate(
    session_id: str,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    uploaded_files: list[Any],
) -> dict[str, Any] | None:
    """Envía un turno al endpoint multipart y devuelve el JSON o None si falla."""
    endpoint = f"{SESSIONS_ENDPOINT}/{session_id}/estimate"

    data = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded in uploaded_files or []:
        files.append(
            (
                "attachments",
                (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                ),
            )
        )

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(endpoint, data=data, files=files or None)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", {})
        except ValueError:
            detail = {}
        _handle_error_response(exc.response.status_code, detail)
        return None
    except httpx.RequestError as exc:
        st.error(f"No se pudo contactar con el backend ({BACKEND_URL}): {exc}")
        return None


def _handle_error_response(status_code: int, detail: Any) -> None:
    if not isinstance(detail, dict):
        st.error(f"El backend respondió con {status_code}: {detail}")
        return
    error_type = detail.get("error")
    if status_code == 400 and error_type == "input_guardrail":
        st.error(
            f"La entrada fue rechazada por el guardrail "
            f"(`{detail.get('category')}`): {detail.get('reason')}"
        )
    elif status_code == 404 and error_type == "session_not_found":
        st.warning("La sesión ha caducado. Iniciando una nueva.")
        _reset_session()
    elif status_code == 415:
        st.error(f"Tipo de adjunto no soportado: {detail.get('reason')}")
    elif status_code == 413:
        st.error(f"Adjunto demasiado grande: {detail.get('reason')}")
    else:
        st.error(f"Error {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------


def _render_mode_selector() -> None:
    """Selector de modo de estimación. Cambiarlo resetea la sesión."""
    with st.sidebar:
        st.markdown("### ⚙️ Mode")
        current = st.session_state.get("estimation_mode", "actor")
        chosen_mode = st.radio(
            "Estimation mode",
            options=ESTIMATION_MODES,
            index=ESTIMATION_MODES.index(current),
            help=(
                "El modo se fija al crear la sesión. Cambiarlo arranca una "
                "conversación nueva."
            ),
        )
        if chosen_mode != current:
            st.session_state.estimation_mode = chosen_mode
            _reset_session()
            st.rerun()
        st.divider()


def _render_metadata_panel() -> None:
    metadata = st.session_state.get("project_metadata") or {}
    with st.sidebar:
        st.markdown("### Project metadata")
        if not metadata or all(not v for v in metadata.values()):
            st.caption("Aún no hay hechos consolidados sobre el proyecto.")
        else:
            if metadata.get("project_name"):
                st.markdown(f"**Project:** {metadata['project_name']}")
            if metadata.get("assumed_team_size") is not None:
                st.markdown(f"**Team size:** {metadata['assumed_team_size']}")
            techs = metadata.get("mentioned_technologies") or []
            if techs:
                st.markdown(f"**Technologies:** {', '.join(techs)}")
            if metadata.get("agreed_scope"):
                st.markdown(f"**Scope:** {metadata['agreed_scope']}")

        st.divider()
        if st.button("Nueva conversación", use_container_width=True):
            _reset_session()
            st.rerun()

        session_id = st.session_state.get("session_id")
        if session_id:
            st.caption(f"Session: `{session_id[:8]}…`")


def _render_estimation(response: dict[str, Any]) -> None:
    result = response.get("result", {})
    prompt_version = response.get("prompt_version", "unknown")

    summary = result.get("summary", "")
    is_out_of_scope = summary.startswith("Out of scope:")

    if is_out_of_scope:
        st.warning(summary)
        st.caption(
            "El modelo marcó la petición como fuera del alcance del estimador."
        )
    else:
        st.markdown("### Summary")
        st.markdown(summary)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Duration", f"{result.get('total_duration_weeks', 0)} weeks")
        col_b.metric("Cost", f"{result.get('total_cost_eur', 0):,} EUR")
        col_c.metric("Confidence", f"{result.get('confidence_pct', 0)} %")
        st.progress(min(max(result.get("confidence_pct", 0) / 100, 0.0), 1.0))

        phases = result.get("phases", [])
        if phases:
            st.markdown("### Breakdown by phase")
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

    st.caption(f"Prompt version: `{prompt_version}`")


def _render_acb_panel(response: dict[str, Any]) -> None:
    """Muestra tier y, en modo ACB, las iteraciones con sus issues por severidad."""
    tier = response.get("tier")
    mode = response.get("estimation_mode")
    if tier:
        st.caption(f"Tier: `{tier}` · Mode: `{mode}`")
    if mode != "actor_critic_boss":
        return
    converged = response.get("acb_converged")
    total = response.get("acb_total_iterations")
    badge = "✅ converged" if converged else "⚠️ synthesized (no consensus)"
    st.markdown(f"**Actor-Critic-Boss** — {total} iteration(s) — {badge}")
    for it in response.get("acb_iterations") or []:
        feedback = it.get("critic_feedback", {})
        decision = it.get("boss_decision", {})
        with st.expander(
            f"Iteration {it.get('iteration')} — verdict: "
            f"{feedback.get('verdict')} → boss: {decision.get('decision')}"
        ):
            issues = feedback.get("issues", [])
            if not issues:
                st.caption("Sin issues en esta iteración.")
            for issue in issues:
                st.markdown(
                    f"- **[{issue.get('severity')}]** `{issue.get('field_path')}`: "
                    f"{issue.get('description')}  \n"
                    f"  _Fix_: {issue.get('suggested_fix')}"
                )


def _render_history() -> None:
    turns = st.session_state.get("turns") or []
    if not turns:
        return
    with st.expander(f"History ({len(turns)} turns)", expanded=False):
        for index, turn in enumerate(turns, start=1):
            st.markdown(f"**Turn {index} — user**")
            st.markdown(turn["user"])
            st.markdown(f"**Turn {index} — assistant** (summary)")
            assistant_summary = (
                turn["assistant"].get("result", {}).get("summary", "")
            )
            st.markdown(assistant_summary)
            st.divider()


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------


def _render_form() -> dict[str, Any] | None:
    with st.form("turn_form", clear_on_submit=False):
        transcript = st.text_area(
            "Transcript / project description",
            height=180,
            placeholder="Describe the project or paste a meeting transcript...",
        )
        col_left, col_right = st.columns(2)
        with col_left:
            project_type = st.selectbox(
                "Project type", options=PROJECT_TYPES, index=4
            )
        with col_right:
            detail_level = st.selectbox(
                "Detail level", options=DETAIL_LEVELS, index=1
            )
        output_format = st.radio(
            "Output format", options=OUTPUT_FORMATS, index=0, horizontal=True
        )
        attachments = st.file_uploader(
            "Attachments (PDF or .docx)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Send turn")
        if not submitted:
            return None
        if not transcript or len(transcript.strip()) < 10:
            st.warning("El transcript debe tener al menos 10 caracteres.")
            return None
        return {
            "transcript": transcript.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
            "attachments": attachments or [],
        }


def main() -> None:
    st.set_page_config(page_title="Estimator", page_icon="📐", layout="wide")
    st.title("Software project estimator (conversational)")
    st.write(
        "Conversación iterativa para refinar la estimación de un proyecto de "
        "software. Cada turno actualiza la memoria del proyecto y respeta el "
        "límite de la ventana deslizante del historial."
    )

    _render_mode_selector()

    session_id = _ensure_session()
    if session_id is None:
        return

    _render_metadata_panel()

    payload = _render_form()
    if payload is not None:
        with st.spinner("Generating estimation..."):
            response = _call_estimate(
                session_id=session_id,
                transcript=payload["transcript"],
                project_type=payload["project_type"],
                detail_level=payload["detail_level"],
                output_format=payload["output_format"],
                uploaded_files=payload["attachments"],
            )
        if response is not None:
            st.session_state.last_response = response
            st.session_state.turns.append(
                {"user": payload["transcript"], "assistant": response}
            )

    last_response = st.session_state.get("last_response")
    if last_response is not None:
        st.divider()
        _render_estimation(last_response)
        _render_acb_panel(last_response)
        _render_history()


if __name__ == "__main__":
    main()
