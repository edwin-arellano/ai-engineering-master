"""Cliente conversacional Streamlit para el servicio estimator-cag.

Este archivo es deliberadamente un **cliente HTTP independiente** del backend:
no importa nada de `app.*`. Consume el endpoint SSE `/api/v1/estimate/stream`
del backend FastAPI vía httpx.

Diseño explicado por Antonio en la sesión live de S3:
- Si el frontend se rompe, el API sigue respondiendo a otros clientes.
- Si el día de mañana cambia el frontend (a Next.js, Vue, lo que sea), el
  contrato HTTP del backend no se toca.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import streamlit as st
from dotenv import load_dotenv

# Cargar .env del proyecto (mismas variables que usa el backend)
load_dotenv()


# === Configuración ===

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/api/v1/estimate/stream"
REQUEST_TIMEOUT = float(os.environ.get("STREAMLIT_TIMEOUT", "180"))


# === SSE parsing y consumo del backend ===

def _parse_sse_events(response: httpx.Response) -> Iterator[tuple[str, str]]:
    """Parsea el stream SSE en tuplas (event_type, data).

    httpx no tiene parser SSE integrado. Implementamos el mínimo necesario:
    - `event: <tipo>` define el tipo del próximo evento.
    - `data: <contenido>` añade contenido al evento actual (puede haber varias
      líneas data; se concatenan con '\\n').
    - Línea vacía = fin del evento, se emite.
    - Líneas que empiezan por `:` son comentarios/heartbeats, se ignoran.
    """
    current_event = "message"
    data_lines: list[str] = []

    for line in response.iter_lines():
        if line.startswith(":"):
            continue
        if not line:
            if data_lines:
                yield current_event, "\n".join(data_lines)
                data_lines = []
            current_event = "message"
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    # Por si el stream termina sin línea vacía final
    if data_lines:
        yield current_event, "\n".join(data_lines)


def stream_estimation(transcription: str) -> Iterator[str]:
    """Consume el endpoint SSE y yield text chunks listos para `st.write_stream`.

    Los eventos `event: delta` aportan texto. `event: done` cierra el stream.
    `event: error` levanta RuntimeError con el mensaje del backend.
    """
    payload = {"transcription": transcription}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for event_type, data in _parse_sse_events(response):
            if event_type == "done":
                return
            if event_type == "error":
                raise RuntimeError(f"Backend devolvió error: {data}")
            if event_type in ("delta", "message") and data:
                yield data


# === App Streamlit ===

st.set_page_config(
    page_title="Estimator CAG",
    page_icon="🧮",
    layout="centered",
)

st.title("🧮 Estimator CAG")
st.caption(f"Backend: `{BACKEND_URL}` · Endpoint: `/api/v1/estimate/stream`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(stream_estimation(prompt))
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
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})
