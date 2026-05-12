"""Cliente conversacional Streamlit para el servicio estimator-cag.

Reutiliza el system prompt y los ejemplos CAG del servicio FastAPI
(misma fuente de verdad) pero hace sus propias llamadas en streaming
directamente al SDK del proveedor configurado en LLM_PROVIDER.

En esta rama (`pre-session-03`) el Streamlit NO consume el endpoint
FastAPI: hace sus propias llamadas al SDK porque las funciones del
backend (`generate_estimation`) son no-streaming. La integración con
el endpoint vía SSE entra en `session-03` (live).
"""

from __future__ import annotations

from collections.abc import Iterator

import streamlit as st
from anthropic import Anthropic
from openai import OpenAI

from app.config import Settings, get_settings
from app.schemas.estimation import ExampleFormat, OutputFormat, PreprocessingType
from app.services.llm_service import build_system_prompt


# === Streaming helpers ===

def _stream_anthropic(
    settings: Settings,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Genera la respuesta token a token desde Anthropic.

    Usa el context manager `client.messages.stream(...)` y expone
    el `text_stream`, que `st.write_stream` puede consumir directamente.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY no está configurada en .env"
        )

    client = Anthropic(api_key=settings.anthropic_api_key)
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    ) as stream:
        yield from stream.text_stream


def _stream_openai(
    settings: Settings,
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Genera la respuesta token a token desde OpenAI Chat Completions.

    El system prompt se prepende al array de mensajes (a diferencia de
    Anthropic, donde el system va como parámetro separado).
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada en .env"
        )

    client = OpenAI(api_key=settings.openai_api_key)
    full_messages = [{"role": "system", "content": system}] + messages

    stream = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=full_messages,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def _build_initial_system_prompt(settings: Settings) -> str:
    """Construye el system prompt una sola vez por sesión.

    Usa los defaults del servidor para num_examples, preprocessing y
    output_format. La selección aleatoria de ejemplos en
    `build_system_prompt` se fija al cachear el resultado en
    `st.session_state`, por lo que el chat mantiene un prompt estable
    a lo largo de la conversación.
    """
    return build_system_prompt(
        num_examples=settings.default_num_examples,
        example_format=ExampleFormat.MARKDOWN,
        output_format=OutputFormat(settings.default_output_format),
        preprocessing=PreprocessingType(settings.default_preprocessing),
    )


# === App ===

settings = get_settings()

st.set_page_config(
    page_title="Estimator CAG",
    page_icon="🧮",
    layout="centered",
)

st.title("🧮 Estimator CAG")
st.caption(
    f"Provider: **{settings.llm_provider}** · "
    f"Model: **{settings.llm_model}** · "
    f"Examples: **{settings.default_num_examples}**"
)

# Inicializar el system prompt una vez por sesión (estabiliza la selección
# aleatoria de ejemplos para esa conversación).
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = _build_initial_system_prompt(settings)

# Inicializar el historial de la conversación.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Renderizar el historial existente.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Aceptar input del usuario.
if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Streaming según el proveedor configurado en .env
        try:
            if settings.llm_provider == "anthropic":
                stream = _stream_anthropic(
                    settings=settings,
                    system=st.session_state.system_prompt,
                    messages=st.session_state.messages,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                )
            elif settings.llm_provider == "openai":
                stream = _stream_openai(
                    settings=settings,
                    system=st.session_state.system_prompt,
                    messages=st.session_state.messages,
                    max_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                )
            else:
                st.error(
                    f"LLM_PROVIDER desconocido: {settings.llm_provider!r}. "
                    f"Valores soportados: 'anthropic' | 'openai'."
                )
                st.stop()

            response = st.write_stream(stream)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error llamando al LLM: {exc}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})
