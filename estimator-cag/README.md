# Estimator CAG

Servicio FastAPI que recibe la transcripción de una reunión con un
cliente y devuelve una estimación de software generada por un LLM,
usando arquitectura **CAG** (Cache Augmented Generation): el contexto
de ejemplos previos se inyecta directamente en el system prompt en cada
llamada — sin base de datos ni retrieval semántico.

En la rama `pre-session-04` el servicio da el salto de "chat con
textarea libre" a **producto con formulario tipado y prompts
versionados como código**:

- El endpoint `POST /api/v1/estimate` cambia drásticamente: deja de
  aceptar `transcription` con preprocessing/evaluation/thinking_budget
  y pasa a recibir un cuerpo `{description, project_type, detail_level,
  output_format}` producido por un **formulario** en el cliente.
- Los prompts salen del código a **templates Jinja2 versionados** bajo
  `app/prompts/estimation/v1/` (system, user, examples). El `loader.py`
  es el único punto que toca los templates.
- El cliente Streamlit pasa de chat conversacional a **formulario con
  `st.form`**, captura los parámetros tipados y hace POST al backend.
- El endpoint `POST /api/v1/estimate/stream` (SSE) de session-03 **se
  mantiene intacto** con su schema legacy y su flujo de streaming —
  pendiente de migración o eliminación en session-04.

Las cinco capas de session-03 (wrapper LiteLLM con fallback, cache
exact-match Redis, SSE, structlog con request_id, Streamlit como
cliente HTTP puro) siguen exactamente igual.

---

## Requisitos

- **Docker** y **Docker Compose** (recomendado).
- Alternativa local: Python 3.11+ y `uv`
  ([instalación](https://docs.astral.sh/uv/getting-started/installation/)).
- Una API key activa en al menos uno de los dos proveedores: Anthropic u
  OpenAI.
- `jq` para los scripts de demo (`brew install jq` en macOS).

---

## Setup con Docker (recomendado)

```bash
cd estimator-cag
cp .env.example .env
# Edita .env y rellena ANTHROPIC_API_KEY (o OPENAI_API_KEY)

docker compose up --build
```

El servicio queda disponible en:

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

`docker-compose.yml` monta `./app` y `./fixtures` como volúmenes
read-only y arranca `uvicorn --reload`, así que cualquier cambio en el
código se recarga sin reconstruir la imagen.

Para parar:

```bash
docker compose down
```

---

## Setup local sin Docker

```bash
cd estimator-cag
cp .env.example .env
# Edita .env

uv sync
uv run uvicorn app.main:app --reload
```

---

## Interfaz de producto (Streamlit con formulario)

`streamlit_app.py` es un **cliente HTTP puro**: no importa nada de
`app.*`. A partir de pre-session-04 ya no es un chat conversacional:
es un **formulario** con `st.form` que captura parámetros tipados y
hace `POST /api/v1/estimate` al backend con el nuevo schema. La
respuesta llega de golpe tras un `st.spinner` — sin streaming token
a token.

### Arrancar el sistema completo

Necesitas dos procesos: el backend (con Redis) en Docker, y el
Streamlit local con hot-reload.

```bash
# Terminal 1: backend FastAPI + Redis
cd estimator-cag
docker compose up --build

# Terminal 2: Streamlit
cd estimator-cag
uv run streamlit run streamlit_app.py
```

- Backend: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Streamlit: http://localhost:8501

### Campos del formulario

| Campo | Widget | Mapeo |
|---|---|---|
| Project description | `st.text_area` (20–2000 chars) | `description` |
| Project type | `st.selectbox` | `mobile_app` / `web_saas` / `internal_tool` / `data_pipeline` |
| Output format | `st.selectbox` | `phases_table` / `line_items` / `narrative` |
| Detail level | `st.radio` horizontal | `summary` / `medium` / `detailed` |

Al enviar el formulario se hace `POST /api/v1/estimate`. La última
estimación se persiste en `st.session_state.last_result` para que
sobreviva a reruns parciales de Streamlit (cambiar un selectbox tras
recibir una respuesta no la borra de la pantalla).

### Qué cambia respecto a `session-03`

| Aspecto | session-03 | pre-session-04 |
|---|---|---|
| UX del cliente | Chat conversacional | Formulario tipado |
| Endpoint consumido | `/api/v1/estimate/stream` (SSE) | `/api/v1/estimate` (POST + JSON) |
| Streaming | Sí, token a token | No, respuesta de golpe con spinner |
| Construcción del prompt | `build_legacy_system_prompt` con f-strings | `render_estimation_prompt` con Jinja2 versionado |

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `PRIMARY_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Modelo primario en formato `<provider>/<model>` (LiteLLM) |
| `FALLBACK_MODEL` | `openai/gpt-4o-mini` | Modelo al que LiteLLM Router rota si el primario falla |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout por llamada al LLM |
| `LLM_NUM_RETRIES` | `2` | Reintentos antes de activar fallback |
| `LLM_TEMPERATURE` | `0.3` | Temperatura de muestreo (default) |
| `LLM_MAX_TOKENS` | `4000` | Máximo de tokens en la respuesta (default) |
| `ANTHROPIC_API_KEY` | *(vacío)* | API key de Anthropic. Al menos UNA de las dos debe estar configurada o la app falla al arrancar. |
| `OPENAI_API_KEY` | *(vacío)* | API key de OpenAI |
| `REDIS_URL` | `redis://localhost:6379/0` | URL del Redis. En docker-compose se sobrescribe a `redis://redis:6379/0`. |
| `CACHE_TTL_SECONDS` | `86400` | TTL de las entradas de cache (24h) |
| `CACHE_ENABLED` | `true` | Si `false`, la cache se desactiva globalmente |
| `DEFAULT_NUM_EXAMPLES` | `3` | Número de ejemplos a inyectar (informativo; el request manda) |
| `DEFAULT_PREPROCESSING` | `none` | Estrategia de preprocesado por defecto |
| `DEFAULT_OUTPUT_FORMAT` | `markdown` | Formato de salida por defecto |
| `BACKEND_URL` | `http://localhost:8000` | URL del backend desde el Streamlit |
| `ENVIRONMENT` | `development` | `development` → ConsoleRenderer; `production` → JSONRenderer |
| `LOG_LEVEL` | `INFO` | Nivel mínimo de los logs estructurados |

> Las variables `LLM_PROVIDER` y `LLM_MODEL` de session-02 quedaron
> obsoletas: ahora un solo identificador con prefijo (`PRIMARY_MODEL`)
> sustituye al par proveedor + modelo y se lo damos directamente a
> LiteLLM Router.

---

## Request y respuesta del endpoint principal

`POST /api/v1/estimate` acepta un cuerpo JSON con cuatro campos, todos
obligatorios:

| Campo | Tipo | Descripción |
|---|---|---|
| `description` | string (20–2000) | Descripción libre del proyecto. |
| `project_type` | enum | `mobile_app` / `web_saas` / `internal_tool` / `data_pipeline` |
| `detail_level` | enum | `summary` / `medium` / `detailed` |
| `output_format` | enum | `phases_table` / `line_items` / `narrative` |

La respuesta es minimal:

```json
{
  "text": "| phase | duration_weeks | cost_eur | ...",
  "prompt_version": "v1"
}
```

`prompt_version` permite saber qué subdirectorio bajo
`app/prompts/estimation/` produjo la respuesta. Útil para A/B testing y
para depurar regresiones cuando se introduce un `v2/`.

> **Nota**: features de session-02 (preprocessing, evaluation,
> thinking_budget, token_usage, cache_hit, output_format=json) **se
> eliminaron del endpoint /estimate**. Reaparecerán en session-04 con un
> diseño nuevo basado en structured outputs y guardrails. Mientras
> tanto, `evaluation_service.py` y `context/examples.py` quedan en
> standby, alimentando solo el endpoint legacy `/estimate/stream`.

Ejemplo de uso con curl:

```bash
./examples/example-form-request.sh
```

---

## Estructura de prompts (Jinja2)

Los prompts viven en `app/prompts/`:

```
app/prompts/
├── loader.py                       ← único punto que toca los templates
└── estimation/
    └── v1/
        ├── system.j2               ← rol del modelo + reglas de output
        ├── user.j2                 ← envoltorio mínimo de la description
        └── examples.j2             ← 3 few-shot examples
```

Diseño del `Environment` de Jinja2 en `loader.py`:

- **`StrictUndefined`**: si un template referencia una variable no
  provista, el render rompe con `UndefinedError` en lugar de generar un
  prompt malformado silenciosamente.
- **`trim_blocks` + `lstrip_blocks`**: los bloques `{% %}` no
  introducen saltos de línea espurios.
- **`FileSystemLoader(app/prompts/)`**: permite usar `{% include "..." %}`
  desde cualquier subdirectorio.

Versionado: para iterar en un prompt sin tocar el existente, copia
`v1/` a `v2/`, modifica, y cambia el parámetro `version="v2"` en el
loader o en el llamador. Esto permite A/B testing y rollback rápido.
En esta rama solo existe `v1/`.

---

## Tests

```bash
# Solo los tests del template (no tocan APIs externas, < 1s):
uv run pytest tests/prompts/ -v

# Todos los tests (incluye health, requiere el .venv configurado):
uv run pytest -v

# Dentro del contenedor (los tests/ no están copiados; usar local):
uv run pytest
```

En esta rama hay:

- `tests/test_health.py` — heredado de pre-session-02.
- `tests/prompts/test_estimation_v1.py` — 6 tests sobre el render del
  template Jinja2: interpolación literal de `description`, ramas de
  `output_format`, instrucción condicional de `detail_level`,
  humanización del `project_type`, inclusión del bloque `<examples>`,
  comportamiento de `StrictUndefined`.

---

## Endpoints disponibles

| Método | Path | Schema | Notas |
|---|---|---|---|
| `GET` | `/health` | — | Health check |
| `POST` | `/api/v1/estimate` | `EstimationRequest` → `EstimationResponse` | **Schema nuevo** desde pre-session-04: `{description, project_type, detail_level, output_format}` → `{text, prompt_version}`. Prompt renderizado vía Jinja2. |
| `POST` | `/api/v1/estimate/stream` | `StreamEstimationRequest` → SSE | **Legacy de session-03**: acepta `{transcription, num_examples}`. Eventos `delta` / `done` / `error`. Pendiente de migración o eliminación en session-04. |

### Por qué dos endpoints (estado transicional)

Esta rama deja el proyecto en un estado **intencionadamente híbrido**:

- `/estimate` es el endpoint nuevo del producto: formulario tipado,
  prompt compuesto con templates versionados.
- `/estimate/stream` se mantiene intacto desde session-03 para no
  perder la infraestructura de streaming, aunque ningún cliente del
  proyecto lo consume ya.

En session-04 (sesión en vivo) habrá que decidir si se migra el stream
al nuevo schema, se sustituye por uno compatible con Instructor +
structured outputs, o se elimina directamente. Hasta entonces, los dos
schemas coexisten — los del legacy con prefijo `Legacy*` en
`app/schemas/legacy_estimation.py` para que la separación sea obvia.

---

## Cache exact-match Redis

Toda llamada al LLM pasa primero por la cache. La clave es un SHA-256
sobre `{system_prompt, user_message, model, max_tokens, thinking_budget}`.
TTL = 24h.

| Caso | Comportamiento |
|---|---|
| Cache hit en `/estimate` | Respuesta inmediata con `cache_hit: true`. Sin llamada al LLM. |
| Cache hit en `/estimate/stream` | Un único evento SSE `delta` con la respuesta completa, luego `done`. |
| Redis caído al arrancar | `cache_connection_failed` en logs, `enabled=False`, app sigue funcionando sin cache. |
| Redis cae a mid-flight | Los métodos `get`/`set` registran un warning y devuelven `None` / no-op. |

### Cuándo hace hit `/estimate`

`/estimate` usa **selección random de ejemplos** por defecto (preserva
el demo del punto de saturación de session-02). Dos llamadas idénticas
suelen producir cache miss porque el system prompt cambia al rotar los
ejemplos. Hace hit solo cuando coinciden todos los parámetros y la
selección random produce el mismo subset.

### Cuándo hace hit `/estimate/stream`

`/estimate/stream` usa **selección determinista** (`deterministic=True`):
los primeros N ejemplos en el orden de `ESTIMATION_EXAMPLES`. La segunda
llamada idéntica garantiza cache hit, mismo system prompt → misma clave.

---

## Observabilidad

`structlog` configurado en `app.core.logging_config`. Dual output:

- `ENVIRONMENT=development` → `ConsoleRenderer` coloreado.
- `ENVIRONMENT=production` → `JSONRenderer` line-delimited.

Cada request HTTP arrastra un `request_id` (UUID4 o el del header
`X-Request-ID` si el cliente lo envía). El middleware lo bindea al
contexto vía `structlog.contextvars`, así que **todos** los logs
emitidos durante esa request lo llevan automáticamente.

Eventos clave a vigilar:

| Evento | Cuándo se emite |
|---|---|
| `cache_connected` | Al arrancar, si Redis responde al ping |
| `cache_connection_failed` | Al arrancar, si Redis no está disponible |
| `llm_cache_hit` | El wrapper encuentra la respuesta en cache |
| `llm_call_started` / `llm_call_completed` | Ciclo de llamada al LLM (no-stream) |
| `llm_stream_started` / `llm_stream_completed` | Ciclo de streaming |
| `llm_call_failed` / `llm_stream_failed` | Error durante la llamada |
| `stream_client_disconnected` | Cliente cerró el SSE antes del fin |

---

## Arquitectura

```
HTTP POST /api/v1/estimate                  HTTP POST /api/v1/estimate/stream
        │ (nuevo schema tipado)                       │ (schema legacy)
        ▼                                             ▼
  Router (delgado)                              Router (bridge sync→async)
        │                                             │
        ▼                                             ▼
  generate_estimation(request)               wrapper.complete_stream(...)
        │                                             │ (iterador sync)
        ▼                                             │
  render_estimation_prompt(request, v1)               ├─ loop.run_in_executor()
        │                                             │   (thread productor)
        ├── app/prompts/estimation/v1/system.j2       │
        ├── app/prompts/estimation/v1/user.j2         ├─ asyncio.Queue
        │   {% include estimation/v1/examples.j2 %}   │
        │                                             ▼
        ▼                                       sse_starlette.EventSourceResponse
  wrapper.complete(system, user, ...)                 │
        │                                             ▼
        ▼                                       chunks SSE al cliente
  EstimationResponse(text, prompt_version)


Wrapper (app/core/llm_wrapper.py) — sin cambios desde session-03:
        ┌────────────────────────────────────────┐
        │ 1. _make_cache_key (SHA-256)           │
        │ 2. ExactMatchCache.get(key) → hit?     │
        │    └─ sí: yield text / return dict     │
        │ 3. router.completion(...)              │
        │    └─ LiteLLM Router: primary→fallback │
        │ 4. normalize dict                      │
        │ 5. ExactMatchCache.set(key, value)     │
        │ 6. yield / return                      │
        └────────────────────────────────────────┘
```

### Decisiones de la rama

- **Borrón y cuenta nueva en `/estimate`**: el schema viejo no se
  preserva, no hay backwards compatibility. Las features de session-02
  (preprocessing, evaluation, thinking_budget, etc.) desaparecen del
  endpoint principal y reaparecerán en session-04 con un diseño nuevo.
- **El loader es el único punto que toca los templates**. Si encuentras
  `_env.from_string(...)` o `open()` de archivos `.j2` fuera de
  `app/prompts/loader.py`, es un bug.
- **`StrictUndefined` siempre activo**: render rompe con error si una
  variable no está definida. Detecta typos y refactors a medias.
- **`evaluation_service.py` y `context/examples.py` quedan en standby**:
  no se invocan, pero siguen compilando con imports al paquete
  `legacy_estimation`.
- **El Streamlit pasa de chat a formulario** y **deja de consumir
  SSE**. La UX de streaming queda solo en el endpoint legacy hasta que
  session-04 decida qué hacer con él.

---

## Qué entra en session-04

- **Structured outputs** con Instructor + Pydantic para forzar JSON
  validado en lugar de texto libre.
- **Guardrails** de input/output: Moderation API, validators de PII,
  scope checks, LLM-as-judge.
- **Cache semántico** con embeddings + `redisvl.SemanticCache`, en
  paralelo al cache exact-match actual.
- **Decisión sobre `/estimate/stream`** y `legacy_estimation.py`:
  migrar, reemplazar o eliminar.
- **Posible extracción del Streamlit** a otro proyecto separado.
