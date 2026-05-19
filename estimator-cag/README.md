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

`docker-compose.yml` monta `./app` como volumen read-only y arranca
`uvicorn --reload`, así que cualquier cambio en el código se recarga
sin reconstruir la imagen. Desde S04 la imagen de Redis es
`redis/redis-stack` (no `redis:7-alpine`) porque `redisvl.SemanticCache`
requiere el módulo **RediSearch**. RedisInsight queda expuesto en
`http://localhost:8001` para inspeccionar el índice y las keys.

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
| Project description | `st.text_area` (10–4000 chars) | `description` |
| Project type | `st.selectbox` | `mobile_app` / `web_saas` / `internal_tool` / `integration` / `other` |
| Detail level | `st.selectbox` | `summary` / `medium` / `detailed` |
| Output format | `st.radio` horizontal | `phases_table` / `line_items` / `narrative` |

> Desde S04 el resultado se renderiza estructurado: `st.metric` × 3
> (duration / cost / confidence), `st.progress` con la confianza global,
> `st.dataframe` con la tabla de fases y sus assumptions, badge
> `📦 From cache (...)` cuando aplica y `st.warning` cuando el modelo
> marca la petición como `Out of scope:`.

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

---

## Sesión 04

La rama `session-04` añade structured outputs con Instructor, cinco
capas de guardrails y cache semántico con `redisvl`, eliminando todo
el código legacy de streaming y evaluación.

### Cambios destructivos respecto a `pre-session-04`

- **El endpoint `POST /api/v1/estimate/stream` ya no existe.** El
  servicio sirve una única ruta: `POST /api/v1/estimate` con respuesta
  síncrona y estructurada.
- **El wrapper LiteLLM expone un único método público**
  `complete_structured(...)`. `complete()` y `complete_stream()` se
  han eliminado.
- **`app/schemas/legacy_estimation.py`, `app/services/evaluation_service.py`
  y `app/context/`** se han borrado del repo.
- **La dependencia `sse-starlette`** se ha retirado.
- **`app/core/cache.py`** se ha movido (con `git mv`) a
  `app/services/cache/exact_match_cache.py` y convive ahora con
  `app/services/cache/semantic_cache.py`.
- **El enum `ProjectType`** sustituye `data_pipeline` por
  `integration` y añade `other`.

### Variables de entorno nuevas

| Variable | Default | Descripción |
|---|---|---|
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | Modelo de embeddings de OpenAI usado por el cache semántico |
| `EMBEDDINGS_DIMENSIONS` | `1536` | Dimensiones del vector |
| `SEMANTIC_CACHE_ENABLED` | `true` | Activa el cache semántico |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Umbral de similitud para hit (0-1) |
| `SEMANTIC_CACHE_TTL_SECONDS` | `86400` | TTL del cache semántico |
| `SEMANTIC_CACHE_NAME` | `estimator_semantic_cache` | Nombre del índice en Redis |
| `MODERATION_ENABLED` | `true` | Activa OpenAI Moderation API en input |
| `MIN_CONFIDENCE_PCT` | `30` | Umbral mínimo de confianza para cachear |
| `PROMPT_VERSION` | `v2` | Versión del template que se renderiza por defecto |

### Las cinco capas de guardrails

1. **Pydantic sintáctico del input** — `EstimationRequest` valida
   tipos, rangos y longitudes. Política: exception (Pydantic).
2. **Validación semántica del input** — `app/guardrails/input_guardrails.py`
   aplica regex de prompt injection, detección de PII (email, IBAN,
   teléfono) y OpenAI Moderation API en cascada. Política: exception
   (`InputGuardrailError` → HTTP 400).
3. **System prompt robusto** — `app/prompts/estimation/v2/system.j2`
   incluye `<scope>` y `<numerical_constraints>` que instruyen al
   modelo a devolver `"Out of scope:"` con totales a cero cuando la
   descripción no encaja. Política: filter (comportamiento del modelo).
4. **Validators de `EstimationResult`** — `total_must_match_sum_of_phases`
   (±1 semana, ±5% coste) y `low_confidence_must_be_explicit`.
   Política: fix con retry (Instructor reintenta hasta 3 veces).
5. **Filtro de salida** — `app/guardrails/output_guardrails.should_cache_result`
   evita persistir respuestas out-of-scope o de baja confianza.
   Política: filter (no cachea pero sí devuelve al cliente).

### Pipeline del endpoint

```
POST /api/v1/estimate
 │
 ├─ Input guardrails (regex + PII + Moderation)
 │   └─ falla → 400 con {error, category, reason}
 │
 ├─ Exact-match cache (Redis, key = SHA256(request + prompt_version))
 │   └─ hit → cached=true, cache_level="exact_match"
 │
 ├─ Semantic cache (redisvl, bucket = v2:project_type:detail_level:output_format)
 │   └─ hit → popular exact-match, cached=true, cache_level="semantic"
 │
 ├─ Render prompt v2 (Jinja2)
 │
 ├─ Instructor + LiteLLM Router (Anthropic primary, OpenAI fallback)
 │   └─ Pydantic validators fallan → Instructor retry (hasta 3 veces)
 │
 ├─ Output guardrails (out-of-scope o low confidence → no cachear)
 │
 └─ Respuesta: {result, prompt_version, cached, cache_level}
```

### Reorganización del árbol

```
app/
├── core/
│   ├── llm_wrapper.py        ← solo complete_structured
│   └── logging_config.py
├── guardrails/               ← nuevo
│   ├── input_guardrails.py
│   └── output_guardrails.py
├── prompts/
│   └── estimation/
│       ├── v1/               ← intacto (referencia)
│       └── v2/               ← nuevo (scope + numerical_constraints + ejemplos JSON)
├── routers/
│   └── estimations.py        ← un único endpoint
├── schemas/
│   └── estimation.py         ← Phase + EstimationResult + EstimationResponse rico
└── services/
    ├── cache/                ← nuevo paquete
    │   ├── exact_match_cache.py   ← movido desde app/core/cache.py
    │   └── semantic_cache.py
    └── llm_service.py        ← pipeline completo
```

> Los scripts curl en `examples/` corresponden al endpoint stream
> legacy y a respuestas en texto libre. Se mantienen sin mantenimiento
> como referencia histórica y **no se actualizan** para session-04.

---

## Pre-sesión 05

La rama `pre-session-05` transforma el servicio de "single-shot tipado" a
"conversacional con memoria y adjuntos". El endpoint `POST /api/v1/estimate`
desaparece; toda la interacción pasa por sesiones.

### Endpoints

- `POST /api/v1/sessions` → crea una sesión vacía y devuelve
  `{session_id, created_at}` (HTTP 201).
- `POST /api/v1/sessions/{session_id}/estimate` → `multipart/form-data` con
  `transcript`, `project_type`, `detail_level`, `output_format` y un campo
  `attachments` opcional (cero o más archivos PDF/.docx). Devuelve
  `EstimationResponse` (HTTP 200).

Errores HTTP estructurados:

- `400 input_guardrail` con `detail.category` cuando regex/PII/Moderation
  rechazan el input.
- `404 session_not_found` si el `session_id` no existe o caducó por TTL idle.
- `413 attachment_too_large` si un adjunto excede `ATTACHMENT_MAX_BYTES`.
- `415 unsupported_attachment` si el MIME no es PDF ni .docx.

### Variables de entorno nuevas

| Variable | Default | Descripción |
|---|---|---|
| `MAX_TURNS` | `6` | Pares user+assistant que sobreviven a la ventana deslizante del historial. |
| `SESSION_IDLE_TTL_SECONDS` | `86400` | Inactividad tras la cual una sesión se purga (24 h). |
| `ATTACHMENT_MAX_BYTES` | `10485760` | Tamaño máximo aceptado por adjunto (10 MB). |
| `PROMPT_VERSION` | `v3` | Templates `app/prompts/estimation/v3/` activos por defecto. |

### Memoria conversacional: tres estructuras independientes

- `ConversationHistory` (`app/schemas/session.py`): array de pares
  user/assistant con ventana deslizante (`MAX_TURNS`). El system prompt **no
  vive aquí**; se regenera en cada turno desde el `project_metadata`. Eso es
  lo que da resistencia al truncado.
- `ProjectMetadata`: hechos destilados sobre el proyecto en curso
  (`project_name`, `assumed_team_size`, `mentioned_technologies`,
  `agreed_scope`). Sobrevive al truncado del historial.
- `Session`: agregado con `session_id`, timestamps, `history` y
  `project_metadata`. Vive en memoria del proceso vía `SessionStore`
  (`threading.Lock` + TTL idle).

### Adjuntos: camino B (extracción local)

El servicio IA extrae el texto del PDF (`pypdf`) o del .docx (`python-docx`)
y lo concatena al user prompt con delimitadores
`<attachment filename="...">...</attachment>`. Mantiene la portabilidad del
wrapper Instructor/LiteLLM (no nos atamos a la Files API de un proveedor) y
prepara la pieza de chunking que entra con RAG en el módulo 3.

Coste asumido: se pierde la información visual del PDF (diagramas
embebidos). Para estimaciones de software, el texto cubre el caso 95%.

### LLM extractor de `project_metadata`

Tras cada turno, una segunda llamada a `complete_structured` con
`response_model=ProjectMetadataUpdate` devuelve un patch con los hechos
nuevos que aportó el intercambio. `ProjectMetadata.apply_patch` lo aplica
**sin sobrescribir hechos previos con nulos**: las listas se mergean por
unión y los escalares solo se actualizan cuando el patch trae valor no nulo.

Coste: una llamada extra al LLM por turno (~1-2 s, céntimos). Aceptable a
cambio de robustez frente a reformulaciones e idiomas mezclados.

### Pipeline del endpoint conversacional

```
POST /api/v1/sessions/{id}/estimate
 │
 ├─ Cargar sesión (404 si caducada)
 │
 ├─ Extraer adjuntos (415 si MIME no soportado, 413 si demasiado grande)
 │
 ├─ Input guardrails sobre transcript + texto de adjuntos
 │   └─ falla → 400 con {error, category, reason}
 │
 ├─ Render prompt v3 con project_metadata + bloque <attachments>
 │
 ├─ messages = history.to_api_messages(system) + último user
 │
 ├─ wrapper.complete_structured_with_messages → EstimationResult
 │
 ├─ extract_metadata_update (segunda llamada LLM)
 │   └─ apply_patch sobre session.project_metadata
 │
 ├─ history.append_turn (ventana deslizante)
 │
 └─ session_store.save → 200 EstimationResponse
```

### Cache de S04: infraestructura dormida

`app/services/cache/` sigue en el repo (clases `ExactMatchCache`,
`SemanticCacheService`, `make_exact_match_key`, `make_bucket_key`) pero
**no se invoca** desde el flujo conversacional. Justificación: la cache key
en multi-turno depende de `(transcript, attachments, history, metadata)`,
así que la tasa real de hits sería ínfima.

Mantener la infra disponible es barato y deja la puerta abierta a usos
futuros (cachear extracciones de PDF por hash, por ejemplo). El shim
`app/schemas/estimation_compat.CachedRequest` da a las clases del cache un
tipo válido sin reintroducir `EstimationRequest`.

### Limitaciones documentadas

- **Multi-worker**: `SessionStore` vive en memoria del proceso. Con
  `uvicorn --workers N` un cliente puede aterrizar en un worker que no
  conoce su `session_id`. Para multi-worker hay que migrar a Redis como
  backend de sesiones; queda fuera de pre-S05.
- **Persistencia**: el reinicio del servicio borra todas las sesiones.
- **Panel de metadata en Streamlit**: optimista, sin endpoint `GET` de
  sesión.
