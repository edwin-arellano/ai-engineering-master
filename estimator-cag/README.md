# Estimator CAG

Servicio FastAPI que recibe la transcripción de una reunión con un
cliente y devuelve una estimación de software generada por un LLM,
usando arquitectura **CAG** (Cache Augmented Generation): el contexto
de ejemplos previos se inyecta directamente en el system prompt en cada
llamada — sin base de datos ni retrieval semántico.

En la rama `session-03` el servicio incorpora cinco capas que lo
acercan a "producción": **wrapper LiteLLM** con fallback automático
Anthropic ↔ OpenAI, **cache exact-match Redis** con TTL 24h, endpoint
**`/api/v1/estimate/stream` con SSE** para respuesta token a token,
**observabilidad estructurada** con `structlog` (request_id por
petición, contexto vinculado, dual output dev/prod), y **Streamlit
desacoplado** (cliente HTTP puro que consume el SSE, sin importar nada
del backend).

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

## Interfaz conversacional (Streamlit)

`streamlit_app.py` es un **cliente HTTP puro**: no importa nada de
`app.*`. Consume el endpoint SSE `/api/v1/estimate/stream` del backend
FastAPI vía `httpx`. El contrato entre frontend y backend es solo HTTP
+ SSE — si mañana el frontend cambia a Next.js o Vue, el backend no se
toca.

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

### Funcionalidad

- Chat con persistencia de mensajes dentro de la sesión del navegador.
- Streaming token a token consumiendo SSE (`st.write_stream` sobre el
  iterador de `httpx.stream`).
- En **cache hit**, la respuesta llega instantánea en un único chunk.
- Si el backend está caído, se muestra un error claro y la página no
  se queda colgada.

### Qué cambia respecto a `pre-session-03`

| Aspecto | pre-session-03 | session-03 |
|---|---|---|
| Imports | `from app.services...` | Solo `httpx`, `streamlit`, `dotenv` |
| Cómo llama al LLM | SDK directo en el Streamlit | HTTP SSE al backend |
| Multi-turn | Sí (envía historial) | No (single-shot por petición) |
| Selección de proveedor | `LLM_PROVIDER` en el Streamlit | El backend decide vía LiteLLM Router |

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

## Parámetros del request

`POST /api/v1/estimate` acepta un cuerpo JSON con los siguientes campos.
Sólo `transcription` es obligatorio.

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `transcription` | string (≥50) | — | Transcripción de la reunión. |
| `num_examples` | int 0..5 | `3` | Cuántos ejemplos CAG inyectar. `0` = sin contexto. |
| `example_format` | `"markdown"` | `"markdown"` | Formato de serialización de los ejemplos. |
| `preprocessing` | `"none" \| "inline_cleaning" \| "two_phase"` | `"none"` | Estrategia de limpieza de la transcripción. |
| `model` | string \| null | `null` | Override del modelo. Si es `null`, usa `LLM_MODEL`. |
| `max_tokens` | int 1..16000 | `4000` | Máximo de tokens de salida. |
| `temperature` | float 0..2 | `0.3` | Temperatura de muestreo (ignorada con thinking en Anthropic). |
| `thinking_budget` | int 0..8000 | `0` | Budget de extended thinking (sólo Anthropic Claude 4.x). |
| `output_format` | `"markdown" \| "json"` | `"markdown"` | Formato exigido al LLM. |
| `usage` | bool | `true` | Incluir desglose de tokens en la respuesta. |
| `evaluation` | bool | `true` | Ejecutar evaluación estructural nivel 1. |

### Estrategias de preprocesado

- **`none`**: la transcripción se envía tal cual al modelo de
  estimación. Más rápido y barato.
- **`inline_cleaning`**: se prepende un bloque de instrucciones al
  system prompt indicando al modelo que ignore mentalmente las
  divagaciones, interrupciones y ruido típicos de una reunión. Una sola
  llamada al LLM.
- **`two_phase`**: dos llamadas. La primera extrae los requisitos
  limpios y deduplicados; la segunda genera la estimación a partir de
  esos requisitos. Mayor latencia y coste, pero el input de la
  estimación viene ya destilado. La salida de la primera fase se
  devuelve en el campo `extracted_requirements`.

---

## Estructura de la respuesta

Ejemplo abreviado de `EstimationResponse`:

```json
{
  "estimation": "## Estimación: ...\n\n### Desglose de tareas\n...",
  "model": "claude-haiku-4-5-20251001",
  "provider": "anthropic",
  "finish_reason": "end_turn",
  "preprocessing_type": "none",
  "output_format": "markdown",
  "latency_ms": 4218,
  "token_usage": {
    "input_tokens": 1842,
    "output_tokens": 612,
    "total_tokens": 2454,
    "preprocessing_input_tokens": 0,
    "preprocessing_output_tokens": 0
  },
  "extracted_requirements": null,
  "evaluation": {
    "has_title": true,
    "has_breakdown_table": true,
    "has_total_sections": true,
    "has_team_sections": true,
    "has_duration_sections": true,
    "declared_total_hours": 175,
    "sum_row_hours": 175,
    "hours_match": true,
    "finish_reason_ok": true,
    "score": 1.0,
    "issues": []
  }
}
```

- **`evaluation.score`** es un float entre `0.0` y `1.0`, promedio
  simple de 7 chequeos booleanos:
  `has_title`, `has_breakdown_table`, `has_total_sections`,
  `has_team_sections`, `has_duration_sections`, `hours_match`,
  `finish_reason_ok`.
- **`evaluation.issues`** es la lista de problemas detectados en
  lenguaje natural (vacía si todo está bien).
- **`finish_reason`** refleja la razón de finalización tal como la
  reporta el SDK (`"stop"`, `"end_turn"`, `"max_tokens"`, `"length"`,
  etc.). El evaluator considera `"stop"` y `"end_turn"` como OK.

---

## Demos de la sesión en vivo

La carpeta `examples/` contiene 10 scripts `curl + jq` que reproducen
las pruebas que se hicieron en la sesión. Para ejecutarlos, levanta el
servicio con Docker y verifica precondiciones:

```bash
docker compose up --build -d
./examples/00_setup.sh   # verifica que jq existe y /health responde
```

| Script | Qué demuestra |
|---|---|
| `01_basic.sh` | Llamada base con todos los defaults. |
| `02_no_examples.sh` | `num_examples=0`: sin CAG, score esperado bajo. |
| `03_one_example.sh` | `num_examples=1`: score sube notablemente. |
| `04_three_examples.sh` | `num_examples=3`: el "punto dulce". |
| `05_five_examples.sh` | `num_examples=5`: saturación de contexto. |
| `06_inline_cleaning.sh` | Preprocesado inline sobre la transcripción larga. |
| `07_two_phase.sh` | Preprocesado en dos fases; ver `extracted_requirements`. |
| `08_thinking_budget.sh` | Extended thinking en Anthropic con budget 2000. |
| `09_truncation.sh` | `max_tokens=200`: fuerza truncado, `finish_reason_ok=false`. |
| `10_json_output.sh` | `output_format="json"`: el evaluator parsea JSON. |

Para inspeccionar sólo el score y los issues de cualquier script:

```bash
./examples/04_three_examples.sh | jq '.evaluation.score, .evaluation.issues'
```

---

## Tests

```bash
# Dentro del contenedor
docker compose exec estimator uv run pytest -v

# O local
uv run pytest
```

En esta rama sólo se mantiene `tests/test_health.py`.

---

## Endpoints disponibles

| Método | Path | Schema | Notas |
|---|---|---|---|
| `GET` | `/health` | — | Health check |
| `POST` | `/api/v1/estimate` | `EstimationRequest` → `EstimationResponse` | Mantiene todas las features de session-02 (preprocessing, evaluation, JSON output, thinking_budget). La respuesta incluye `cache_hit`. |
| `POST` | `/api/v1/estimate/stream` | `StreamEstimationRequest` → SSE | Deliberadamente simple: solo `transcription` y `num_examples`. Eventos `delta` / `done` / `error`. |

### Por qué dos endpoints

`/estimate` y `/estimate/stream` cubren casos de uso distintos:

- `/estimate` es la API de poder: control fino del request, evaluación
  automática, preprocesado opcional, JSON estructurado. Ideal para
  pipelines automáticos y benchmarks.
- `/estimate/stream` es la API de UX: respuesta inmediata, cache hit
  instantáneo, parámetros mínimos. Optimizada para clientes
  conversacionales (Streamlit, web, móvil).

Mezclar las dos contamina ambas: un endpoint con preprocesado +
streaming tendría latencias impredecibles y eventos SSE mezclando
fases. Antonio fue explícito sobre esto en la sesión live.

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
        │                                            │
        ▼                                            ▼
  Router (delgado)                             Router (con bridge sync→async)
        │                                            │
        ▼                                            ▼
  generate_estimation(request)              wrapper.complete_stream(...)
        │                                            │ (iterador sync)
        ├─[ if preprocessing == two_phase ]          │
        │      └── wrapper.complete()                ├─ loop.run_in_executor()
        │                                            │   (thread productor)
        ├── build_system_prompt(deterministic=False) │
        │                                            ├─ asyncio.Queue
        ├── wrapper.complete(...)                    │   (puente entre thread y loop)
        │                                            │
        ├─[ if evaluation ]                          ▼
        │      └── evaluate_estimation()       sse_starlette.EventSourceResponse
        │                                            │
        └── EstimationResponse(...)                  ▼
              + cache_hit                       chunks SSE al cliente


Wrapper (app/core/llm_wrapper.py):
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

- **Wrapper completamente síncrono**: `complete` y `complete_stream`
  son síncronos. El puente al event loop async vive solo en el
  endpoint stream (`run_in_executor` + `asyncio.Queue`). Patrón
  consistente con el material del curso.
- **Toda llamada al LLM pasa por el wrapper**. Si encuentras un
  `litellm.completion(...)` fuera de `app/core/llm_wrapper.py`, es un
  bug.
- **`/estimate/stream` solo acepta `transcription` y `num_examples`**.
  Sin preprocessing, sin evaluation, sin thinking_budget. Por diseño.
- **Selección determinista de ejemplos en el endpoint stream**: sin
  esto, la cache no haría hits nunca.
- **La cache es best-effort**. Errores de Redis nunca rompen
  requests; solo log warnings.
- **Streamlit nunca importa de `app.*`**. Solo `httpx`, `streamlit`,
  `dotenv`.
