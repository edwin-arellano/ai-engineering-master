# Estimator CAG

Servicio FastAPI que recibe la transcripción de una reunión con un
cliente y devuelve una estimación de software generada por un LLM,
usando arquitectura **CAG** (Cache Augmented Generation): el contexto
de ejemplos previos se inyecta directamente en el system prompt en cada
llamada — sin base de datos ni retrieval semántico.

En la rama `session-02` el servicio evoluciona desde un scaffolding
mínimo a una primera versión "production-aware": parametrizable por
request (número de ejemplos, formato de salida, preprocesado de la
transcripción, modelo override, extended thinking en Anthropic, etc.),
con **evaluación estructural automática** del output, **dockerizado**
para arranque uniforme, y con **scripts de demo** que reproducen las
pruebas hechas en la sesión en vivo.

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

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Proveedor del LLM: `anthropic` o `openai` |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Modelo concreto del proveedor seleccionado |
| `LLM_TEMPERATURE` | `0.3` | Temperatura de muestreo (default) |
| `LLM_MAX_TOKENS` | `4000` | Máximo de tokens en la respuesta (default) |
| `ANTHROPIC_API_KEY` | *(vacío)* | API key de Anthropic |
| `OPENAI_API_KEY` | *(vacío)* | API key de OpenAI |
| `DEFAULT_NUM_EXAMPLES` | `3` | Número de ejemplos a inyectar (informativo; el request manda) |
| `DEFAULT_PREPROCESSING` | `none` | Estrategia de preprocesado por defecto |
| `DEFAULT_OUTPUT_FORMAT` | `markdown` | Formato de salida por defecto |
| `ENVIRONMENT` | `development` | Entorno de ejecución |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

> Si cambias `LLM_PROVIDER`, recuerda actualizar `LLM_MODEL` para que
> apunte a un modelo válido del nuevo proveedor (por ejemplo,
> `gpt-4o-mini` si pasas a OpenAI).

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

## Arquitectura

El flujo de una petición pasa por las siguientes fases dentro de
`app.services.llm_service.generate_estimation`:

```
HTTP POST /api/v1/estimate
        │
        ▼
  Router (delgado)
        │
        ▼
  generate_estimation(request)
        │
        ├─[ if preprocessing == two_phase ]
        │      └── _extract_requirements()  ← 1ª llamada LLM
        │
        ├── build_system_prompt(
        │      num_examples, example_format,
        │      output_format, preprocessing)
        │
        ├── _call_llm(...)                 ← 2ª llamada LLM
        │      ├── _call_anthropic(...)    ← extracts text blocks, normalizes
        │      └── _call_openai(...)       ← extracts message.content, normalizes
        │
        ├─[ if evaluation ]
        │      └── evaluate_estimation()   ← regex + parseo, sin IA
        │
        └── EstimationResponse(
              estimation, model, provider,
              finish_reason, latency_ms,
              token_usage, evaluation, ...)
```

### Decisiones de la rama

- **Sin abstracción de proveedores**: dispatch manual entre Anthropic
  y OpenAI.
- **Sin cache de respuestas**: cada petición llama al LLM.
- **Sin streaming**: la respuesta se devuelve completa.
- **Sin structured outputs forzados**: `output_format: "json"` se
  apoya en instrucciones del prompt; el LLM puede fallar y devolver
  Markdown, lo cual el evaluator detectará.
- **Evaluación sólo nivel 1**: 100% determinista (regex, parseo,
  aritmética). Niveles 2 y 3 entran en sesiones posteriores.
- **Selección aleatoria de ejemplos**: cuando `num_examples < 5` se
  hace `random.sample`.
