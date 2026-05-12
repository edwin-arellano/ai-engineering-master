# Estimator CAG

Servicio FastAPI que recibe la transcripción de una reunión con un
cliente y devuelve una estimación de software generada por un LLM,
usando arquitectura CAG (Cache Augmented Generation): el contexto de
ejemplos previos se inyecta directamente en el system prompt en cada
llamada — sin base de datos ni retrieval semántico.

## Requisitos

- Python 3.11+
- `uv` instalado ([instalación](https://docs.astral.sh/uv/getting-started/installation/))
- Una API key activa en al menos uno de los dos proveedores: Anthropic u OpenAI.

## Setup

```bash
cd estimator-cag
cp .env.example .env
# Edita .env y rellena ANTHROPIC_API_KEY (o OPENAI_API_KEY) según el proveedor que vayas a usar
uv sync
```

## Arrancar el servicio

```bash
uv run uvicorn app.main:app --reload
```

El servicio queda disponible en:

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Probar el endpoint de estimación

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "transcription": "El cliente, una empresa de logística mediana con 80 empleados, necesita una aplicación móvil para que sus conductores reporten incidencias en ruta (averías, retrasos, daños en mercancía). La app debe permitir adjuntar fotos con geolocalización, funcionar offline cuando no hay cobertura y sincronizar al recuperar conexión. Por otro lado, el panel de administración web debe mostrar un mapa con incidencias en tiempo real, permitir asignar incidencias a un responsable y generar reportes semanales. Quieren login con Google Workspace. El plazo ideal sería 8 semanas."
}
JSON
```

O con el archivo de prueba incluido:

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{transcription: .}' transcriptions/sample_meeting.txt)"
```

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Proveedor del LLM: `anthropic` o `openai` |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Modelo concreto del proveedor seleccionado |
| `LLM_TEMPERATURE` | `0.3` | Temperatura de muestreo |
| `LLM_MAX_TOKENS` | `2048` | Máximo de tokens en la respuesta |
| `ANTHROPIC_API_KEY` | *(vacío)* | API key de Anthropic |
| `OPENAI_API_KEY` | *(vacío)* | API key de OpenAI |
| `ENVIRONMENT` | `development` | Entorno de ejecución |
| `LOG_LEVEL` | `INFO` | Nivel de logging |

> Si cambias `LLM_PROVIDER`, recuerda actualizar `LLM_MODEL` para que
> apunte a un modelo válido del nuevo proveedor (por ejemplo,
> `gpt-4o-mini` si pasas a OpenAI).

## Estructura del proyecto

```
estimator-cag/
├── app/
│   ├── main.py             ← Aplicación FastAPI + /health
│   ├── config.py           ← Settings (Pydantic BaseSettings)
│   ├── routers/            ← Endpoints HTTP (delgados)
│   ├── services/           ← Lógica de negocio (LLM)
│   ├── schemas/            ← Contratos Pydantic
│   └── context/            ← Datos estáticos inyectados en el prompt (CAG)
├── tests/                  ← Suite de pytest
└── transcriptions/         ← Transcripciones de prueba
```

## Tests

```bash
uv run pytest
```

## Arquitectura CAG en esta rama

- **Sin** abstracción de proveedores: el `LLMService` despacha entre
  Anthropic y OpenAI con un `if/elif`. La capa de abstracción
  (LiteLLM) entra en la **sesión 3**.
- **Sin** cache: cada petición llama al LLM. El cache exact-match
  Redis entra en la **sesión 3**; el cache semántico, en la **sesión 4**.
- **Sin** streaming: la respuesta se devuelve completa. SSE entra en
  la **sesión 3**.
- **Sin** structured outputs: la estimación es texto libre en Markdown.
  Instructor + JSON Schema entran en la **sesión 4**.
- **Single-turn**: el servicio es transaccional, no conversacional.
  No se gestiona historial de mensajes.
