# Máster AI Engineering — Entregables

Repositorio con los entregables del Máster AI Engineering de LIDR.

Cada sesión tiene dos ramas: `pre-session-NN` (entregable previo a la
sesión en vivo) y `session-NN` (estado tras la sesión en vivo).

> La Sesión 1 (notebook Jupyter con primeras llamadas a APIs de LLM) se
> omite de este repositorio porque su entregable es un notebook en
> Google Colab, no un proyecto Python.

## Estructura

- `estimator-cag/`: servicio FastAPI que construye un estimador de
  software basado en transcripciones de reunión. Crece a lo largo de
  todas las sesiones (CAG → wrapper → formulario → structured outputs
  → guardrails → cache semántico).

## Ramas

| Rama | Contenido |
|---|---|
| `pre-session-02` | Scaffolding FastAPI con arquitectura CAG (`POST /api/v1/estimate`) |
| `session-02` | Servicio parametrizable: preprocesado, evaluación estructural, JSON/Markdown, Dockerización, 10 scripts curl de demo |
| `pre-session-03` | Cliente conversacional Streamlit con streaming token a token, importando `build_system_prompt` y los ejemplos del backend |
| `session-03` | Wrapper LiteLLM con fallback, cache Redis, endpoint SSE `/api/v1/estimate/stream`, observabilidad `structlog`, Streamlit desacoplado a cliente HTTP puro |
| `pre-session-04` | Endpoint `/api/v1/estimate` con formulario tipado (description + 3 enums), prompts Jinja2 versionados bajo `app/prompts/estimation/v1/`, Streamlit con `st.form` en lugar de chat |
| `session-04` | Structured outputs con Instructor sobre LiteLLM Router, cinco capas de guardrails (regex + PII + Moderation + Pydantic validators + filtro de salida), cache semántico con `redisvl.SemanticCache`, template `v2` con `<scope>` y `<numerical_constraints>`, Streamlit que renderiza `EstimationResult`. Elimina endpoint stream y todo el código legacy. |
| `pre-session-05` | Servicio conversacional con sesiones: `POST /api/v1/sessions` + `POST /api/v1/sessions/{id}/estimate` (multipart). Memoria persistente entre turnos (`ProjectMetadata` separada del historial), ventana deslizante de mensajes, adjuntos PDF/.docx con extracción local (`pypdf` + `python-docx`), template `v3` con bloques condicionales y LLM extractor de metadata. Elimina el endpoint single-shot; el cache de S04 queda dormido. |
| `session-05` | Compresión híbrida con anclas (regex sobre mensajes user; reemplaza la ventana deslizante como default), tier dinámico resuelto en runtime y materializado como bloque condicional en `v3/system.j2`, patrón **Actor-Critic-Boss** con tres roles aislados (`_run_actor` reutilizado, `CriticService` con `field_path`+`suggested_fix`, `BossService` con presupuesto de 3 iteraciones y síntesis con fallback graceful), evals standalone (`evals/run.py`) con golden dataset de ~16 casos. `EstimationResponse` enriquecido con `tier`, `estimation_mode` y rastro de iteraciones del ACB. |
| `pre-session-06` | Baseline cuantitativo del CAG (**stress test**, no RAG). Instrumentación aditiva: wrapper observable con tokens/coste por llamada vía sink `TurnMetrics` + `app/core/pricing.py`, evento `turn_observed` (13 campos), framework de métricas de evals (`MetricResult` + `run_all_metrics`), endpoint debug `GET /api/v1/sessions/{id}`. Un runner HTTP (`evals/stress/run.py`) somete el CAG a tres escenarios multi-turno × tamaños de adjunto y produce `results.csv` + `REPORT.md` con tres curvas (latencia/tokens, coste/turno, recall/N). Mide en qué eje se rompe el CAG antes de adoptar RAG. |
| `session-06` | **Módulo de ingesta de datos** aislado (`app/ingest/`), pasos 1–4 del "viaje del dato". Catálogo de fuentes como código (`data_catalog.yaml` validado con Pydantic), contrato `Document` canónico, pipeline `loaders → parsers (json/txt/xlsx) → cleaning (pandas + Pandera con política cuarentena/descarte) → normalizers → orchestrator`, endpoint `POST /api/v1/ingest` (bloqueante en threadpool). Evaluador de fuentes vía LLM sobre solo hechos (sin contenido crudo) y evaluador de viabilidad CAG/RAG que consume el baseline pre-S06. Seed sintético reproducible (gitignored), catálogo comiteado. Sin vectorización ni Presidio. |
| `pre-session-07` | **Pipeline de embeddings y chunking** aislado (`app/embedding_pipeline/`, ejercicio S7-01). `JSONStructuralChunker` (un componente = un chunk, contextual chunk headers, metadata filtrable fuera del texto, `token_count` con tiktoken), `LiteLLMEmbedder` (`text-embedding-3-small` 1536 dims, batching + retry exponencial, coste por tokens reales), endpoint `POST /embeddings/ingest` (v0.7.0). CLI `compare.py` con coseno a mano (sin numpy/sklearn) y `sanity_check.py` → `SANITY_CHECK.md`. Sample sintético comiteado (15 presupuestos anidados). Sin persistencia ni retrieval (S08 pgvector). |
| `session-07` | **Refactor de arquitectura por capas** (`git mv`, historia preservada): `app/` se reorganiza en `foundations` (infra: wrapper/config/logging/metrics/pricing/cache/prompts), `domain` (schemas), `api/routers`, `ingest` (S06 intacto) y `generation/{cag,agentic,rag}`. **Suite de 8 estrategias de chunking** sobre la interfaz `Chunker` común (`structural`, `recursive`, `fixed_size`, `sentence_window`, `hierarchical`, `semantic`, `propositional`, `contextual_retrieval`) con registry e inyección; huérfanos (`is_orphan`) que no se vectorizan; comparador con métricas (chunks/huérfanos/percentiles/latencia/coseno vs query) y comparación de dimensiones 1536 vs 768. Endpoint `POST /embeddings/ingest` gana `strategy` opcional. Sin persistencia/pgvector, retrieval ni Presidio (S08+). |

## Cómo arrancar

Cada subproyecto tiene su propio README con instrucciones específicas.
Empieza por `estimator-cag/README.md`.
