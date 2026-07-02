# pre-S11 — Generación verificable: citación a nivel de línea + baseline RAGAS

Entregable del pre-work de la Sesión 11 sobre el **path single-pass RAG**
(`/api/v1/estimate-from-transcript`). Dos partes: (1) citación verificable por
línea con detección de citas colgantes, (2) baseline de calidad con RAGAS.

---

## 1. Baseline RAGAS (4 métricas × 5 consultas)

Juez: `gpt-4o-mini`. Embeddings: `text-embedding-3-small`. Corpus single-pass con
`search_mode=vector`, `reranking=False` (defaults de producción). Generación con el
prompt `v2` (atribución por línea). Reproducible con `scripts/measure_ragas.py`
(fases `generate` / `evaluate`; ver nota de entorno al final).

| id  | consulta                                              | faithfulness | answer_relevancy | context_precision | context_recall |
|-----|-------------------------------------------------------|:------------:|:----------------:|:-----------------:|:--------------:|
| q01 | E-commerce (catálogo, carrito, panel admin)           |    0.500     |      0.413       |       0.935       |     0.333      |
| q02 | API banca móvil (OAuth 2.0 + PSD2)                    |    0.409     |      0.329       |       0.986       |     0.333      |
| q03 | Integración de pagos Stripe (suscripciones, webhooks) |    0.591     |      0.000       |       0.500       |     0.667      |
| q04 | Telemedicina (citas + notas clínicas)                 |    0.730     |      0.000       |       0.989       |     0.667      |
| q05 | Gestión documental interna con RBAC                   |    0.750     |      0.000       |       0.000       |     0.000      |
| **—** | **Promedio**                                        |  **0.596**   |    **0.148**     |     **0.682**     |   **0.400**    |

### Nota (lo más llamativo)

- **`answer_relevancy` se desploma a 0 en q03–q05**: RAGAS genera preguntas inversas
  desde la respuesta y mide su similitud con la consulta; las respuestas más verbosas
  y de **confianza baja/media** (q03 `low`, q05 `low`) — con mucho `reasoning` en prosa
  y, en q05, sin módulos — disparan el clasificador de "respuesta no comprometida".
  Señal de que el `render_answer` del arnés debería separar la estimación del
  razonamiento, no que la recuperación falle.
- **`faithfulness` moderada (0.60 promedio)** es coherente con la **citación gruesa**:
  el `evidence` se copia a nivel de chunk (p. ej. `"Estimated hours: 90"`), no a nivel de
  cifra anclada en el presupuesto original; el juez penaliza afirmaciones derivadas
  (conversión horas→días, módulos sintetizados) que no aparecen literales en el contexto.
- **`context_recall` flojo (0.40) y nulo en q05**: el caso de gestión documental con RBAC
  es de **cobertura débil a propósito** (el corpus no lo tiene; el análogo LMS no cubre
  versionado/indexado). `context_precision=0` en q05 confirma que ningún chunk recuperado
  respalda el `ground_truth` — exactamente el caso donde la generación honesta debe marcar
  **asunciones** en vez de inventar cifras.

---

## 2. Reporte de citaciones (real)

Salida de `verify_citations(estimate, context)` para la transcripción
*"API de banca móvil con autenticación OAuth 2.0 y cumplimiento PSD2"* (q02). La
estimación generada con `v2` cita por línea `source_id` + `document_id` + `evidence`
**literal** copiado del chunk:

```
- [EVIDENCIA] PSD2 compliance module (11.25 d)
    source_id=BUD-2024-001::PSD2-002::task | document_id=BUD-2024-001 | evidence='Estimated hours: 90'
- [EVIDENCIA] OAuth 2.0 authentication backend (15.0 d)
    source_id=BUD-2024-001::AUTH-001::task | document_id=BUD-2024-001 | evidence='Estimated hours: 120'
- [EVIDENCIA] Transaction ledger service (13.75 d)
    source_id=BUD-2024-001::TXN-003::task | document_id=BUD-2024-001 | evidence='Estimated hours: 110'
```

**CitationReport (sin manipular)** — las 3 líneas resuelven contra el contexto recuperado:

```json
{ "total_lines": 3, "grounded_lines": 3, "insufficient_lines": 0, "dangling": [] }
```

### Detección de cita colgante (introducida a propósito)

Inyectando un `source_id` inexistente (`BUD-9999-999::FAKE-01`) en una de las tareas,
la verificación post-generación lo detecta, lo reporta y lo loguea correlacionado por
`request_id` (vía `structlog.contextvars`):

```
[warning] rag.dangling_citations  count=1 refs=['BUD-9999-999::FAKE-01']
```

```json
{ "total_lines": 3, "grounded_lines": 2, "insufficient_lines": 0, "dangling": ["BUD-9999-999::FAKE-01"] }
```

La integridad es **estructural** (la fuente existió en el contexto recuperado), **no
semántica** (que la fuente diga lo que la línea afirma — eso se difiere al directo S11).
Política por defecto: detectar + reportar + loguear (sin bloquear). Con
`REJECT_ON_DANGLING=true`, una cita colgante hace que el endpoint responda **422**.

---

## 3. Qué entró (resumen técnico)

- **`Citation` enriquecida**: `source_id` (= chunk_ref, verificable contra
  `included_refs`) + `document_id` (= budget_id, presupuesto histórico) + `evidence`
  (span/cifra **literal** del chunk). Los tres obligatorios y no vacíos.
- **Bloque de contexto** (`augmentation.py`) expone ahora `document_id` además de
  `source_id`, para que el modelo pueda atribuir y copiar evidencia.
- **Prompt de generación `v2`** (`rag_estimation/v2/system.j2`): fuerza atribución por
  línea con `source_id` + `document_id` literales y `evidence` verbatim; tareas sin
  soporte → `is_assumption=true` con `sources=[]`. `v1` queda para rollback/A-B.
  Equivalencia documentada: **`grounded ≡ not is_assumption`**.
- **`verify_citations` → `CitationReport`** (`total_lines`, `grounded_lines`,
  `insufficient_lines`, `dangling`). Política configurable (`REJECT_ON_DANGLING`).
- **Contrato HTTP enriquecido (no roto)**: `EstimateFromTranscriptResult` gana
  `citation_report`; `invalid_citations` se conserva como alias de `citation_report.dangling`.
- **Golden set** con `ground_truth` por consulta (estimación de referencia experta).

### Diferido al directo S11
Detección de alucinaciones (verificación **semántica**), compresión/destilación de
contexto, síntesis multi-presupuesto en rangos honestos, salud/reindexación del índice,
y las extensiones de RAGAS (comparativas/regresiones).

### No tocado (verificable por construcción)
El **flujo invertido** (`/api/v1/estimate-structured`, `StructuredEstimate`) deriva las
horas de vecinos históricos de forma **determinista**: su procedencia (`TaskNeighbor`)
ya resuelve, localiza y es trazable. No se le aplica esta verificación ni RAGAS.

---

## Nota de entorno (RAGAS)

La app fija `openai` **2.x** (LiteLLM/Instructor) y el juez de RAGAS necesita el stack
`langchain` con `openai` **1.x** — incompatibles en un mismo proceso. Por eso
`scripts/measure_ragas.py` se ejecuta en **dos fases**:

```bash
# Fase 1 — genera las 4 entradas RAGAS (venv de la app, openai 2.x):
PYTHONPATH=. uv run python scripts/measure_ragas.py generate

# Fase 2 — evalúa en entorno aislado (ragas + openai 1.x), sin importar la app:
uv run --no-project \
  --with 'ragas>=0.2,<0.3' --with 'langchain-openai<0.3' \
  --with 'openai<2' --with datasets --with pandas \
  python scripts/measure_ragas.py evaluate
```

Versión usada: `ragas==0.2.15`. (RAGAS 0.4.x importa
`langchain_community.chat_models.vertexai`, eliminado en el stack langchain 1.x → se
pinea la serie 0.2.)
