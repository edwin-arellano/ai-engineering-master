# Stress test del CAG — REPORT

> Baseline cuantitativo del CAG del `estimator` previo a la sesión 6.
> Corrida real contra el endpoint HTTP (`--http`), modo `actor`.
> 72 turnos (69 OK, 3 fallos), coste medido ≈ **$1.19 USD** (es un piso; ver caveats).

## Metodología

- **Runner**: `uv run python -m evals.stress.run --http http://127.0.0.1:8000 --scenarios growing,pivot,contradiction --attachment-sizes 0,100 --repeats 1`.
- **Escenarios**: `growing` (20 turnos, requisitos coherentes acumulándose), `pivot` (8 turnos, cambio de stack en t5), `contradiction` (8 turnos, cambio de presupuesto en t8). Cada uno con adjunto `0 KB` y `100 KB` (≈100k chars de texto extraído).
- **Latencia** = wall-clock del turno completo (lo que percibe el cliente), no la suma de latencias LLM.
- **Coste/tokens** = agregado de todas las llamadas LLM del turno (actor + extractor de metadata).
- **Memoria (drift)** = match literal case-insensitive del `fact_to_remember` sobre la memoria persistente del snapshot (`running_summary` ∪ `anchored_facts` ∪ `project_metadata`), no sobre el output del turno.

## Caveats de honestidad (leer antes de las cifras)

1. **El caché está dormido.** El pipeline conversacional no consulta caché (el exact-match/semántico era del flujo single-shot eliminado en pre-S05). Por tanto `cache_hit_kind` es siempre `"none"` y las columnas de hit rate son 0% **por construcción, no por degradación**. Se incluyen por fidelidad al formato. Consecuencia de diseño relevante: el CAG **paga el contexto completo en cada turno**, lo que hace su curva de coste más empinada que una implementación con caché.

2. **El "single provider" no se cumplió en la práctica.** El Router del wrapper usa `routing_strategy="simple-shuffle"` con primary (`anthropic/claude-haiku-4-5-20251001`) y fallback (`openai/gpt-4o-mini`) bajo el **mismo alias**, así que reparte llamadas entre ambos modelos en cada turno (no solo ante error). Es comportamiento heredado de session-05; no se tocó (la instrumentación es aditiva).

3. **El coste medido es un piso.** Durante la corrida, LiteLLM reportó el fallback como `gpt-4o-mini-2024-07-18` (con sufijo de fecha), que no estaba en `MODEL_COSTS` → esas llamadas se tarifaron a $0 (74 eventos `pricing_model_unknown`). Por eso varios turnos de `growing` muestran `cost_usd=0` pese a tener tokens. `app/core/pricing.py` ya incluye la variante para futuras corridas; el coste real de esta corrida es algo mayor que el reportado (gpt-4o-mini es ~7x más barato que Haiku, así que el sesgo es moderado). **La forma de las curvas no se ve afectada**: `tokens_in` se usa como señal primaria de crecimiento.

## Tabla resumen

| Escenario | Adjunto | Turnos | Fallos | P50 lat (ms) | P95 lat (ms) | Coste (USD) | Exact hit | Sem. hit | Recall medio |
|---|---|---|---|---|---|---|---|---|---|
| growing | 0 KB | 20 | 0 | 20 754 | 77 534 | 0.2700 | 0% | 0% | 100% |
| growing | 100 KB | 20 | 2 | 27 617 | 65 500 | 0.5547* | 0% | 0% | 100% |
| pivot | 0 KB | 8 | 0 | 11 210 | 15 858 | 0.0301* | 0% | 0% | 100% |
| pivot | 100 KB | 8 | 0 | 27 700 | 34 617 | 0.1010* | 0% | 0% | 100% |
| contradiction | 0 KB | 8 | 1 | 19 588 | 23 744 | 0.1162* | 0% | 0% | 86% |
| contradiction | 100 KB | 8 | 0 | 18 300 | 27 404 | 0.1136 | 0% | 0% | 88% |

\* coste subestimado por el gap de pricing (caveat 3). Recall sobre turnos exitosos.

## Curva 1 — latencia vs `tokens_in`

La latencia crece monótonamente con el tamaño del contexto. Es la señal más fuerte del baseline.

| `tokens_in` (bucket) | n | tokens medios | latencia media (ms) |
|---|---|---|---|
| < 5 000 | 6 | 4 178 | 9 746 |
| 5 000 – 15 000 | 23 | 9 154 | 20 858 |
| 15 000 – 40 000 | 37 | 27 939 | 31 497 |
| > 40 000 | 3 | 67 474 | 80 583 |

De ~4k a ~67k tokens de entrada, la latencia media del turno se multiplica por **~8.3x** (9.7s → 80.6s). Latencia mín/mediana/máx (turnos OK): **3 982 / 20 167 / 170 014 ms**.

## Curva 2 — coste y contexto acumulados vs turno (`growing`, 0 KB)

Como `cost_usd` tiene ceros por el gap de pricing, se acompaña de `tokens_in` (señal limpia).

| turn_index | `tokens_in` | coste turno (USD) | coste acumulado (USD) |
|---|---|---|---|
| 1 | 3 954 | 0.00218 | 0.00218 |
| 5 | 5 946 | 0.00000† | 0.02896 |
| 10 | 11 613 | 0.01882 | 0.07594 |
| 12 | 16 942 | 0.00000† | 0.07594 |
| 15 | 15 094 | 0.02794 | 0.14598 |
| 20 | 13 824 | 0.02609 | 0.27002 |

† turno servido íntegramente por el fallback no tarifado (caveat 3).

El coste por turno pasa de **$0.00218** (t1) a **$0.02609** (t20): un factor **~12x**. El acumulado de 20 turnos es **$0.27** (solo el actor+extractor, sin adjunto). `tokens_in` crece de ~3 950 a un pico de ~16 940 (t12), ~4.3x, porque cada turno reinyecta el historial completo.

## Curva 3 — recall (`memory_drift`) vs N

| N (turnos) | `growing` recall ("Nimbus") | `contradiction` recall |
|---|---|---|
| 1 | 100% | 100% |
| 5 | 100% | 100% |
| 10 | 100% | — |
| 15 | 100% | — |
| 20 | 100% | — |

En `growing`, el nombre del proyecto **sobrevive al 100% hasta N=20**: la memoria persistente del CAG (`project_metadata`) no se degrada con la longitud. En `contradiction` el recall medio baja a ~86–88%, pero por un motivo distinto: el fact cambia a propósito (presupuesto 30000 → 80000), así que parte del "drift" es la actualización esperada, no olvido.

## Fallos duros (500)

| Escenario | Adjunto | Turno | Latencia antes del fallo | Causa |
|---|---|---|---|---|
| contradiction | 0 KB | 7 | 47 300 ms | Instructor agota reintentos (validación) |
| growing | 100 KB | 18 | 93 563 ms | Validación + contexto saturado |
| growing | 100 KB | 19 | 134 543 ms | Validación + contexto saturado |

Los 500 ocurren cuando el modelo no produce un `EstimationResult` coherente (suma de fases ≠ total) y Instructor agota sus 3 reintentos (incluido el fallback). Aparecen bajo contexto alto y/o adjunto grande, con latencias de 47–135s antes de fallar.

## Lectura

**Párrafo 1 — dónde y cómo se rompe el CAG.** La degradación **no** viene de la memoria: el recall del `project_name` se mantiene al **100% hasta N=20** porque sobrevive en `project_metadata`, que se reinyecta cada turno. La dimensión dominante es la **latencia**: el **99% de los turnos (68/69) incumple un SLA de 4s**, con P50 de 20.2s, P95 de 65.5s y un máximo de **170s**; y crece de forma monótona con el contexto (×8.3 entre <5k y >40k tokens de entrada). El segundo modo de ruptura son los **fallos duros (500)**: aparecen a partir de contextos grandes —`contradiction` t7, y `growing`+100 KB t18–t19 con 94–135s antes de agotar reintentos—, cuando el modelo de tier bajo deja de cuadrar la aritmética de las fases con un prompt saturado. El coste crece (≈×12 por turno entre t1 y t20) pero en absoluto sigue siendo bajo ($0.27 por 20 turnos sin adjunto), así que **no** es el cuello de botella.

**Párrafo 2 — qué justifica saltar a RAG.** La raíz de los tres síntomas (latencia, coste, fallos de validación) es la misma: **el CAG reinyecta todo el historial en cada turno** y `tokens_in` crece sin techo (de ~3 950 a >16 900 tokens en `growing`, y de golpe a ~67k con un adjunto de 100 KB). RAG —recuperar solo los fragmentos relevantes en vez de arrastrar la conversación entera— acotaría `tokens_in` por turno y, con él, aplanaría la curva de latencia, contendría el coste y reduciría los fallos por saturación de contexto. El **caso límite** que justifica el salto es nítido en los datos: turnos con historial largo **y** adjunto grande (`growing`+100 KB, t18–t19), donde la latencia llega a 93–135s y el turno directamente **falla con 500**. A partir de ~15–40k tokens de entrada el CAG ya está fuera de cualquier SLA interactivo; ese es el punto donde el baseline deja de ser aceptable y RAG pasa de "optimización" a "necesidad".
