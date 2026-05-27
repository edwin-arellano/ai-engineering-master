# Stress test del CAG — REPORT

> Baseline cuantitativo del CAG del `estimator` previo a la sesión 6.
> Corrida real contra el endpoint HTTP (`--http`), modo `actor`.
> 72 turnos (71 OK, 1 fallo), coste medido = **$1.46 USD** (ambos modelos tarifados).

## Metodología

- **Runner**: `uv run python -m evals.stress.run --http http://127.0.0.1:8000 --scenarios growing,pivot,contradiction --attachment-sizes 0,100 --repeats 1`.
- **Escenarios**: `growing` (20 turnos, requisitos coherentes acumulándose), `pivot` (8 turnos, cambio de stack en t5), `contradiction` (8 turnos, cambio de presupuesto en t8). Cada uno con adjunto `0 KB` y `100 KB` (≈100k chars de texto extraído).
- **Latencia** = wall-clock del turno completo (lo que percibe el cliente), no la suma de latencias LLM.
- **Coste/tokens** = agregado de todas las llamadas LLM del turno (actor + extractor de metadata).
- **Memoria (drift)** = match literal case-insensitive del `fact_to_remember` sobre la memoria persistente del snapshot (`running_summary` ∪ `anchored_facts` ∪ `project_metadata`), no sobre el output del turno.

## Caveats de honestidad (leer antes de las cifras)

1. **El caché está dormido.** El pipeline conversacional no consulta caché (el exact-match/semántico era del flujo single-shot eliminado en pre-S05). Por tanto `cache_hit_kind` es siempre `"none"` y las columnas de hit rate son 0% **por construcción, no por degradación**. Se incluyen por fidelidad al formato. Consecuencia de diseño relevante: el CAG **paga el contexto completo en cada turno**, lo que hace su curva de coste más empinada que una implementación con caché.

2. **El "single provider" no se cumple en la práctica.** El Router del wrapper usa `routing_strategy="simple-shuffle"` con primary (`anthropic/claude-haiku-4-5-20251001`) y fallback (`openai/gpt-4o-mini`) bajo el **mismo alias**, así que reparte llamadas entre ambos modelos en cada turno (no solo ante error). Es comportamiento heredado de session-05; no se tocó (la instrumentación es aditiva). Las curvas, por tanto, reflejan una mezcla de ambos modelos, no Haiku puro.

3. **Coste real, ambos modelos tarifados.** Esta corrida se ejecutó con `app/core/pricing.py` ya corregido: la variante `gpt-4o-mini-2024-07-18` (con la que el Router responde buena parte de las llamadas por simple-shuffle) tiene tarifa, así que **ningún turno OK reporta coste 0** y el total ($1.46) es real. Los precios siguen siendo placeholders a verificar contra las tarifas oficiales; lo que el ejercicio lee es la **forma** de la curva, correcta independientemente del precio absoluto.

## Tabla resumen

| Escenario | Adjunto | Turnos | Fallos | P50 lat (ms) | P95 lat (ms) | Coste (USD) | Exact hit | Sem. hit | Recall medio |
|---|---|---|---|---|---|---|---|---|---|
| growing | 0 KB | 20 | 0 | 23 658 | 96 868 | 0.2669 | 0% | 0% | 100% |
| growing | 100 KB | 20 | 0 | 24 786 | 69 614 | 0.6086 | 0% | 0% | 100% |
| pivot | 0 KB | 8 | 0 | 7 106 | 13 709 | 0.0490 | 0% | 0% | 100% |
| pivot | 100 KB | 8 | 0 | 19 303 | 23 394 | 0.1879 | 0% | 0% | 100% |
| contradiction | 0 KB | 8 | 0 | 7 929 | 20 240 | 0.1097 | 0% | 0% | 88% |
| contradiction | 100 KB | 8 | 1 | 16 170 | 30 232 | 0.2341 | 0% | 0% | 86% |

Recall sobre turnos exitosos. Coste total de la corrida: **$1.46**.

## Curva 1 — latencia vs `tokens_in`

La latencia crece con el tamaño del contexto, aunque con ruido considerable en la franja media (el Router reparte entre dos modelos y la latencia de cada uno varía).

| `tokens_in` (bucket) | n | tokens medios | latencia media (ms) |
|---|---|---|---|
| < 5 000 | 5 | 3 779 | 15 406 |
| 5 000 – 15 000 | 28 | 9 086 | 23 938 |
| 15 000 – 40 000 | 35 | 29 064 | 25 136 |
| > 40 000 | 3 | 77 398 | 50 518 |

De ~3.8k a ~77k tokens de entrada, la latencia media del turno se multiplica por **~3.3x** (15.4s → 50.5s). Latencia mín/mediana/máx (turnos OK): **3 884 / 19 144 / 108 366 ms**.

## Curva 2 — coste y contexto acumulados vs turno (`growing`, 0 KB)

| turn_index | `tokens_in` | coste turno (USD) | coste acumulado (USD) |
|---|---|---|---|
| 1 | 5 360 | 0.01048 | 0.01048 |
| 5 | 14 582 | 0.02121 | 0.04802 |
| 10 | 9 652 | 0.00569 | 0.07101 |
| 12 | 10 734 | 0.01842 | 0.10848 |
| 15 | 14 068 | 0.02018 | 0.15263 |
| 20 | 13 159 | 0.02520 | 0.26690 |

El coste por turno pasa de **$0.01048** (t1) a **$0.02520** (t20): un factor **~2.4x**. El acumulado de 20 turnos es **$0.267** (solo actor+extractor, sin adjunto); con adjunto de 100 KB sube a **$0.61**. El coste por turno fluctúa (el reparto de modelos del Router introduce varianza), pero la tendencia acumulada es claramente creciente porque cada turno reinyecta el historial completo.

## Curva 3 — recall (`memory_drift`) vs N

| N (turnos) | `growing` recall ("Nimbus") | `contradiction` recall |
|---|---|---|
| 1 | 100% | 100% |
| 5 | 100% | 100% |
| 10 | 100% | — |
| 15 | 100% | — |
| 20 | 100% | — |

En `growing`, el nombre del proyecto **sobrevive al 100% hasta N=20**, con y sin adjunto (20/20 en ambos): la memoria persistente del CAG (`project_metadata`) no se degrada con la longitud. En `contradiction` el recall medio baja a ~86–88%, pero por un motivo distinto: el fact cambia a propósito (presupuesto 30000 → 80000), así que parte del "drift" es la actualización esperada, no olvido.

## Fallos duros (500)

| Escenario | Adjunto | Turno | Latencia antes del fallo | Causa |
|---|---|---|---|---|
| contradiction | 100 KB | 7 | 52 488 ms | Instructor agota reintentos (validación) bajo contexto + adjunto |

El 500 ocurre cuando el modelo no produce un `EstimationResult` coherente (suma de fases ≠ total) y Instructor agota sus 3 reintentos (incluido el fallback). Aparece bajo contexto alto y/o adjunto grande, tras ~52s de reintentos. La tasa de fallo es baja (1/72 ≈ 1.4%) pero correlaciona con la saturación de contexto: la corrida previa, con los mismos parámetros, produjo 3 fallos, todos en turnos altos o con adjunto de 100 KB.

## Lectura

**Párrafo 1 — dónde y cómo se rompe el CAG.** La degradación **no** viene de la memoria: el recall del `project_name` se mantiene al **100% hasta N=20** (con y sin adjunto) porque sobrevive en `project_metadata`, que se reinyecta cada turno. La dimensión dominante es la **latencia**: el **99% de los turnos (70/71) incumple un SLA de 4s**, con P50 de 19.1s, P95 de 62.9s y un máximo de **108s**, y crece con el contexto (×3.3 entre <5k y >40k tokens de entrada, con un P95 de 97s en `growing` sin adjunto). El segundo modo de ruptura son los **fallos duros (500)**: aparecen bajo contexto saturado —`contradiction`+100 KB t7, con 52s antes de agotar reintentos— cuando el modelo de tier bajo deja de cuadrar la aritmética de las fases con un prompt grande. El coste crece (≈×2.4 por turno entre t1 y t20) pero en absoluto sigue siendo bajo ($0.27 por 20 turnos sin adjunto, $0.61 con adjunto de 100 KB), así que **no** es el cuello de botella.

**Párrafo 2 — qué justifica saltar a RAG.** La raíz de los tres síntomas (latencia, coste, fallos de validación) es la misma: **el CAG reinyecta todo el historial en cada turno** y `tokens_in` crece sin techo (de ~5.4k a ~14k tokens en `growing`, y de golpe a ~77k con un adjunto de 100 KB). RAG —recuperar solo los fragmentos relevantes en vez de arrastrar la conversación entera— acotaría `tokens_in` por turno y, con él, aplanaría la curva de latencia, contendría el coste y reduciría los fallos por saturación de contexto. El **caso límite** que justifica el salto es nítido en los datos: turnos con contexto alto **y** adjunto grande (`contradiction`+100 KB t7), donde la latencia llega a 52s y el turno directamente **falla con 500**; y el P95 de 97s en `growing` sin adjunto a N alto. A partir de ~15–40k tokens de entrada el CAG ya está fuera de cualquier SLA interactivo; ese es el punto donde el baseline deja de ser aceptable y RAG pasa de "optimización" a "necesidad".
