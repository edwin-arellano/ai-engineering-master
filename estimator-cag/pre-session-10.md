# Pre-Sesión 10 — Búsqueda híbrida + reranking: medición y conclusiones

Pre-trabajo de la Sesión 10. Se amplía el pipeline de recuperación de `session-09`
(vectorial con filtros de metadata + `distance_threshold`) con:

1. **Rama léxica** full-text de PostgreSQL (`content_tsv`, config `'spanish'`, índice GIN).
2. **Fusión RRF** (`k=60`) de los rankings vectorial y léxico (solo posiciones).
3. **Reranking** cross-encoder `BAAI/bge-reranker-v2-m3` con patrón *recall-then-rerank*
   (recall amplio top-50 → cross-encoder → top-5).

Las cuatro configuraciones (A/B/C/D) son invocables por API (`search_mode`, `reranking`)
y se miden con un golden set anotado a mano (`scripts/golden_set.json`, 5 consultas,
relevancia binaria) vía el endpoint de debug `/api/v1/retrieve-debug` (retrieval puro,
sin pagar la generación LLM en cada run).

## Metodología

- **Métrica:** precision@5 sobre los `budget_id` recuperados (dedup por presupuesto).
- **Latencia:** mediana de `search_time_ms` del pipeline (retrieval puro: excluye
  reformulación y HTTP); se reporta también el wall-clock como referencia.
- **Medición en caliente:** 4 runs por consulta, se descarta la 1ª (calentamiento),
  mediana de las 3 restantes.
- **Corpus:** doble eje `budget_component` (32) + `historical_task` (32), 15 presupuestos
  del sample. **Importante:** el contenido del corpus está en **inglés**; las consultas del
  golden set, en **español**.

Se reportan **dos tablas**: con el filtro de sector de S09 (comportamiento de producción)
y sin él (ranking sobre todo el corpus), porque el filtro cambia radicalmente lo que la
métrica puede ver (ver conclusiones).

## Resultados

### Tabla 1 — con filtro de metadata (sector) de S09 [producción]

| Config | Búsqueda | Reranking | precision@5 | Latencia retrieval (ms) | Wall-clock (ms) |
|---|---|---|---|---|---|
| A | Vectorial | No | 0.533 | 222 | 1873 |
| B | Híbrida | No | 0.533 | 210 | 1722 |
| C | Vectorial | Sí | 0.533 | 632 | 2204 |
| D | Híbrida | Sí | 0.533 | 655 | 2102 |

### Tabla 2 — sin filtro de sector (ranking sobre todo el corpus)

| Config | Búsqueda | Reranking | precision@5 | Latencia retrieval (ms) | Wall-clock (ms) |
|---|---|---|---|---|---|
| A | Vectorial | No | 0.28 | 224 | 1796 |
| B | Híbrida | No | 0.28 | 235 | 1785 |
| C | Vectorial | Sí | 0.32 | 1309 | 2778 |
| D | Híbrida | Sí | 0.32 | 1314 | 2660 |

## Conclusiones

**1. En producción (Tabla 1), las 4 configuraciones empatan en precision@5 = 0.533, y no
es casualidad.** El filtro de sector de S09 (`filters_from_reformulation`) reduce el
retrieval a un único sector, y cada sector del corpus tiene ≤4 presupuestos. Como todos
caben en el top-5, **precision@5 se vuelve insensible al orden**: ni la fusión híbrida ni
el reranking pueden mejorar una métrica que solo mira "qué presupuestos entran", no "en qué
orden", cuando el conjunto entero ya entra. El reranking, en este escenario, **solo añade
coste** (~410 ms de latencia de retrieval, de 222 a 632 ms) sin ganancia alguna.

**2. El valor del reranking solo emerge cuando hay sitio para reordenar (Tabla 2).** Quitando
el filtro de sector, el pipeline compite sobre los 15 presupuestos y el orden sí importa:
el reranking sube precision@5 de **0.28 a 0.32** (+0.04, ~14% relativo). El caso más claro
es **q05** (gestión documental, dominio colindante): el vector puro entierra el único
relevante (BUD-2023-015) fuera del top-5 (precision 0.00), y el cross-encoder lo **rescata**
al top-5 (0.20). El cross-encoder, leyendo consulta y candidato juntos, ordena mejor que el
bi-encoder cuando el relevante está en el pool pero mal posicionado.

**3. La búsqueda híbrida apenas aporta en este corpus, por una razón concreta y solucionable.**
B ≈ A y D ≈ C: la rama léxica casi no mueve el ranking. La causa es el **desajuste cross-lingual**:
el corpus está en inglés ("telehealth", "cart", "appointment") y las consultas en español
("telemedicina", "carrito", "citas"). La config `'spanish'` del `tsvector` solo casa los
tecnicismos que coinciden literalmente (OAuth, PSD2, e-commerce); el resto del vocabulario
español no encuentra sus equivalentes ingleses, así que la rama léxica recupera poco y RRF
queda dominada por la vectorial. **Primer experimento para la sesión en vivo:** indexar el
contenido también con config inglesa (o normalizar la query léxica al inglés), o alimentar
la transcripción cruda en vez de `search_text` a la rama léxica.

**4. ¿El reranking justifica su latencia en este caso de uso?** Con los números delante:
**hoy no, en producción.** Mientras el filtro de sector pre-seleccione ≤5 presupuestos, el
reranking paga ~410 ms–1.1 s de retrieval por **cero** ganancia de precision@5 (Tabla 1).
Donde sí ganaría (+0.04, Tabla 2) el coste sube a ~5-6× la latencia de retrieval (de ~224 a
~1309 ms). Ahora bien, ese coste hay que mirarlo contra el **denominador correcto**: la
petición completa de producto incluye reformulación (~1 s) y generación LLM (varios segundos),
así que +1 s de rerank es ruido frente al total. La conclusión no es "el reranking es malo",
sino **"este corpus es demasiado pequeño y está demasiado pre-filtrado para que el reranking
tenga margen"**.

**Recomendación operativa:** mantener los defaults de producción en **vectorial / sin rerank**
(`RAG_SEARCH_MODE=vector`, `RERANKING_ENABLED=false`) mientras el corpus siga siendo pequeño y
filtrado por sector — es la opción barata y la métrica confirma que no se pierde nada. Activar
reranking (e híbrida bien calibrada) se vuelve rentable cuando: (a) crece el número de
presupuestos por sector, (b) se relaja el pre-filtro de sector, o (c) se sube `top_k` por
encima del tamaño del sector. El interruptor ya existe (parámetro de endpoint + Settings), así
que el cambio es de configuración, no de código. Antes de eso, la palanca de mayor ROI es
**arreglar el desajuste cross-lingual de la rama léxica**, no el reranking.
