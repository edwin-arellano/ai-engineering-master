# Indexación HNSW — reporte (session-08)

Corpus: **30 033 chunks** (33 reales `budget_component` + 30 000 sintéticos de stress),
1536 dims. Índice adoptado: `chunks_embedding_halfvec_idx` (HNSW, `halfvec_cosine_ops`,
m=16, ef_construction=128). Extensión `vector` 0.8.2. Medidas con la parte SQL aislada
del embedder (las queries se embeben una vez y se cronometra solo la función de repositorio).

## Baseline vs índice (mediana de la parte SQL, k=5, 10 runs × 5 queries = 50 runs)

| Modo             | Mediana global |
|------------------|----------------|
| exact (seq scan) | 122.36 ms      |
| indexed (HNSW)   | 4.50 ms        |

El índice HNSW half-vec recorta la latencia de BD **~27×** (122 → 4.5 ms). El seq scan
recorre los 30 033 vectores en cada query; el HNSW navega el grafo de proximidad.

## Punto dulce de `ef_search` (k=5, 5 runs × 4 queries)

| ef_search | recall@5 | p50 ms |
|-----------|----------|--------|
| 10        | 0.750    | 2.47   |
| 20        | 1.000    | 2.34   |
| 40        | 1.000    | 2.58   |  ← adoptado (`HNSW_EF_SEARCH=40`)
| 80        | 1.000    | 3.22   |
| 120       | 1.000    | 3.72   |
| 200       | 1.000    | 5.43   |

El recall satura en **1.0 ya en ef=20** para este snapshot de 30 k vectores. Se adopta
**40** (no 20) deliberadamente: da margen al crecimiento del corpus —el punto dulce sube
con el volumen: a 1 M vectores 20 puede quedarse corto— a un coste de latencia
despreciable (2.58 vs 2.34 ms). Por debajo de 20 (ef=10) el recall ya cae a 0.75.

## float32 vs half-vec (k=5, 5 runs × 3 queries)

| Índice   | Tamaño | recall@5 | p50 ms |
|----------|--------|----------|--------|
| float32  | 235 MB | 1.000    | 4.80   |
| half-vec | 117 MB | 1.000    | 3.44   |

Half-vec (float16, 2 bytes/dim) ocupa **exactamente la mitad** que float32 (235 → 117 MB)
**sin pérdida de recall** (1.000 en ambos) y con latencia igual o mejor. La precisión extra
de float32 es ruido para vectores normalizados de OpenAI: se paga en disco y RAM sin
ganar calidad. Por eso el índice adoptado y migrado es el half-vec; el float32 solo se
crea/dropea ad-hoc en `scripts/compare_index.py` para este demo.

## Conclusión

HNSW half-vec con `ef_search=40` da **recall ~1.0**, **la mitad de tamaño** que float32 y
latencia de BD (~4.5 ms) **~27× por debajo** del baseline seq scan (~122 ms). El operador
`<=>` está alineado con `halfvec_cosine_ops` y la expresión de la query (`embedding::halfvec(1536)`)
es idéntica a la indexada — verificado con `EXPLAIN ANALYZE`:

```
Index Scan using chunks_embedding_halfvec_idx on chunks
  Order By: ((embedding)::halfvec(1536) <=> $0)
Execution Time: ~5 ms
```

Desalinear el operador o la expresión haría que Postgres cayera a seq scan **sin avisar**;
por eso el `EXPLAIN ANALYZE` es la puerta de aceptación (test `test_hnsw_index_scan.py`),
no un extra.

> Nota: la latencia de `/search` end-to-end (~880 ms) está dominada por la llamada de
> embedding de la query (red), no por la BD. Por eso estas medidas aíslan la parte SQL.
