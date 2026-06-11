-- Queries de operación/mantenimiento de la base vectorial. Ejecutar en psql:
--   docker compose exec -T postgres psql -U estimator -d estimator -f /ruta/monitoring.sql
-- (o copiar/pegar las que interesen). Las lecturas (1-4, 6) son baratas; las de
-- mantenimiento (5) hacen I/O o reconstruyen índices: ventanas de bajo tráfico.

-- 1. Salud de la tabla: filas vivas vs muertas, último ANALYZE/autovacuum/VACUUM.
--    n_dead_tup alto → el planner trabaja con estadísticas viejas y el índice degrada.
SELECT relname, n_live_tup, n_dead_tup,
       last_analyze, last_autoanalyze, last_vacuum, last_autovacuum
FROM pg_stat_user_tables
WHERE relname IN ('chunks', 'documents');

-- 2. Tamaño de los índices (el HNSW half-vec domina el espacio de la tabla).
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY pg_relation_size(indexrelid) DESC;

-- 3. Conteo de chunks por tipo (budget_component reales vs synthetic de stress).
SELECT chunk_type, count(*) FROM chunks GROUP BY chunk_type ORDER BY count(*) DESC;

-- 4. Uso de cada índice (scans). idx_scan = 0 en el HNSW → la query no lo está usando
--    (operador/expresión desalineados): revisar repository._build_halfvec_search_stmt.
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE relname = 'chunks';

-- 5. Mantenimiento (NO en horario punta).
--    ANALYZE: actualiza estadísticas del planner. Rápido, no bloquea.
ANALYZE chunks;
--    VACUUM: recupera espacio de filas borradas/actualizadas. No bloquea pero hace I/O.
VACUUM chunks;
--    VACUUM ANALYZE combinado.
VACUUM ANALYZE chunks;
--    REINDEX concurrente: reconstruye el índice en paralelo, swap atómico, tabla accesible.
--    Coste alto; ventana de bajo tráfico.
REINDEX INDEX CONCURRENTLY chunks_embedding_halfvec_idx;

-- 6. Verificar que el índice sigue válido tras un swap/reindex.
SELECT indexrelid::regclass AS index, indisvalid, indisready
FROM pg_index
WHERE indexrelid = 'chunks_embedding_halfvec_idx'::regclass;

-- =============================================================================
-- RUNBOOK DE PRODUCCIÓN (documentación; NO se ejecuta desde una migración Alembic
-- porque CONCURRENTLY no puede correr dentro de una transacción).
--
-- Reconstrucción / creación de índice sin downtime, en ventana de bajo tráfico:
--
--   SET maintenance_work_mem = '512MB';          -- acelera la construcción del grafo HNSW
--   SET max_parallel_maintenance_workers = 2;    -- paraleliza la construcción
--   CREATE INDEX CONCURRENTLY chunks_embedding_halfvec_idx
--     ON chunks USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
--     WITH (m = 16, ef_construction = 128);
--   -- o, si el índice ya existe y se degradó:
--   REINDEX INDEX CONCURRENTLY chunks_embedding_halfvec_idx;
--
-- Degradación a largo plazo: con churn alto (muchos INSERT/UPDATE/DELETE) el grafo HNSW
-- pierde calidad de vecindad y crecen las dead tuples. Disciplina: autovacuum afinado +
-- REINDEX CONCURRENTLY periódico. Garbage in → garbage out: vectores mal normalizados o
-- de modelos distintos contaminan el espacio y bajan el recall aunque el índice esté sano.
-- =============================================================================
