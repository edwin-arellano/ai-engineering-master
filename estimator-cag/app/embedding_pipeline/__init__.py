"""Pipeline de embeddings y chunking (S07, pre-ejercicio).

Módulo standalone: chunker estructural de presupuestos JSON, embedder vía
LiteLLM y endpoint ``POST /embeddings/ingest``. No se cablea a ``app/ingest/``
ni al contrato ``Document`` de S06; usa su propio ``Budget`` anidado. Sin
persistencia ni retrieval (eso llega en S08+).
"""
