"""Ingesta de las colecciones particionadas no-budget (S10): transcripts y
technical_docs. Cada módulo expone loader + chunker de texto + función de ingesta
idempotente que puebla su tabla vía repository.ingest_document(model=...)."""
