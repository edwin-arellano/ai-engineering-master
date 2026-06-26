"""Ingesta de la colección `technical_doc_chunks` (S10): documentación técnica de
referencia (corpus sembrado por scripts/seed_technical_docs.py). Loader desde
data/technical_docs/, chunker por secciones `##` e ingesta idempotente por source_path.
chunk_type=technical_reference.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.foundations.config import get_settings
from app.generation.rag.chunking.common import count_tokens, is_orphan, recursive_split
from app.generation.rag.embedding.embedder import LiteLLMEmbedder
from app.generation.rag.persistence.models import TechnicalDocChunkRow
from app.generation.rag.persistence.repository import (
    TECHNICAL_REFERENCE,
    get_document_id_by_source_path,
    ingest_document,
)
from app.generation.rag.schemas import Chunk

logger = structlog.get_logger(__name__)

TECHNICAL_DOCS_DIR = "data/technical_docs"
# Encabezado de sección de nivel 2 ("## Título") como separador de troceado temático.
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def load_technical_docs(
    directory: str = TECHNICAL_DOCS_DIR,
) -> list[tuple[str, str, dict]]:
    """Lee los `*.md` del directorio. Devuelve (source_path, text, meta). La tecnología
    se deriva del nombre del fichero (slug)."""
    base = Path(directory)
    out: list[tuple[str, str, dict]] = []
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        source_path = f"technical_docs/{path.name}"
        out.append(
            (source_path, text, {"technology": path.stem, "year": date.today().year})
        )
    return out


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Parte el markdown por encabezados `##`. Devuelve (titulo_seccion, cuerpo). El
    preámbulo previo a la primera sección se etiqueta como 'overview'."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [("overview", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("overview", preamble))
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        sections.append((title, body))
    return sections


def chunk_technical_doc(text: str, *, source_path: str, meta: dict) -> list[Chunk]:
    """Trocea un doc técnico por secciones `##`; secciones largas se subdividen con
    recursive_split. metadata lleva technology y section."""
    settings = get_settings()
    technology = meta.get("technology", Path(source_path).stem)
    chunks: list[Chunk] = []
    counter = 0
    for title, body in _split_sections(text):
        for piece in recursive_split(body, max_tokens=settings.chunk_max_tokens):
            tokens = count_tokens(piece)
            chunk_id = f"{technology}::{title}::{counter}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    metadata={
                        "technology": technology,
                        "section": title,
                        "year": meta.get("year", date.today().year),
                        "chunk_id": chunk_id,
                        "strategy": "technical_reference",
                    },
                    token_count=tokens,
                    is_orphan=is_orphan(tokens),
                )
            )
            counter += 1
    return chunks


async def ingest_technical_docs(
    *,
    embedder: LiteLLMEmbedder | None = None,
    session_factory: async_sessionmaker,
    directory: str = TECHNICAL_DOCS_DIR,
) -> dict:
    """Ingesta idempotente (por source_path) de los docs técnicos a
    technical_doc_chunks. Devuelve conteos {documents, chunks, skipped}."""
    embedder = embedder or LiteLLMEmbedder()
    documents = chunks_total = skipped = 0
    for source_path, text, meta in load_technical_docs(directory):
        async with session_factory() as session:
            if await get_document_id_by_source_path(session, source_path) is not None:
                skipped += 1
                logger.info("technical_docs.skip_existing", source_path=source_path)
                continue
            chunks = [
                c
                for c in chunk_technical_doc(text, source_path=source_path, meta=meta)
                if not c.is_orphan
            ]
            if not chunks:
                continue
            embedded = embedder.embed_many(chunks)
            _, created = await ingest_document(
                session,
                model=TechnicalDocChunkRow,
                source_path=source_path,
                document_type="technical_reference",
                document_metadata={**meta, "collection": "technical_docs"},
                embedded_chunks=embedded,
                chunk_type=TECHNICAL_REFERENCE,
            )
            documents += 1
            chunks_total += created
            logger.info("technical_docs.ingested", source_path=source_path, chunks=created)
    return {"documents": documents, "chunks": chunks_total, "skipped": skipped}
