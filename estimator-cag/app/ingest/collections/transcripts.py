"""Ingesta de la colección `transcript_chunks` (S10): transcripciones de reunión con
el cliente. Loader desde examples/transcripts/, chunker de texto por bloques temáticos
(recursive_split) e ingesta idempotente por source_path. chunk_type=transcript_segment.
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
from app.generation.rag.persistence.models import TranscriptChunkRow
from app.generation.rag.persistence.repository import (
    TRANSCRIPT_SEGMENT,
    get_document_id_by_source_path,
    ingest_document,
)
from app.generation.rag.schemas import Chunk

logger = structlog.get_logger(__name__)

TRANSCRIPTS_DIR = "examples/transcripts"
# Año de 4 dígitos en el nombre del fichero, si lo hubiera (p.ej. 2024_reunion.txt).
_YEAR_IN_NAME = re.compile(r"(20\d{2})")


def load_transcripts(
    directory: str = TRANSCRIPTS_DIR,
) -> list[tuple[str, str, dict]]:
    """Lee los `*.txt` del directorio (excluye los `*.out.txt`, que son trazas).
    Devuelve (source_path, text, meta). meta mínima: año (parseado del nombre o el
    actual) y source (filename)."""
    base = Path(directory)
    out: list[tuple[str, str, dict]] = []
    for path in sorted(base.glob("*.txt")):
        if path.name.endswith(".out.txt"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        match = _YEAR_IN_NAME.search(path.stem)
        year = int(match.group(1)) if match else date.today().year
        source_path = f"transcripts/{path.name}"
        out.append((source_path, text, {"year": year, "source": path.name}))
    return out


def chunk_transcript(text: str, *, source_path: str, meta: dict) -> list[Chunk]:
    """Trocea una transcripción en segmentos temáticos (recursive_split por párrafos).
    Los huérfanos (demasiado cortos) se marcan pero no se filtran aquí (el ingestor los
    descarta)."""
    settings = get_settings()
    source = meta.get("source", Path(source_path).name)
    segments = recursive_split(text, max_tokens=settings.chunk_max_tokens)
    chunks: list[Chunk] = []
    for index, segment in enumerate(segments):
        tokens = count_tokens(segment)
        chunks.append(
            Chunk(
                chunk_id=f"{source}::seg-{index}",
                text=segment,
                metadata={
                    **meta,
                    "chunk_id": f"{source}::seg-{index}",
                    "strategy": "transcript_segment",
                },
                token_count=tokens,
                is_orphan=is_orphan(tokens),
            )
        )
    return chunks


async def ingest_transcripts(
    *,
    embedder: LiteLLMEmbedder | None = None,
    session_factory: async_sessionmaker,
    directory: str = TRANSCRIPTS_DIR,
) -> dict:
    """Ingesta idempotente (por source_path) de todas las transcripciones a
    transcript_chunks. Devuelve conteos {documents, chunks, skipped}."""
    embedder = embedder or LiteLLMEmbedder()
    documents = chunks_total = skipped = 0
    for source_path, text, meta in load_transcripts(directory):
        async with session_factory() as session:
            if await get_document_id_by_source_path(session, source_path) is not None:
                skipped += 1
                logger.info("transcripts.skip_existing", source_path=source_path)
                continue
            chunks = [c for c in chunk_transcript(text, source_path=source_path, meta=meta) if not c.is_orphan]
            if not chunks:
                continue
            embedded = embedder.embed_many(chunks)
            _, created = await ingest_document(
                session,
                model=TranscriptChunkRow,
                source_path=source_path,
                document_type="meeting_transcript",
                document_metadata={**meta, "collection": "transcripts"},
                embedded_chunks=embedded,
                chunk_type=TRANSCRIPT_SEGMENT,
            )
            documents += 1
            chunks_total += created
            logger.info("transcripts.ingested", source_path=source_path, chunks=created)
    return {"documents": documents, "chunks": chunks_total, "skipped": skipped}
