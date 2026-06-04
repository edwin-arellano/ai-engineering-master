"""Extracción local de texto desde PDFs y archivos Word.

Camino B del ejercicio: el servicio IA lee el contenido del adjunto y envía
solo texto al LLM. Esto preserva la portabilidad del wrapper LiteLLM/Instructor
(no nos atamos a la Files API de Anthropic ni a la de OpenAI) y prepara el
terreno para el chunking de RAG del módulo 3, donde esta lógica de extracción
es la primera pieza del pipeline.

Tipos soportados:

- ``application/pdf`` → ``pypdf``
- ``application/vnd.openxmlformats-officedocument.wordprocessingml.document``
  (.docx) → ``python-docx``

Cualquier otro tipo se rechaza con ``UnsupportedAttachmentError``, que el
router traduce a HTTP 415.
"""

from __future__ import annotations

from io import BytesIO
from typing import NamedTuple

import structlog
from docx import Document as DocxDocument
from fastapi import UploadFile
from pypdf import PdfReader

logger = structlog.get_logger(__name__)

PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
SUPPORTED_MIMES = {PDF_MIME, DOCX_MIME}


class UnsupportedAttachmentError(ValueError):
    """Tipo MIME no soportado."""


class AttachmentTooLargeError(ValueError):
    """El adjunto excede ``ATTACHMENT_MAX_BYTES``."""


class ExtractedAttachment(NamedTuple):
    """Resultado de extraer texto de un adjunto."""

    filename: str
    mime_type: str
    text: str
    byte_size: int


def _extract_pdf(data: bytes) -> str:
    """Devuelve el texto plano del PDF con marcadores de página."""
    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(f"--- Page {index} ---\n{page_text}")
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    """Devuelve el texto plano del .docx, párrafo a párrafo."""
    document = DocxDocument(BytesIO(data))
    return "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text and paragraph.text.strip()
    )


async def extract_attachment(
    upload: UploadFile, max_bytes: int
) -> ExtractedAttachment:
    """Lee el ``UploadFile`` y extrae texto según su tipo MIME."""
    mime_type = upload.content_type or ""
    if mime_type not in SUPPORTED_MIMES:
        raise UnsupportedAttachmentError(
            f"Tipo MIME no soportado: {mime_type!r} (archivo {upload.filename!r})"
        )

    data = await upload.read()
    if len(data) > max_bytes:
        raise AttachmentTooLargeError(
            f"{upload.filename!r} pesa {len(data)} bytes, "
            f"máximo permitido {max_bytes}"
        )

    if mime_type == PDF_MIME:
        text = _extract_pdf(data)
    else:
        text = _extract_docx(data)

    logger.info(
        "attachment_extracted",
        filename=upload.filename,
        mime_type=mime_type,
        byte_size=len(data),
        extracted_chars=len(text),
    )
    return ExtractedAttachment(
        filename=upload.filename or "unnamed",
        mime_type=mime_type,
        text=text,
        byte_size=len(data),
    )


def build_attachments_block(attachments: list[ExtractedAttachment]) -> str:
    """Concatena los textos extraídos con delimitadores claros para el prompt."""
    if not attachments:
        return ""
    blocks = [
        f'<attachment filename="{a.filename}">\n{a.text}\n</attachment>'
        for a in attachments
    ]
    return "\n\n".join(blocks)
