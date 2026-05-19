"""Tests de la extracción de adjuntos."""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document as DocxDocument
from fastapi import UploadFile
from pypdf import PdfWriter
from starlette.datastructures import Headers

from app.services.attachments import (
    DOCX_MIME,
    PDF_MIME,
    AttachmentTooLargeError,
    UnsupportedAttachmentError,
    build_attachments_block,
    extract_attachment,
)


def _make_upload(data: bytes, filename: str, content_type: str) -> UploadFile:
    """Construye un UploadFile con el content-type correcto en las headers."""
    return UploadFile(
        file=BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def _make_minimal_pdf_bytes() -> bytes:
    """Crea un PDF mínimo válido (1 página vacía) para test."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    """Crea un .docx con los párrafos indicados."""
    document = DocxDocument()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


async def test_extract_pdf_succeeds() -> None:
    upload = _make_upload(_make_minimal_pdf_bytes(), "spec.pdf", PDF_MIME)

    result = await extract_attachment(upload, max_bytes=1024 * 1024)
    assert result.mime_type == PDF_MIME
    assert result.filename == "spec.pdf"


async def test_extract_docx_succeeds() -> None:
    docx_bytes = _make_docx_bytes(["Hello world", "Second paragraph"])
    upload = _make_upload(docx_bytes, "notes.docx", DOCX_MIME)

    result = await extract_attachment(upload, max_bytes=1024 * 1024)
    assert result.mime_type == DOCX_MIME
    assert "Hello world" in result.text
    assert "Second paragraph" in result.text


async def test_extract_rejects_unsupported_mime() -> None:
    upload = _make_upload(b"\x89PNG\r\n...", "image.png", "image/png")

    with pytest.raises(UnsupportedAttachmentError):
        await extract_attachment(upload, max_bytes=1024 * 1024)


async def test_extract_rejects_oversized_file() -> None:
    big_bytes = _make_docx_bytes(["x" * 100])
    upload = _make_upload(big_bytes, "big.docx", DOCX_MIME)

    with pytest.raises(AttachmentTooLargeError):
        await extract_attachment(upload, max_bytes=10)


def test_build_attachments_block_empty() -> None:
    assert build_attachments_block([]) == ""


def test_build_attachments_block_concatenates_with_delimiters() -> None:
    from app.services.attachments import ExtractedAttachment

    a = ExtractedAttachment("spec.pdf", PDF_MIME, "Page 1 text", 123)
    b = ExtractedAttachment("notes.docx", DOCX_MIME, "Paragraph text", 456)
    block = build_attachments_block([a, b])
    assert '<attachment filename="spec.pdf">' in block
    assert '<attachment filename="notes.docx">' in block
    assert "Page 1 text" in block
    assert "Paragraph text" in block
