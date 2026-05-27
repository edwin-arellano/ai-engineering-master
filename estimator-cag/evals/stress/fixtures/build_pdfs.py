"""Genera PDFs sintéticos calibrados para el escenario de adjuntos grandes.

Los PDFs NO se comitean (ver .gitignore en esta carpeta). El runner los
regenera porque su contenido es determinístico. Los tamaños se calibran por
chars de texto embebido; el cap real del sistema es `attachment_max_bytes`
(bytes, default 10 MiB), así que estos fixtures quedan muy por debajo del
límite y nunca disparan el 413.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

FIXTURES_DIR = Path(__file__).parent
LOREM = (
    "Project specification. The system must integrate with SAP and includes a "
    "GDPR audit phase. Team of four engineers, one PM, one designer. "
)

# Aproximación: ~4 chars/token, ~1KB ≈ 1000 chars de texto plano embebido.
TARGETS_KB = {5: 5_000, 20: 20_000, 50: 50_000, 100: 100_000}


def build_pdf(target_chars: int, out_path: Path) -> None:
    """Escribe un PDF con `target_chars` de texto, paginando A4 cuando se llena."""
    pdf_canvas = canvas.Canvas(str(out_path), pagesize=A4)
    text = pdf_canvas.beginText(40, 800)
    written = 0
    while written < target_chars:
        text.textLine(LOREM)
        written += len(LOREM)
        if text.getY() < 40:
            pdf_canvas.drawText(text)
            pdf_canvas.showPage()
            text = pdf_canvas.beginText(40, 800)
    pdf_canvas.drawText(text)
    pdf_canvas.save()


def build_all() -> dict[int, Path]:
    """Genera todos los fixtures y devuelve {kb: ruta}."""
    paths: dict[int, Path] = {}
    for kb, chars in TARGETS_KB.items():
        out = FIXTURES_DIR / f"attach_{kb}kb.pdf"
        build_pdf(chars, out)
        paths[kb] = out
    return paths


if __name__ == "__main__":
    for kb, path in build_all().items():
        print(f"{kb}KB → {path} ({path.stat().st_size} bytes)")
