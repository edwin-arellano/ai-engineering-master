"""Augmentation de calidad (S11): reorden por extremos + compresión extractiva. Sin LLM."""

from __future__ import annotations

from app.generation.rag.retrieval.augmentation import extract_keypoints, reorder_by_edges


class _Chunk:
    """Doble mínimo con el único atributo que reorder_by_edges usa (distance)."""

    def __init__(self, name: str, distance: float) -> None:
        self.name = name
        self.distance = distance


def test_reorder_puts_best_first_and_second_best_last():
    # distance menor = mejor. Entrada desordenada a propósito.
    chunks = [
        _Chunk("c2", 0.30),
        _Chunk("c0", 0.10),  # mejor
        _Chunk("c4", 0.50),
        _Chunk("c1", 0.20),  # 2º mejor
        _Chunk("c3", 0.40),
    ]
    ordered = reorder_by_edges(chunks)
    names = [c.name for c in ordered]
    assert names[0] == "c0"  # el más fuerte al principio
    assert names[-1] == "c1"  # el 2º más fuerte al final
    # Los débiles quedan hacia el centro.
    assert names == ["c0", "c2", "c4", "c3", "c1"]


def test_extract_keypoints_reduces_and_keeps_figures():
    content = (
        "Este es un párrafo de relleno con mucha palabrería introductoria que no aporta.\n"
        "El componente de autenticación se estima en 120 horas.\n"
        "Consideraciones generales y texto de relleno sin ninguna señal relevante aquí.\n"
        "El módulo de pagos se estima en 90 horas."
    )
    out = extract_keypoints(content, max_chars=200)
    assert len(out) < len(content)  # comprime
    assert "120" in out and "90" in out  # conserva las cifras
    assert "relleno con mucha palabrería" not in out  # descarta el ruido sin señal


def test_extract_keypoints_returns_short_content_unchanged():
    short = "120 horas de auth"
    assert extract_keypoints(short, max_chars=600) == short
