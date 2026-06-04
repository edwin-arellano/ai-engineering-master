"""Compresión de historial: anclas + resumen acumulativo."""

from app.generation.cag.sessions.compression.anchors import detect_anchors
from app.generation.cag.sessions.compression.policy import apply_compression
from app.generation.cag.sessions.compression.summarizer import Summarizer

__all__ = ["detect_anchors", "apply_compression", "Summarizer"]
