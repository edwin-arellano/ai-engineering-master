"""Similitud coseno calculada a mano (sin numpy/sklearn)."""

from __future__ import annotations

import math

from scripts.compare import cosine_similarity


def test_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_identical_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_known_value():
    # [1,1] vs [1,0]: cos = 1 / (sqrt(2)*1) = 1/sqrt(2)
    assert math.isclose(cosine_similarity([1.0, 1.0], [1.0, 0.0]), 1 / math.sqrt(2))


def test_zero_norm_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
