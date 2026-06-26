"""QueryRouter — cascada de routing. Unit (sin LLM): niveles explícito, determinista y
fallback. Integration (@pytest.mark.integration): el nivel LLM con modelo real."""

from __future__ import annotations

import pytest

from app.foundations.config import get_settings
from app.generation.rag.retrieval.router import QueryRouter
from app.generation.rag.schemas import SearchTarget


class _RaisingWrapper:
    """Wrapper falso: simula un fallo del clasificador LLM para forzar el fallback."""

    def complete_structured(self, **_kwargs):
        raise RuntimeError("LLM down")


def _router(wrapper=None) -> QueryRouter:
    return QueryRouter(wrapper=wrapper, settings=get_settings())


def test_explicit_level_skips_classifier():
    decision, level = _router().route(
        "lo que sea", explicit=[SearchTarget.BUDGETS]
    )
    assert level == "explicit"
    assert decision.targets == [SearchTarget.BUDGETS]


def test_deterministic_budgets():
    decision, level = _router().route("¿cuánto costó la integración de pagos?")
    assert level == "deterministic"
    assert SearchTarget.BUDGETS in decision.targets


def test_deterministic_transcripts():
    decision, level = _router().route("¿qué dijo el cliente en la reunión?")
    assert level == "deterministic"
    assert SearchTarget.TRANSCRIPTS in decision.targets


def test_deterministic_technical():
    decision, level = _router().route("detalle del protocolo OAuth y estándar PKCE")
    assert level == "deterministic"
    assert SearchTarget.TECHNICAL_DOCS in decision.targets


def test_fallback_when_llm_fails_searches_all():
    # Query sin vocabulario determinista → cae al LLM; el wrapper falla → fallback a todo.
    decision, level = _router(_RaisingWrapper()).route(
        "plataforma con catálogo, carrito y panel"
    )
    assert level == "fallback"
    assert set(decision.targets) == set(SearchTarget)


@pytest.mark.integration
def test_llm_level_with_real_model():
    from app.foundations.llm_wrapper import LLMWrapper

    decision, level = _router(LLMWrapper(get_settings())).route(
        "plataforma de e-commerce con catálogo de productos y carrito"
    )
    assert level == "llm"
    assert 1 <= len(decision.targets) <= 3
