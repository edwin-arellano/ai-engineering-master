"""Fase 1 (reformulación). Unit: mockea el wrapper y verifica que se usa el alias
`reformulator` y la temperatura del setting, sin pegar a ningún LLM real."""

from __future__ import annotations

from app.foundations.config import get_settings
from app.foundations.llm_wrapper import REFORMULATOR_ALIAS
from app.generation.rag.retrieval.reformulation import reformulate_transcript
from app.generation.rag.schemas import ReformulatedQuery


class _SpyWrapper:
    """Stub de LLMWrapper: registra los kwargs de la llamada y devuelve un brief fijo."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return ReformulatedQuery(
            project_function="Plataforma de telemedicina con videoconsultas",
            technologies=["python", "postgresql"],
            sector="healthcare",
            scale="medium",
            countries=["ES"],
            constraints=["HIPAA"],
            search_text="telemedicina videoconsulta historia clínica electrónica",
        )


def test_reformulate_uses_reformulator_alias_and_temperature():
    settings = get_settings()
    wrapper = _SpyWrapper()

    result = reformulate_transcript(
        transcript="Reunión ruidosa sobre una app de telemedicina ...",
        wrapper=wrapper,
        settings=settings,
    )

    assert isinstance(result, ReformulatedQuery)
    assert result.sector == "healthcare"
    assert result.search_text  # denso, no vacío

    assert len(wrapper.calls) == 1
    call = wrapper.calls[0]
    assert call["alias"] == REFORMULATOR_ALIAS
    assert call["response_model"] is ReformulatedQuery
    assert call["temperature"] == settings.reformulator_temperature
    assert call["max_tokens"] == settings.reformulator_max_tokens
