"""Tests del detector de anclas heurístico."""

from app.domain.session import ChatMessage
from app.generation.cag.sessions.compression.anchors import detect_anchors


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


def test_detects_nda() -> None:
    messages = [_user("There is an NDA in place before we share specs.")]
    anchors = detect_anchors(messages)
    assert any(a.startswith("[nda]") for a in anchors)


def test_detects_deadline_and_budget() -> None:
    messages = [
        _user("Hard deadline is end of Q3 and the budget is locked at 50k."),
    ]
    anchors = detect_anchors(messages)
    labels = {a.split("]")[0][1:] for a in anchors}
    assert "deadline" in labels
    assert "locked_budget" in labels


def test_ignores_assistant_messages() -> None:
    # El detector NO debe mirar mensajes del assistant.
    messages = [_assistant("I will assume there is an NDA and a hard deadline.")]
    assert detect_anchors(messages) == []


def test_deduplicates_anchors() -> None:
    messages = [_user("NDA NDA NDA confidentiality agreement.")]
    anchors = detect_anchors(messages)
    # Aunque el patrón aparezca varias veces, el snippet idéntico no se duplica.
    assert len(anchors) == len(set(anchors))


def test_no_false_positive_on_opinions() -> None:
    messages = [_user("I'd like to know the team's opinion on the color scheme.")]
    assert detect_anchors(messages) == []
