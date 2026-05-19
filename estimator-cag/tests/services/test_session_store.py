"""Tests del SessionStore in-memory."""

from __future__ import annotations

import time

import pytest

from app.services.sessions.session_store import (
    SessionNotFoundError,
    SessionStore,
)


def test_create_returns_session_with_unique_id() -> None:
    store = SessionStore(idle_ttl_seconds=3600)
    a = store.create()
    b = store.create()
    assert a.session_id != b.session_id
    assert store.size() == 2


def test_get_returns_existing_session() -> None:
    store = SessionStore(idle_ttl_seconds=3600)
    created = store.create()
    fetched = store.get(created.session_id)
    assert fetched.session_id == created.session_id


def test_get_unknown_session_raises() -> None:
    store = SessionStore(idle_ttl_seconds=3600)
    with pytest.raises(SessionNotFoundError):
        store.get("does-not-exist")


def test_save_refreshes_last_activity() -> None:
    store = SessionStore(idle_ttl_seconds=3600)
    session = store.create()
    initial_activity = session.last_activity_at
    time.sleep(0.01)
    store.save(session)
    assert session.last_activity_at > initial_activity


def test_purge_expired_sessions() -> None:
    """TTL=0 purga todas las sesiones antes del próximo acceso (modo test)."""
    store = SessionStore(idle_ttl_seconds=0)
    session = store.create()
    with pytest.raises(SessionNotFoundError):
        store.get(session.session_id)
    assert store.size() == 0


def test_delete_removes_session() -> None:
    store = SessionStore(idle_ttl_seconds=3600)
    session = store.create()
    assert store.delete(session.session_id) is True
    assert store.delete(session.session_id) is False


def test_history_sliding_window() -> None:
    """La ventana deslizante descarta los pares más antiguos."""
    store = SessionStore(idle_ttl_seconds=3600)
    session = store.create()

    for i in range(10):
        session.history.append_turn(
            user_content=f"user message {i}",
            assistant_content=f"assistant reply {i}",
            max_turns=3,
        )

    # 3 pares × 2 mensajes = 6 mensajes en total.
    assert len(session.history.messages) == 6
    # Los más recientes deben ser los pares 7, 8, 9.
    first_kept = session.history.messages[0]
    assert "user message 7" in first_kept.content


def test_apply_patch_preserves_existing_facts() -> None:
    """apply_patch nunca borra hechos con nulos del patch."""
    from app.schemas.session import ProjectMetadata, ProjectMetadataUpdate

    initial = ProjectMetadata(
        project_name="BookFlow",
        assumed_team_size=4,
        mentioned_technologies=["Swift", "Kotlin"],
        agreed_scope="Login, calendar, push reminders.",
    )
    # Patch que solo trae una tecnología nueva.
    patch = ProjectMetadataUpdate(mentioned_technologies=["FastAPI"])
    merged = initial.apply_patch(patch)

    assert merged.project_name == "BookFlow"
    assert merged.assumed_team_size == 4
    assert merged.agreed_scope == "Login, calendar, push reminders."
    assert "Swift" in merged.mentioned_technologies
    assert "Kotlin" in merged.mentioned_technologies
    assert "FastAPI" in merged.mentioned_technologies
