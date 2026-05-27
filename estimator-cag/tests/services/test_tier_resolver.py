"""Tests del resolver de tier heurístico."""

from app.schemas.session import ProjectMetadata
from app.schemas.tier import UserTier
from app.services.tiers.resolver import TierContext, TierResolver


def _ctx(transcript: str) -> TierContext:
    return TierContext(transcript=transcript, project_metadata=ProjectMetadata())


def test_executive_keywords_resolve_executive() -> None:
    resolver = TierResolver()
    tier = resolver.resolve(_ctx("Prepare a business case for the board, go/no-go."))
    assert tier is UserTier.EXECUTIVE


def test_pm_keywords_resolve_pm() -> None:
    resolver = TierResolver()
    tier = resolver.resolve(_ctx("I need the roadmap with milestones and sprints."))
    assert tier is UserTier.PM


def test_developer_keywords_resolve_developer() -> None:
    resolver = TierResolver()
    tier = resolver.resolve(_ctx("Break down the API endpoints and the backend stack."))
    assert tier is UserTier.DEVELOPER


def test_default_when_no_rule_matches() -> None:
    resolver = TierResolver(default_tier=UserTier.PM)
    tier = resolver.resolve(_ctx("Something completely unrelated."))
    assert tier is UserTier.PM


def test_failing_rule_does_not_break_resolver() -> None:
    def broken_rule(ctx: TierContext):
        raise RuntimeError("boom")

    def good_rule(ctx: TierContext):
        return UserTier.EXECUTIVE, "good"

    resolver = TierResolver(rules=(broken_rule, good_rule))
    # La regla rota se salta; la siguiente resuelve.
    assert resolver.resolve(_ctx("anything")) is UserTier.EXECUTIVE
