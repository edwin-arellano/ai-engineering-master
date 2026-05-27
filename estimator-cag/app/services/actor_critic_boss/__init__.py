"""Paquete del patrón Actor-Critic-Boss."""

from app.services.actor_critic_boss.boss import BossService
from app.services.actor_critic_boss.critic import CriticService

__all__ = ["BossService", "CriticService"]
