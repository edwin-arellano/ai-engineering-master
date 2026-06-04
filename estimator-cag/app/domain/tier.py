"""Enum del tier de usuario.

El tier es una dimensión de producto, no de autorización: define qué
experiencia recibe el usuario (qué énfasis tiene la estimación), no qué puede
hacer. Se resuelve en runtime con el TierResolver.
"""

from enum import StrEnum


class UserTier(StrEnum):
    """Perfiles de usuario que el estimator atiende de forma diferenciada."""

    DEVELOPER = "developer"
    PM = "pm"
    EXECUTIVE = "executive"
