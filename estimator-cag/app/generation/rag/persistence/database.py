"""Engine async + sessionmaker + dependencia FastAPI de sesión.

El engine se crea una vez (singleton de módulo). `expire_on_commit=False` evita
que los objetos queden expirados tras el commit y disparen lazy-loads en async.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.foundations.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Dependencia FastAPI: una sesión async por request, cerrada al terminar."""
    async with AsyncSessionLocal() as session:
        yield session
