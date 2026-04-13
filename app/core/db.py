from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import sqlalchemy as sa
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings

settings = get_settings()
engine = create_async_engine(
    settings.postgres_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
)
AsyncSessionFactory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_ALEMBIC_DIR = _REPO_ROOT / "alembic"


async def ensure_schema_ready() -> None:
    import app.models  # noqa: F401

    async with engine.begin() as connection:
        existing_tables, applied_revisions = await connection.run_sync(
            lambda sync_conn: (
                set(sa.inspect(sync_conn).get_table_names()),
                {
                    str(row[0])
                    for row in sync_conn.execute(sa.text("SELECT version_num FROM alembic_version"))
                }
                if "alembic_version" in sa.inspect(sync_conn).get_table_names()
                else set(),
            )
        )
    expected_tables = {table.name for table in SQLModel.metadata.sorted_tables}
    missing_tables = sorted(expected_tables - existing_tables)
    if missing_tables:
        raise RuntimeError(
            "database schema is not ready; run 'alembic upgrade head' before starting the app. "
            f"Missing tables: {', '.join(missing_tables)}"
        )
    alembic_config = Config(str(_ALEMBIC_INI))
    alembic_config.set_main_option("script_location", str(_ALEMBIC_DIR))
    expected_revisions = set(ScriptDirectory.from_config(alembic_config).get_heads())
    if applied_revisions != expected_revisions:
        raise RuntimeError(
            "database schema is out of date; "
            "run 'alembic upgrade head' before starting the app. "
            f"Current revisions: {sorted(applied_revisions)}; "
            f"expected: {sorted(expected_revisions)}"
        )


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session
