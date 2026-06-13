"""
Piezo.AI — Alembic Environment Configuration
===============================================
Uses SYNCHRONOUS SQLAlchemy for migrations (more reliable on macOS).
Reads DATABASE_URL from environment, strips asyncpg driver prefix.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from piezo_db.base import Base
from piezo_db.models import (  # noqa: F401 — import to register models
    Dataset,
    Material,
    TrainingJob,
    TrainedModel,
    Prediction,
    PredictionBatch,
)

config = context.config

# Override sqlalchemy.url from environment if set
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # Strip async driver — use sync psycopg2 for migrations
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    config.set_main_option("sqlalchemy.url", database_url)
else:
    # Also strip asyncpg from alembic.ini default
    url = config.get_main_option("sqlalchemy.url") or ""
    config.set_main_option(
        "sqlalchemy.url",
        url.replace("postgresql+asyncpg://", "postgresql://")
    )

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects to DB synchronously."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
