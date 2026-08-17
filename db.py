from datetime import datetime, timezone
from typing import Optional
import os

from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Column, DateTime, text, inspect

# DATABASE_URL, when set (e.g. on Railway/Render pointing at a managed
# Postgres like Neon), takes priority — this is the persistent-across-
# restarts path. Falls back to a local SQLite file only when it's unset,
# so local dev without any DB configured still works exactly as before.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# TEMP DIAGNOSTIC — see HANDOFF2.md open bug (CharacterCard infinite loading
# on back-nav). Confirms which engine path Railway's deployed container
# actually takes, since this can silently diverge from source if
# DATABASE_URL isn't reaching the container. Remove once confirmed.
print(f"🔍 DB backend: {'postgres' if DATABASE_URL else 'sqlite-fallback'}")

if DATABASE_URL:
    # Neon (and some other providers) hand out URLs starting with
    # "postgres://", which SQLAlchemy's psycopg2 dialect no longer accepts —
    # it wants the explicit "postgresql://" scheme. Normalize either way.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    # check_same_thread is a SQLite-only connect arg — Postgres doesn't
    # understand it and errors if passed.
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=1800,
    )
else:
    DB_PATH = os.environ.get("DATABASE_PATH", "./avatar_app.db")
    _db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(_db_dir, exist_ok=True)
    sqlite_url = f"sqlite:///{DB_PATH}"
    # check_same_thread=False is required for SQLite + FastAPI's threadpool
    # (requests can be served on a different thread than the one that opened
    # the connection). Safe here since SQLModel/SQLAlchemy still serializes
    # access to the underlying connection per-session.
    engine = create_engine(sqlite_url, echo=False, connect_args={"check_same_thread": False})


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    character_name: Optional[str] = Field(default=None, index=True)
    role: str  # "user" | "assistant"
    content: str = ""       # what actually gets sent to the LLM as history
    text: Optional[str] = None       # original user-facing text (user turns)
    text_en: Optional[str] = None    # assistant reply, English
    text_ja: Optional[str] = None    # assistant reply, Japanese
    time: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
        )
    )
    def time_iso(self) -> str:
        """
        SQLite has no real timezone-aware datetime type — DateTime(timezone=True)
        stores the value correctly, but SQLAlchemy reads it back as a *naive*
        datetime (tzinfo dropped), even though the clock value is still UTC.
        Calling .isoformat() on that naive value produces a string with no
        offset/Z suffix, and JS's `new Date(...)` treats an offset-less string
        as *local* time — so the frontend silently reinterpreted a UTC
        timestamp as local time, shifting every displayed time by the
        server/browser's UTC offset. Reattaching tzinfo=utc here (it's always
        UTC — that's the only thing ever written to this column) fixes the
        serialized string so the frontend converts it correctly.
        """
        value = self.time
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class UserSettings(SQLModel, table=True):
    user_id: str = Field(primary_key=True)
    ui_language: Optional[str] = "en"
    response_language: Optional[str] = "ja"
    last_avatar: Optional[str] = None

    persona_overrides: Optional[str] = None


class AppSettings(SQLModel, table=True):

    app_id: str = Field(primary_key=True)

    settings_group: str = Field(default="", primary_key=True)
    ui_language: Optional[str] = "en"
    response_language: Optional[str] = "ja"
    last_avatar: Optional[str] = None
    persona_overrides: Optional[str] = None


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_missing_columns()
    _migrate_app_settings_group()


def _migrate_missing_columns() -> None:

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table_name, table in SQLModel.metadata.tables.items():
            if not inspector.has_table(table_name):
                continue  # brand-new table — create_all() already made it fully
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(engine.dialect)
                conn.execute(text(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {col_type}'
                ))
                print(f"🛠️  db migration: added missing column {table_name}.{column.name}")


def _migrate_app_settings_group() -> None:

    table_name = AppSettings.__tablename__
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return  # create_all() will make it fresh, already on the composite key

    pk = inspector.get_pk_constraint(table_name)
    pk_columns = set(pk.get("constrained_columns") or [])
    if pk_columns == {"app_id", "settings_group"}:
        return  # already migrated (or created fresh) — nothing to do

    existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
    if "settings_group" not in existing_columns:
        # Defensive — _migrate_missing_columns() runs immediately before
        # this and should already have added it, but don't assume.
        with engine.begin() as conn:
            conn.execute(text(
                f'ALTER TABLE "{table_name}" ADD COLUMN "settings_group" VARCHAR'
            ))

    dialect = engine.dialect.name
    with engine.begin() as conn:
        # Backfill any existing rows (created before settings_group
        # existed) to the same default new rows get, so scope=app usage
        # with no group set keeps landing on the same row as before.
        conn.execute(text(
            f'UPDATE "{table_name}" SET settings_group = \'\' WHERE settings_group IS NULL'
        ))

        if dialect == "postgresql":
            old_pk_name = pk.get("name")
            if old_pk_name:
                conn.execute(text(f'ALTER TABLE "{table_name}" DROP CONSTRAINT "{old_pk_name}"'))
            conn.execute(text(f'ALTER TABLE "{table_name}" ALTER COLUMN settings_group SET NOT NULL'))
            conn.execute(text(
                f'ALTER TABLE "{table_name}" ADD PRIMARY KEY (app_id, settings_group)'
            ))
        elif dialect == "sqlite":

            tmp_name = f"{table_name}_new"
            conn.execute(text(f'DROP TABLE IF EXISTS "{tmp_name}"'))

            conn.execute(text(
                f'CREATE TABLE "{tmp_name}" ('
                f'app_id VARCHAR NOT NULL, '
                f'settings_group VARCHAR NOT NULL DEFAULT \'\', '
                f'ui_language VARCHAR, '
                f'response_language VARCHAR, '
                f'last_avatar VARCHAR, '
                f'persona_overrides VARCHAR, '
                f'PRIMARY KEY (app_id, settings_group)'
                f')'
            ))
            conn.execute(text(
                f'INSERT INTO "{tmp_name}" (app_id, settings_group, ui_language, '
                f'response_language, last_avatar, persona_overrides) '
                f'SELECT app_id, settings_group, ui_language, response_language, '
                f'last_avatar, persona_overrides FROM "{table_name}"'
            ))
            conn.execute(text(f'DROP TABLE "{table_name}"'))
            conn.execute(text(f'ALTER TABLE "{tmp_name}" RENAME TO "{table_name}"'))
        else:
            raise RuntimeError(
                f"_migrate_app_settings_group: no migration path implemented for "
                f"dialect '{dialect}' — add one before deploying against this DB."
            )

    print(f"🛠️  db migration: {table_name} primary key is now (app_id, settings_group)")


def get_session():
    """FastAPI dependency — yields a Session, closes it when the request ends."""
    with Session(engine) as session:
        yield session