"""
db.py — SQLite persistence for chat history and per-user settings.

No accounts/login: each browser generates a random UUID on first visit
(stored in localStorage) and sends it as the `X-User-Id` header on every
request. That UUID is the only thing that identifies a "user" here — see
get_user_id() in backend.py, which reads the header and hands you back a
plain string. Nothing here validates or authenticates it; it's just a
foreign key.

Two tables:
  - ChatMessage   — one row per turn, replaces the old in-memory
                    `conversation_history` list. Same fields the frontend
                    already expects from GET /history (character_name,
                    role, content, text/text_en/text_ja, time).
  - UserSettings  — one row per user_id: last-selected avatar, UI language,
                    reply language. Upserted from a single endpoint.
"""
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

if DATABASE_URL:
    # Neon (and some other providers) hand out URLs starting with
    # "postgres://", which SQLAlchemy's psycopg2 dialect no longer accepts —
    # it wants the explicit "postgresql://" scheme. Normalize either way.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    # check_same_thread is a SQLite-only connect arg — Postgres doesn't
    # understand it and errors if passed.
    engine = create_engine(DATABASE_URL, echo=False)
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


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_missing_columns()


def _migrate_missing_columns() -> None:
    """
    create_all() only creates tables that don't exist yet — it never alters
    a table that's already there. If a field gets added to a model (like
    `last_avatar` on UserSettings) after avatar_app.db already exists on
    disk, the running code expects a column the actual file doesn't have,
    and every query touching it throws "no such column: <field>" — silently,
    since callers here catch/swallow that error. This walks every SQLModel
    table, diffs its Python columns against what's actually in the SQLite
    file, and ALTERs in whatever is missing so existing databases catch up
    without anyone having to delete/recreate the file by hand.
    """
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


def get_session():
    """FastAPI dependency — yields a Session, closes it when the request ends."""
    with Session(engine) as session:
        yield session