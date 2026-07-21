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
from datetime import datetime
from typing import Optional
import os

from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import text, inspect

# Locally this defaults to a file next to the code, same as before. On a
# host with a persistent disk (Render, Fly, etc), the disk is mounted at
# some specific path that is NOT your working directory — set DATABASE_PATH
# in that host's environment variables to point inside the mounted disk
# (e.g. "/var/data/avatar_app.db") so the db file actually lives on
# persistent storage instead of being wiped on every redeploy/restart.
DB_PATH = os.environ.get("DATABASE_PATH", "./avatar_app.db")
_db_dir = os.path.dirname(os.path.abspath(DB_PATH))
os.makedirs(_db_dir, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is required for SQLite + FastAPI's threadpool
# (requests can be served on a different thread than the one that opened
# the connection). Safe here since SQLModel/SQLAlchemy still serializes
# access to the underlying connection per-session.
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    character_name: Optional[str] = Field(default=None, index=True)
    role: str  # "user" | "assistant"
    content: str = ""       # what actually gets sent to the LLM as history
    text: Optional[str] = None       # original user-facing text (user turns)
    text_en: Optional[str] = None    # assistant reply, English
    text_ja: Optional[str] = None    # assistant reply, Japanese
    time: datetime = Field(default_factory=datetime.utcnow)


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