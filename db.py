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
    # JSON-serialized dict, e.g. '{"slot-1::female_ug": {"name": "...", "persona": "...", "personaJa": "..."}}'
    # Stored as text (SQLite/most SQLAlchemy backends have no native dict
    # column) — save_settings()/get_settings() in main.py handle the
    # json.dumps/json.loads at the API boundary, so this column never holds
    # anything but a plain string or None.
    persona_overrides: Optional[str] = None


class AppSettings(SQLModel, table=True):
    """Same shape as UserSettings, but keyed on app_id alone — for
    integrators who want every user of their app to share one settings row
    (last_avatar, languages, persona edits) instead of per-browser
    isolation. Selected via the X-Settings-Scope: app header — see
    get_settings_scope() in app.py. UserSettings (per-browser/UUID, the
    original behavior) is untouched above and remains the default when
    that header is absent."""
    app_id: str = Field(primary_key=True)
    # Second scoping dimension alongside app_id — lets one app_id (one
    # integrator/tenant) share settings per some sub-grouping (e.g. a
    # "character" = scenario + avatar combo) instead of forcing every
    # scope=app user of that app_id onto a single shared row. Selected via
    # the X-Settings-Group header — see get_settings_group() in app.py.
    # Defaults to "" so any existing scope=app usage that never sends this
    # header keeps landing on the same row it always has (app_id, "").
    # Composite primary key with app_id — see _migrate_app_settings_group()
    # below for how this is rolled onto an already-existing table (the
    # generic add-missing-column migration can't handle this on its own:
    # it only adds columns, it never changes which columns make up the
    # primary key).
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


def _migrate_app_settings_group() -> None:
    """
    Unlike _migrate_missing_columns() above (which only ever ADDs a
    column), this handles a schema change that column-add alone can't
    cover: AppSettings' primary key is changing from app_id alone to the
    composite (app_id, settings_group). A table created before this
    change exists on disk with app_id as its only PK column and no
    settings_group column at all — _migrate_missing_columns() will have
    just added settings_group as a plain nullable column with no default,
    but the primary key itself is untouched by that pass, so two rows for
    the same app_id under different settings_group values would still
    collide (session.get(AppSettings, app_id) / merge-by-PK would treat
    them as the same row). This function is what actually finishes the
    job: backfills any NULL settings_group to "" (the same default new
    rows get), then swaps the primary key constraint to the composite
    form. Safe to run every startup — it checks the current PK columns
    first and does nothing once a table is already on the composite key
    (including a table that never existed before this change and was
    therefore created by create_all() with the composite key from day
    one, in which case there's nothing to do here at all).
    """
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
            # SQLite has no ALTER TABLE support for dropping/adding a
            # primary key constraint (or for adding NOT NULL to an
            # existing column) — the only way to change a table's PK is
            # to rebuild it: create a new table with the target schema,
            # copy the (now-backfilled) rows across, drop the old table,
            # rename the new one into place.
            tmp_name = f"{table_name}_new"
            conn.execute(text(f'DROP TABLE IF EXISTS "{tmp_name}"'))
            # Raw DDL rather than reflecting AppSettings.__table__ into a
            # renamed copy — keeps this independent of SQLAlchemy-version
            # differences in the Table-copy API, and matches the plain
            # text()-based approach _migrate_missing_columns() already
            # uses elsewhere in this file. Column types match the other
            # text/optional-string columns on this model (see UserSettings
            # above, which the same columns are modeled after).
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