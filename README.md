# AI Avatar Backend

This is the high-performance FastAPI backend engine for the AI Avatar system. It manages character prompts, context-aware memory tracking, on-the-fly English-to-Japanese translations (including romanization mapping), dual-track TTS viseme generations tailored specifically for 3D lip-sync engines, and persisted per-user/per-app chat history and settings via Postgres or SQLite.

## Features

- **Groq LPU Acceleration** — Leverages high-speed inference via [Groq](https://groq.com/) for conversational generation and Whisper audio transcriptions.
- **Persistent Storage** — Chat history and per-user settings are stored via SQLModel, backed by Postgres (recommended for production — see [Database](#database)) or a local SQLite file for development.
- **Multi-Tenant Identity** — Every request is scoped by an `X-App-Id` (which third-party app/deployment) + `X-User-Id` (which of that app's end-users) pair, so multiple integrators can embed the same backend without their users' data colliding — see [Identity & Multi-Tenancy](#identity--multi-tenancy).
- **Configurable Settings Scoping** — Settings can be stored per-user (default) or per-app via `X-Settings-Scope`, with an optional `X-Settings-Group` dimension to further split app-scoped settings (e.g. per character/scenario within a single app) — see [Settings Scope: Per-App vs Per-User](#settings-scope-per-app-vs-per-user).
- **On-The-Fly Translation Layer** — Seamlessly maps bilingual conversations. Translates incoming Japanese text to English for uniform LLM processing, then streams output shapes translated back to Japanese complete with custom phonetic romanization properties.
- **Dual-Voice Audio & Viseme Synthesis** — Generates synchronized server-side `.mp3` audio tracks (`static/`) alongside granular viseme timeline matrices for exact mouth movements.
- **Smart Pipeline Error Handling** — Rejects blank strings instantly and utilizes non-persisting placeholder mechanics during LLM downtime to protect long-term conversation history states from corruption. Rate-limit errors from Groq are detected specifically and surfaced as a distinct "hit a rate limit, try again in X" message rather than the generic offline fallback, so users get an accurate reason instead of being told to just retry immediately.

## Project Structure

```text
ai-avatar-backend/
├── static/                 # Directory holding temporary generated audio files
├── app.py                  # Primary entry point — FastAPI app, routers, core pipelines
├── db.py                    # SQLModel schema (ChatMessage, UserSettings, AppSettings) + Postgres/SQLite engine setup
├── ai.py                    # Groq LLM client, Whisper transcription, date/time context builder
├── translation.py           # EN<->JA translation helpers
├── json_utils.py             # Tolerant parsing of the LLM's behavior JSON
├── requirements.txt
└── .env                     # System orchestration configuration file
```

## Endpoints

### 1. Configuration & Health
* **`GET /health`** — Validates connection states. Returns the status, AI availability toggles, underlying engine configurations, current target models, and active memory retention parameters.

### 2. Conversational Engine
* **`POST /ask`** — Standard entry path for avatar interaction. Accepts an `AskRequest` payload containing text, personality criteria, and target voice layers. Dynamically transforms text across language boundaries, processes LLM history states, writes synchronized local audio files, persists the turn to the database, and returns explicit tracking structures:
```json
  {
    "reply": "English string response...",
    "translated_reply": "日本語のテキスト...",
    "romanization": "nihongo no tekisuto...",
    "expression": "neutral",
    "animation": "explain",
    "audio_url_en": "/static/temp_en_xyz.mp3",
    "audio_url_ja": "/static/temp_ja_xyz.mp3",
    "audio_url": "/static/temp_ja_xyz.mp3",
    "visemes_en": [...],
    "visemes_ja": [...],
    "visemes": [...],
    "primary": "en"
  }
```
Accepts an optional `teaching_mode` boolean. When `true` (used by language-tutor avatars), the model produces a single reply that naturally mixes the taught language and the student's native language — words in the language being taught are inline-annotated with their reading/romanization in `[brackets]` — instead of generating a separate English reply plus a translated Japanese counterpart. In this mode `translated_reply` mirrors `reply`, `romanization` comes back empty, and `audio_url_en`/`audio_url_ja` both point to the same single generated audio track (synthesized once, with a slower speech rate for learners, after the `[bracket]` annotations are stripped out so they aren't spoken twice).

* **`POST /voice`** — Standalone synthesis node. Generates precise audio URLs and phonetic viseme arrays for independent strings based on custom designated culture parameters (`en` or `ja`).

### 3. Audio & Text Processing
* **`POST /stt`** — Highly robust server-side speech-to-text handling via Groq Whisper. Accepts a multi-part form binary payload (`UploadFile`) alongside structural language hints, passing clean transcriptions back to override volatile client-side audio interpreters.
* **`POST /translate`** — Standalone fast translation controller. Forces conversions straight to English or Japanese dependent on targeted input variables.

### 4. History & Settings Controller
* **`GET /history`** — Source of truth mapping for the current chat display panel. Requires `X-User-Id` (and `X-App-Id` if multi-tenant). Optional `?character_name=` query param scopes to a single avatar's turns.
* **`POST /reset`** — Deletes this user's persisted chat history rows. Optional `?character_name=` clears just that avatar's turns; omitted clears everything for this user.
* **`GET /settings`** — Returns the saved `{ ui_language, response_language, last_avatar }`, or defaults if never saved. Accepts `X-Settings-Scope` (`"app"` or `"user"`, default `"user"`) and, when scope is `"app"`, an optional `X-Settings-Group` — see [Settings Scope: Per-App vs Per-User](#settings-scope-per-app-vs-per-user).
* **`POST /settings`** — Partial update — only send the fields that changed. Returns the full saved settings row. Respects the same `X-Settings-Scope` / `X-Settings-Group` headers as `GET /settings`.

## Identity & Multi-Tenancy

There are no user accounts. Every request that touches history or settings
requires an `X-User-Id` header (400 if missing), and optionally accepts an
`X-App-Id` header. Internally, `get_user_id()` combines them into one
scoped key:
X-App-Id: acme-corp
X-User-Id: user-48213
→ stored/queried as `"acme-corp::user-48213"`

This means two different integrators' users with the identical literal
`user-48213` never collide — every `ChatMessage`/`UserSettings` row is
naturally isolated per app, with no separate `app_id` column or schema
migration required. Requests without `X-App-Id` fall back to a shared
`"default"` tenant (this is what the reference frontend's own demo page
uses, via its hostname-based fallback — see the frontend README's
[Persistence & Identity](../README.md#persistence--identity) section for
the client side of this).

## Settings Scope: Per-App vs Per-User

By default, settings (`last_avatar`, `ui_language`, `response_language`,
`persona_overrides`) are scoped per-user — same `"acme-corp::user-48213"`
key described above. An optional `X-Settings-Scope` header (`"app"` or
`"user"`, default `"user"`) lets an integrator instead share one settings
row across *all* of that app's users:

- **`scope=user`** (default) — settings are read/written to `UserSettings`,
  keyed by `app_id::user_id` as usual. No behavior change from before this
  feature existed.
- **`scope=app`** — settings are read/written to a separate `AppSettings`
  table, keyed by `app_id` (see Settings Group below for a further
  sub-dimension). Every user of that `app_id` shares one settings row —
  useful when an integrator wants one avatar/language choice to apply
  app-wide rather than per visitor.

### Settings Group (grouping within app-scope)

`AppSettings` rows can be further split with an optional `X-Settings-Group`
header (string, defaults to `""` when absent). This only applies when
`X-Settings-Scope: app` — the `user`-scoped path has no group concept.

`AppSettings` is keyed by the composite `(app_id, settings_group)` rather
than `app_id` alone. This matters for integrators who have multiple
"characters" (a scenario + avatar combination) within a single `app_id`
and want settings shared per-character rather than shared across the
entire app — without a group dimension, changing the avatar on one
character would update every character under that `app_id`.

```
X-App-Id: acme-corp
X-Settings-Scope: app
X-Settings-Group: character-a
→ stored/queried as AppSettings("acme-corp", "character-a")
```

Existing `scope=app` integrators who never send `X-Settings-Group` are
unaffected — the header defaults to `""`, which is the same stable value
existing rows were migrated to, so pre-existing app-scoped settings land
in the same row as before this feature shipped.

> `app_id` still means "which integrator/tenant," not "which character."
> Use `X-Settings-Group` for character/scenario-level splitting rather
> than encoding it into `X-App-Id` — keeping `app_id` reserved for
> tenant identity is what any future per-tenant tooling (rate limiting,
> admin dashboards, billing) will assume.

## Database

Storage is handled by `db.py` via SQLModel, and picks its backend based on
the `DATABASE_URL` environment variable:

- **`DATABASE_URL` set** → connects to that Postgres instance (tested against [Neon](https://neon.tech), but any standard Postgres connection string works). `postgres://` URLs are automatically normalized to `postgresql://` for SQLAlchemy's psycopg2 dialect.
- **`DATABASE_URL` unset** → falls back to a local SQLite file (`DATABASE_PATH` env var, defaults to `./avatar_app.db`) — convenient for local development, but **not persistent** on most free-tier hosts with ephemeral filesystems (e.g. Render's free web services wipe local files on every restart/redeploy).

For any real deployment, set `DATABASE_URL` to a real Postgres connection
string. Tables (`chatmessage`, `usersettings`, `appsettings`) are created
automatically on startup via `init_db()`, and a lightweight migration
helper (`_migrate_missing_columns()`) `ALTER TABLE`s in any columns that
were added to the models after the database file/instance already
existed — no manual migrations needed for simple additive schema changes.

> **`AppSettings` primary-key migration (settings-group):** unlike simple
> additive columns, the change that added `settings_group` to
> `AppSettings` altered an *existing column's role* — the table's primary
> key went from `app_id` alone to the composite `(app_id,
> settings_group)`. `create_all()` and `_migrate_missing_columns()` don't
> handle primary-key changes, so this needed a dedicated one-time
> migration (`_migrate_app_settings_group()` in `db.py`), run
> automatically on `init_db()`:
> - **Postgres** — a real `ALTER TABLE`: backfills any `NULL`
>   `settings_group` to `""`, then `DROP CONSTRAINT` / `ADD PRIMARY KEY
>   (app_id, settings_group)`. No table rebuild needed.
> - **SQLite** (local dev) — SQLite can't alter a primary key in place,
>   so this path rebuilds the table and copies existing rows across,
>   landing each at `(app_id, "")` to preserve pre-migration data.
> - Idempotent either way — safe to run against an already-migrated
>   database on every startup.

> Note: `POST /reset`'s response currently always reports
> `"mode": "sqlite"` regardless of which database is actually configured
> — harmless (the frontend doesn't read this field), but worth knowing if
> you're debugging which backend is active; check `DATABASE_URL` directly
> rather than trusting this field.

## Configuration & Setup

### 1. Requirements Setup
Initialize your environment block and unpack required package assets:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

`requirements.txt` should include at minimum:
fastapi
uvicorn
sqlmodel
psycopg2-binary
python-dotenv
edge-tts
openai
pytz
python-multipart

### 2. Environment Variables (`.env`)
Create a `.env` file directly inside the root folder matching the parameters below:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b or any other model
GROQ_STT_MODEL=whisper-large-v3-turbo or any other stt model

# Database — set this for persistent storage (Postgres). Omit it entirely
# to fall back to a local SQLite file for quick local development.
DATABASE_URL=postgresql://user:password@host/dbname
```
*(Note: Ensure your `GROQ_BASE_URL` points to the standard OpenAI-compatible inference path `https://api.groq.com/openai/v1` to prevent 404 connection routing errors during execution).*

### 3. Execution
Fire up the local service pipeline using Uvicorn:
```bash
uvicorn app:app --reload
```
The server will bind immediately to **`http://127.0.0.1:8000`**. You can safely audit your exact operational request/response parameters by checking the interactive UI layer exposed at `http://127.0.0.1:8000/docs`.

### 4. Deployment

Deployed reference instance runs on [Railway](https://railway.app) (git-push deploy, no Dockerfile required) with the database hosted on [Neon](https://neon.tech). Set the same environment variables from step 2 in your host's dashboard, and set the start command to:
uvicorn app:app --host 0.0.0.0 --port $PORT