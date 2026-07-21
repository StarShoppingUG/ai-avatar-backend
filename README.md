# AI Avatar Backend

This is the high-performance FastAPI backend engine for the AI Avatar system. It manages character prompts, context-aware memory tracking, on-the-fly English-to-Japanese translations (including romanization mapping), and dual-track TTS viseme generations tailored specifically for 3D lip-sync engines.

## Features

- **Groq LPU Acceleration** — Leverages high-speed inference via [Groq](https://groq.com/) for Llama conversational generation and Whisper audio transcriptions.
- **Adaptive Memory Management** — Maintains an in-memory window tracking the last active user-avatar interactions.
- **On-The-Fly Translation Layer** — Seamlessly maps bilingual conversations. Translates incoming Japanese text to English for uniform LLM processing, then streams output shapes translated back to Japanese complete with custom phonetic romanization properties.
- **Dual-Voice Audio & Viseme Synthesis** — Generates synchronized server-side `.mp3` audio tracks (`static/`) alongside granular viseme timeline matrices for exact mouth movements.
- **Smart Pipeline Error Handling** — Rejects blank strings instantly and utilizes non-persisting placeholder mechanics during LLM downtime to protect long-term conversation history states from corruption.

## Project Structure

```text
ai-avatar-backend/
├── static/                 # Directory holding temporary generated audio files
├── app.py                  # Primary entry point holding routers and core pipelines
└── .env                    # System orchestration configuration file
```

## Endpoints

### 1. Configuration & Health
* **`GET /health`** — Validates connection states. Returns the status, AI availability toggles, underlying engine configurations, current target models, and active memory retention parameters.
* **`GET /voices`** — Outputs the primary `VOICE_CATALOG` index definitions along with English and Japanese defaults.

### 2. Conversational Engine
* **`POST /ask`** — Standard entry path for avatar interaction. Accepts an `AskRequest` payload containing text, personality criteria, and target voice layers. Dynamically transforms text across language boundaries, processes LLM history states, writes synchronized local audio files, and returns explicit tracking structures:
  ```json
  {
    "reply": "English string response...",
    "translated_reply": "日本語のテキスト...",
    "romanization": "nihongo no tekisuto...",
    "expression": "neutral",
    "animation": "explain",
    "audio_url": "/static/temp_ja_xyz.mp3",
    "visemes": [...]
  }
  ```
* **`POST /voice`** — Standalone synthesis node. Generates precise audio URLs and phonetic viseme arrays for independent strings based on custom designated culture parameters (`en` or `ja`).

### 3. Audio & Text Processing
* **`POST /stt`** — Highly robust server-side speech-to-text handling via Groq Whisper. Accepts a multi-part form binary payload (`UploadFile`) alongside structural language hints, passing clean transcriptions back to override volatile client-side audio interpreters.
* **`POST /translate`** — Standalone fast translation controller. Forces conversions straight to English or Japanese dependent on targeted input variables.

### 4. History Controller
* **`GET /history`** — Source of truth mapping for the current chat display panel.
* **`POST /reset`** — Flushes the temporary in-memory sequence array to clear current conversation contexts.

## Configuration & Setup

### 1. Requirements Setup
Initialize your environment block and unpack required package assets:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install fastapi uvicorn openai python-dotenv
```

### 2. Environment Variables (`.env`)
Create a `.env` file directly inside the root folder matching the parameters below:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant or any other model
GROQ_STT_MODEL=whisper-large-v3-turbo or any other stt model

# Core orchestration parameters: "gro"
LLM_PROVIDER=groq
```
*(Note: Ensure your `GROQ_BASE_URL` points to the standard OpenAI-compatible inference path `https://api.groq.com/openai/v1` to prevent 404 connection routing errors during execution).*

### 3. Execution
Fire up the local service pipeline using Uvicorn:
```bash
uvicorn app:app --reload
```
The server will bind immediately to **`http://127.0.0.1:8000`**. You can safely audit your exact operational request/response parameters by checking the interactive UI layer exposed at `http://127.0.0`.
