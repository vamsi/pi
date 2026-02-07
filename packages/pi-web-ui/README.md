# pi-web-ui

Web UI for AI chat built with FastAPI, WebSockets, and SQLite. Provides a browser-based interface to the same agent runtime used by the CLI.

## Running

```bash
pi-web
# Starts on http://localhost:8000
```

## Architecture

```
Browser (static HTML/JS/CSS)
  |
  | WebSocket (JSON protocol)
  |
FastAPI app (app.py)
  |-- WebSocket handler (ws/handler.py)
  |-- Agent manager (agent_manager.py) -- one Agent per session
  |-- REST API (api/) -- sessions, models, file upload
  |-- Storage (storage/) -- SQLite for sessions, settings, provider keys
  |-- Artifacts (artifacts.py) -- generated file preview
```

### WebSocket protocol

The client and server communicate over a typed JSON protocol defined in `ws/protocol.py`. Messages have a `type` field that determines the payload:

- **Client to server:** `prompt`, `abort`, `set_model`, `set_thinking`, `get_models`, `get_sessions`
- **Server to client:** `text_delta`, `thinking_delta`, `tool_start`, `tool_end`, `done`, `error`, `models_list`, `sessions_list`

The serializer (`ws/serializer.py`) handles conversion between internal agent events and the wire protocol.

### Agent manager

`AgentManager` maintains a map of session IDs to `Agent` instances. When a WebSocket connects, it looks up or creates an agent for the session, subscribes to events, and forwards them as WebSocket messages.

### Storage

SQLite-backed storage with async access via `aiosqlite`:

| Store | Purpose |
|-------|---------|
| `sessions.py` | Session metadata and message history |
| `settings.py` | User preferences (default model, thinking level) |
| `provider_keys.py` | Encrypted API key storage |
| `schema.py` | Database schema creation |

### REST API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/sessions` | List sessions |
| `POST /api/sessions` | Create session |
| `GET /api/models` | List available models |
| `POST /api/upload` | File upload for image input |

### Static files

The frontend is plain HTML, CSS, and vanilla JS served from `static/`. No build step.

## File structure

```
src/pi/web/
    __init__.py
    main.py              Entry point (uvicorn)
    app.py               FastAPI application setup
    config.py            Configuration
    agent_manager.py     Agent lifecycle management
    artifacts.py         Generated file handling
    api/
        sessions.py      Session REST endpoints
        models_api.py    Model listing endpoint
        upload.py        File upload endpoint
    storage/
        database.py      SQLite connection management
        schema.py        Table creation
        sessions.py      Session CRUD
        settings.py      Settings CRUD
        provider_keys.py API key storage
    ws/
        handler.py       WebSocket connection handler
        protocol.py      Message type definitions
        serializer.py    Agent event to wire format conversion
    static/
        css/             Stylesheets
        js/              Client-side JavaScript
```
