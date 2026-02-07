# pi

A modular AI agent toolkit written in Python. Ships as a `uv` workspace with independent packages that compose together -- from raw LLM streaming through stateful agent loops to full terminal and web interfaces.

## Packages

| Package | Purpose | Lines |
|---------|---------|-------|
| [pi-ai](packages/pi-ai/) | Unified LLM API across 9 providers | 7,800 |
| [pi-agent](packages/pi-agent/) | Stateful agent runtime with tool execution | 1,000 |
| [pi-tui](packages/pi-tui/) | Terminal UI framework with differential rendering | 16,400 |
| [pi-coding-agent](packages/pi-coding-agent/) | CLI coding agent with file tools, sessions, extensions | 7,900 |
| [pi-mom](packages/pi-mom/) | Slack bot with Docker sandbox | 7,700 |
| [pi-web-ui](packages/pi-web-ui/) | Web UI with FastAPI and WebSockets | 2,000 |
| [pi-pods](packages/pi-pods/) | GPU pod manager for vLLM deployments | WIP |

## Architecture

```
pi-ai          Low-level LLM streaming (provider-agnostic)
  |
pi-agent       Agent loop with tool execution, steering, follow-ups
  |
pi-coding-agent / pi-mom / pi-web-ui    Application-level consumers
  |
pi-tui         Terminal rendering (used by pi-coding-agent)
```

Each layer depends only on the one below it. `pi-ai` has zero knowledge of agents or tools beyond what the LLM needs. `pi-agent` has zero knowledge of specific tools -- it receives them as configuration. Applications wire everything together.

## Quick start

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd pi
uv sync --all-packages
```

Run tests across all packages:

```bash
uv run pytest
```

Run a specific package's tests:

```bash
uv run pytest packages/pi-ai/tests/ -v
uv run pytest packages/pi-tui/tests/ -v
```

## Development

The workspace is configured in `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

Packages reference each other as workspace dependencies:

```toml
# In packages/pi-agent/pyproject.toml
dependencies = ["pi-ai"]

# In packages/pi-coding-agent/pyproject.toml
dependencies = ["pi-ai", "pi-agent", "pi-tui"]
```

Adding a new package:

```bash
mkdir -p packages/pi-foo/src/pi/foo packages/pi-foo/tests
# Create pyproject.toml, __init__.py
uv sync --all-packages
```

## License

Proprietary.
