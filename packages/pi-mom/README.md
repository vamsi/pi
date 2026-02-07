# pi-mom

Slack bot powered by the agent runtime with Docker sandbox isolation. Runs tools inside containers so the bot can execute code, read/write files, and run commands without affecting the host.

## Running

```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
pi-mom
```

## Architecture

```
Slack (Socket Mode)
  |
  | Events API
  |
Slack handler (slack.py)
  |
Agent (pi-agent) with sandbox tools
  |
Docker sandbox (sandbox.py)
  |-- Container lifecycle
  |-- File transfer (host <-> container)
  |-- Command execution
```

### Slack integration

Uses the `slack-sdk` Socket Mode client. Listens for messages in configured channels, creates or resumes agent sessions per thread, and posts responses back to Slack with formatting.

### Docker sandbox

Each conversation gets its own Docker container. The sandbox module handles:

- Container creation with configurable image, resource limits, and mounts
- Command execution inside containers with timeout
- File upload/download between host and container
- Container cleanup on session end

### Tools

The bot uses modified versions of the standard coding tools that execute inside the Docker sandbox:

| Tool | Description |
|------|-------------|
| `bash` | Execute commands inside the container |
| `read` | Read files from the container filesystem |
| `write` | Write files to the container filesystem |
| `edit` | Edit files inside the container |
| `attach` | Attach files from Slack messages to the container |

### Event system

The bot processes agent events and converts them to Slack message updates. Long-running operations show a typing indicator. Tool executions are logged. Errors are reported inline.

### Session store

Sessions are persisted so conversations can be resumed across bot restarts.

## File structure

```
src/pi/mom/
    __init__.py
    main.py          Entry point
    agent.py         Agent setup and configuration
    context.py       Conversation context management
    download.py      Slack file download
    events.py        Agent event to Slack message conversion
    log.py           Structured logging
    sandbox.py       Docker container management
    slack.py         Slack client and event handler
    store.py         Session persistence
    tools/
        __init__.py  Tool registration
        attach.py    Slack file attachment
        bash.py      Sandboxed shell execution
        edit.py      Sandboxed file editing
        read.py      Sandboxed file reading
        truncate.py  Output truncation for Slack message limits
        write.py     Sandboxed file writing
```

## Configuration

Environment variables:

| Variable | Purpose |
|----------|---------|
| `SLACK_BOT_TOKEN` | Slack bot OAuth token |
| `SLACK_APP_TOKEN` | Slack app-level token (for Socket Mode) |
| `ANTHROPIC_API_KEY` | LLM API key (or set the key for your chosen provider) |
| `PI_MOM_DOCKER_IMAGE` | Container image (default: `python:3.12-slim`) |
| `PI_MOM_ALLOWED_CHANNELS` | Comma-separated channel IDs |
