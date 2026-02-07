"""Agent runner: session management, system prompt, event handling."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pi.agent import Agent
from pi.ai import get_model
from pi.coding import AgentSession, AgentSessionConfig
from pi.coding.core.sessions import SessionManager
from pi.coding.core.resolver import ModelRegistry

from pi.mom import log
from pi.mom.context import MomSettingsManager, sync_log_to_session_manager
from pi.mom.log import LogContext
from pi.mom.sandbox import SandboxConfig, create_executor
from pi.mom.slack import ChannelInfo, SlackContext, UserInfo
from pi.mom.store import ChannelStore
from pi.mom.tools import create_mom_tools, set_upload_function

# Hardcoded model for now
_model = get_model("anthropic", "claude-sonnet-4-5")

_IMAGE_MIME_TYPES: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def _get_image_mime_type(filename: str) -> str | None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _IMAGE_MIME_TYPES.get(ext)


# ============================================================================
# Memory & Skills helpers
# ============================================================================


def _get_memory(channel_dir: str) -> str:
    parts: list[str] = []

    workspace_memory = os.path.join(channel_dir, "..", "MEMORY.md")
    if os.path.exists(workspace_memory):
        try:
            content = Path(workspace_memory).read_text("utf-8").strip()
            if content:
                parts.append(f"### Global Workspace Memory\n{content}")
        except Exception as exc:
            log.log_warning("Failed to read workspace memory", f"{workspace_memory}: {exc}")

    channel_memory = os.path.join(channel_dir, "MEMORY.md")
    if os.path.exists(channel_memory):
        try:
            content = Path(channel_memory).read_text("utf-8").strip()
            if content:
                parts.append(f"### Channel-Specific Memory\n{content}")
        except Exception as exc:
            log.log_warning("Failed to read channel memory", f"{channel_memory}: {exc}")

    return "\n\n".join(parts) if parts else "(no working memory yet)"


def _load_mom_skills(channel_dir: str, workspace_path: str) -> list[Any]:
    """Load skills from workspace and channel directories."""
    try:
        from pi.coding import loadSkillsFromDir  # type: ignore[attr-defined]
    except ImportError:
        return []

    skill_map: dict[str, Any] = {}
    host_workspace = os.path.join(channel_dir, "..")

    def translate(host_path: str) -> str:
        if host_path.startswith(host_workspace):
            return workspace_path + host_path[len(host_workspace):]
        return host_path

    ws_skills_dir = os.path.join(host_workspace, "skills")
    try:
        for skill in loadSkillsFromDir(dir=ws_skills_dir, source="workspace").skills:
            skill.file_path = translate(skill.file_path)
            skill.base_dir = translate(skill.base_dir)
            skill_map[skill.name] = skill
    except Exception:
        pass

    ch_skills_dir = os.path.join(channel_dir, "skills")
    try:
        for skill in loadSkillsFromDir(dir=ch_skills_dir, source="channel").skills:
            skill.file_path = translate(skill.file_path)
            skill.base_dir = translate(skill.base_dir)
            skill_map[skill.name] = skill
    except Exception:
        pass

    return list(skill_map.values())


def _format_skills_for_prompt(skills: list[Any]) -> str:
    try:
        from pi.coding import formatSkillsForPrompt  # type: ignore[attr-defined]
        return formatSkillsForPrompt(skills)
    except ImportError:
        if not skills:
            return "(no skills installed yet)"
        lines = []
        for s in skills:
            lines.append(f"- **{s.name}**: {getattr(s, 'description', '')}")
        return "\n".join(lines)


# ============================================================================
# System prompt builder
# ============================================================================


def _build_system_prompt(
    workspace_path: str,
    channel_id: str,
    memory: str,
    sandbox_config: SandboxConfig,
    channels: list[ChannelInfo],
    users: list[UserInfo],
    skills: list[Any],
) -> str:
    channel_path = f"{workspace_path}/{channel_id}"
    is_docker = sandbox_config.type == "docker"

    channel_mappings = (
        "\n".join(f"{c.id}\t#{c.name}" for c in channels) if channels else "(no channels loaded)"
    )
    user_mappings = (
        "\n".join(f"{u.id}\t@{u.user_name}\t{u.display_name}" for u in users)
        if users
        else "(no users loaded)"
    )

    try:
        local_tz = datetime.now().astimezone().tzinfo
        tz_name = str(local_tz)
    except Exception:
        tz_name = "UTC"

    if is_docker:
        env_desc = (
            "You are running inside a Docker container (Alpine Linux).\n"
            "- Bash working directory: / (use cd or absolute paths)\n"
            "- Install tools with: apk add <package>\n"
            "- Your changes persist across sessions"
        )
    else:
        env_desc = (
            "You are running directly on the host machine.\n"
            f"- Bash working directory: {os.getcwd()}\n"
            "- Be careful with system modifications"
        )

    skills_text = _format_skills_for_prompt(skills) if skills else "(no skills installed yet)"

    return f"""You are mom, a Slack bot assistant. Be concise. No emojis.

## Context
- For current date/time, use: date
- You have access to previous conversation context including tool results from prior turns.
- For older history beyond your context, search log.jsonl (contains user messages and your final responses, but not tool results).

## Slack Formatting (mrkdwn, NOT Markdown)
Bold: *text*, Italic: _text_, Code: `code`, Block: ```code```, Links: <url|text>
Do NOT use **double asterisks** or [markdown](links).

## Slack IDs
Channels: {channel_mappings}

Users: {user_mappings}

When mentioning users, use <@username> format (e.g., <@mario>).

## Environment
{env_desc}

## Workspace Layout
{workspace_path}/
├── MEMORY.md                    # Global memory (all channels)
├── skills/                      # Global CLI tools you create
└── {channel_id}/                # This channel
    ├── MEMORY.md                # Channel-specific memory
    ├── log.jsonl                # Message history (no tool results)
    ├── attachments/             # User-shared files
    ├── scratch/                 # Your working directory
    └── skills/                  # Channel-specific tools

## Skills (Custom CLI Tools)
You can create reusable CLI tools for recurring tasks (email, APIs, data processing, etc.).

### Creating Skills
Store in `{workspace_path}/skills/<name>/` (global) or `{channel_path}/skills/<name>/` (channel-specific).
Each skill directory needs a `SKILL.md` with YAML frontmatter:

```markdown
---
name: skill-name
description: Short description of what this skill does
---

# Skill Name

Usage instructions, examples, etc.
Scripts are in: {{baseDir}}/
```

`name` and `description` are required. Use `{{baseDir}}` as placeholder for the skill's directory path.

### Available Skills
{skills_text}

## Events
You can schedule events that wake you up at specific times or when external things happen. Events are JSON files in `{workspace_path}/events/`.

### Event Types

**Immediate** - Triggers as soon as harness sees the file. Use in scripts/webhooks to signal external events.
```json
{{"type": "immediate", "channelId": "{channel_id}", "text": "New GitHub issue opened"}}
```

**One-shot** - Triggers once at a specific time. Use for reminders.
```json
{{"type": "one-shot", "channelId": "{channel_id}", "text": "Remind Mario about dentist", "at": "2025-12-15T09:00:00+01:00"}}
```

**Periodic** - Triggers on a cron schedule. Use for recurring tasks.
```json
{{"type": "periodic", "channelId": "{channel_id}", "text": "Check inbox and summarize", "schedule": "0 9 * * 1-5", "timezone": "{tz_name}"}}
```

### Cron Format
`minute hour day-of-month month day-of-week`
- `0 9 * * *` = daily at 9:00
- `0 9 * * 1-5` = weekdays at 9:00
- `30 14 * * 1` = Mondays at 14:30
- `0 0 1 * *` = first of each month at midnight

### Timezones
All `at` timestamps must include offset (e.g., `+01:00`). Periodic events use IANA timezone names. The harness runs in {tz_name}. When users mention times without timezone, assume {tz_name}.

### Creating Events
Use unique filenames to avoid overwriting existing events. Include a timestamp or random suffix:
```bash
cat > {workspace_path}/events/dentist-reminder-$(date +%s).json << 'EOF'
{{"type": "one-shot", "channelId": "{channel_id}", "text": "Dentist tomorrow", "at": "2025-12-14T09:00:00+01:00"}}
EOF
```
Or check if file exists first before creating.

### Managing Events
- List: `ls {workspace_path}/events/`
- View: `cat {workspace_path}/events/foo.json`
- Delete/cancel: `rm {workspace_path}/events/foo.json`

### When Events Trigger
You receive a message like:
```
[EVENT:dentist-reminder.json:one-shot:2025-12-14T09:00:00+01:00] Dentist tomorrow
```
Immediate and one-shot events auto-delete after triggering. Periodic events persist until you delete them.

### Silent Completion
For periodic events where there's nothing to report, respond with just `[SILENT]` (no other text). This deletes the status message and posts nothing to Slack. Use this to avoid spamming the channel when periodic checks find nothing actionable.

### Debouncing
When writing programs that create immediate events (email watchers, webhook handlers, etc.), always debounce. If 50 emails arrive in a minute, don't create 50 immediate events. Instead collect events over a window and create ONE immediate event summarizing what happened, or just signal "new activity, check inbox" rather than per-item events. Or simpler: use a periodic event to check for new items every N minutes instead of immediate events.

### Limits
Maximum 5 events can be queued. Don't create excessive immediate or periodic events.

## Memory
Write to MEMORY.md files to persist context across conversations.
- Global ({workspace_path}/MEMORY.md): skills, preferences, project info
- Channel ({channel_path}/MEMORY.md): channel-specific decisions, ongoing work
Update when you learn something important or when asked to remember something.

### Current Memory
{memory}

## System Configuration Log
Maintain {workspace_path}/SYSTEM.md to log all environment modifications:
- Installed packages (apk add, npm install, pip install)
- Environment variables set
- Config files modified (~/.gitconfig, cron jobs, etc.)
- Skill dependencies installed

Update this file whenever you modify the environment. On fresh container, read it first to restore your setup.

## Log Queries (for older history)
Format: `{{"date":"...","ts":"...","user":"...","userName":"...","text":"...","isBot":false}}`
The log contains user messages and your final responses (not tool calls/results).
{"Install jq: apk add jq" if is_docker else ""}

```bash
# Recent messages
tail -30 log.jsonl | jq -c '{{date: .date[0:19], user: (.userName // .user), text}}'

# Search for specific topic
grep -i "topic" log.jsonl | jq -c '{{date: .date[0:19], user: (.userName // .user), text}}'

# Messages from specific user
grep '"userName":"mario"' log.jsonl | tail -20 | jq -c '{{date: .date[0:19], text}}'
```

## Tools
- bash: Run shell commands (primary tool). Install packages as needed.
- read: Read files
- write: Create/overwrite files
- edit: Surgical file edits
- attach: Share files to Slack

Each tool requires a "label" parameter (shown to user).
"""


# ============================================================================
# Helpers
# ============================================================================


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts = []
        for part in result["content"]:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    return json.dumps(result, default=str)


def _format_tool_args_for_slack(args: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in args.items():
        if key == "label":
            continue
        if key == "path" and isinstance(value, str):
            offset = args.get("offset")
            limit = args.get("limit")
            if offset is not None and limit is not None:
                lines.append(f"{value}:{offset}-{offset + limit}")
            else:
                lines.append(value)
            continue
        if key in ("offset", "limit"):
            continue
        if isinstance(value, str):
            lines.append(value)
        else:
            lines.append(json.dumps(value))
    return "\n".join(lines)


def _translate_to_host_path(
    container_path: str,
    channel_dir: str,
    workspace_path: str,
    channel_id: str,
) -> str:
    if workspace_path == "/workspace":
        prefix = f"/workspace/{channel_id}/"
        if container_path.startswith(prefix):
            return os.path.join(channel_dir, container_path[len(prefix):])
        if container_path.startswith("/workspace/"):
            return os.path.join(channel_dir, "..", container_path[len("/workspace/"):])
    return container_path


_SLACK_MAX_LENGTH = 40000


def _split_for_slack(text: str) -> list[str]:
    if len(text) <= _SLACK_MAX_LENGTH:
        return [text]
    parts: list[str] = []
    remaining = text
    part_num = 1
    while remaining:
        chunk = remaining[: _SLACK_MAX_LENGTH - 50]
        remaining = remaining[_SLACK_MAX_LENGTH - 50:]
        suffix = f"\n_(continued {part_num}...)_" if remaining else ""
        parts.append(chunk + suffix)
        part_num += 1
    return parts


# ============================================================================
# AgentRunner
# ============================================================================


@dataclass
class PendingMessage:
    user_name: str
    text: str
    attachments: list[dict[str, str]]
    timestamp: float


class AgentRunner(Protocol):
    async def run(
        self,
        ctx: SlackContext,
        store: ChannelStore,
        pending_messages: list[PendingMessage] | None = None,
    ) -> dict[str, Any]: ...

    def abort(self) -> None: ...


# Cache runners per channel
_channel_runners: dict[str, AgentRunner] = {}


def get_or_create_runner(
    sandbox_config: SandboxConfig, channel_id: str, channel_dir: str
) -> AgentRunner:
    existing = _channel_runners.get(channel_id)
    if existing is not None:
        return existing
    runner = _create_runner(sandbox_config, channel_id, channel_dir)
    _channel_runners[channel_id] = runner
    return runner


def _create_runner(
    sandbox_config: SandboxConfig, channel_id: str, channel_dir: str
) -> AgentRunner:
    executor = create_executor(sandbox_config)
    workspace_path = executor.get_workspace_path(
        channel_dir.replace(f"/{channel_id}", "")
    )

    tools = create_mom_tools(executor, workspace_path)

    memory = _get_memory(channel_dir)
    skills = _load_mom_skills(channel_dir, workspace_path)
    system_prompt = _build_system_prompt(
        workspace_path, channel_id, memory, sandbox_config, [], [], skills
    )

    context_file = os.path.join(channel_dir, "context.jsonl")
    session_manager = SessionManager.open(context_file, channel_dir)
    settings_manager = MomSettingsManager(os.path.join(channel_dir, ".."))

    # ModelRegistry – auth stored outside workspace
    home = os.path.expanduser("~")
    try:
        from pi.coding.core.auth import AuthStorage
        auth_storage = AuthStorage(os.path.join(home, ".pi", "mom", "auth.json"))
        model_registry = ModelRegistry(auth_storage)
    except ImportError:
        model_registry = None  # type: ignore[assignment]
        auth_storage = None

    async def _get_api_key() -> str:
        if auth_storage is not None:
            key = await auth_storage.get_api_key("anthropic")
            if key:
                return key
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
        raise RuntimeError(
            "No API key found for anthropic.\n"
            "Set ANTHROPIC_API_KEY environment variable, or link auth.json from "
            + os.path.join(home, ".pi", "mom", "auth.json")
        )

    try:
        from pi.coding.core.session import convertToLlm  # type: ignore[attr-defined]
        convert_fn = convertToLlm
    except ImportError:
        convert_fn = None

    agent = Agent(
        initial_state={
            "system_prompt": system_prompt,
            "model": _model,
            "thinking_level": "off",
            "tools": tools,
        },
        convert_to_llm=convert_fn,
        get_api_key=_get_api_key,
    )

    loaded = session_manager.build_session_context()
    if loaded.messages:
        agent.replace_messages(loaded.messages)
        log.log_info(
            f"[{channel_id}] Loaded {len(loaded.messages)} messages from context.jsonl"
        )

    # Build AgentSession
    try:
        from pi.coding.core.extensions import create_extension_runtime
        resource_loader_kwargs: dict[str, Any] = {}
    except ImportError:
        pass

    base_tools_override = {t.name: t for t in tools}

    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,  # type: ignore[arg-type]
            cwd=os.getcwd(),
            model_registry=model_registry,
            base_tools_override=base_tools_override,
        )
    )

    # ── Mutable per-run state ────────────────────────────────────────

    class _RunState:
        ctx: SlackContext | None = None
        log_ctx: LogContext | None = None
        pending_tools: dict[str, dict[str, Any]] = {}
        total_usage: dict[str, Any] = {}
        stop_reason: str = "stop"
        error_message: str | None = None
        queue_chain: asyncio.Future[None] | None = None
        _queue_lock: asyncio.Lock = asyncio.Lock()

    run_state = _RunState()

    # ── Slack message queue (serialized) ─────────────────────────────

    class _RunQueue:
        def __init__(self, ctx: SlackContext) -> None:
            self._ctx = ctx
            self._chain: asyncio.Future[None] = asyncio.get_event_loop().create_future()
            self._chain.set_result(None)

        def enqueue(self, fn: Any, error_context: str) -> None:
            prev = self._chain
            loop = asyncio.get_event_loop()
            fut: asyncio.Future[None] = loop.create_future()

            async def _run() -> None:
                await prev
                try:
                    await fn()
                except Exception as exc:
                    log.log_warning(f"Slack API error ({error_context})", str(exc))
                    try:
                        await self._ctx.respond_in_thread(f"_Error: {exc}_")
                    except Exception:
                        pass
                fut.set_result(None)

            asyncio.ensure_future(_run())
            self._chain = fut

        def enqueue_message(
            self,
            text: str,
            target: str,
            error_context: str,
            do_log: bool = True,
        ) -> None:
            for part in _split_for_slack(text):
                if target == "main":
                    self.enqueue(lambda p=part, dl=do_log: self._ctx.respond(p, dl), error_context)
                else:
                    self.enqueue(lambda p=part: self._ctx.respond_in_thread(p), error_context)

        async def wait(self) -> None:
            await self._chain

    run_queue: _RunQueue | None = None

    # ── Subscribe to events once ─────────────────────────────────────

    def _on_event(event: Any) -> None:
        nonlocal run_queue
        if run_state.ctx is None or run_state.log_ctx is None or run_queue is None:
            return

        ctx = run_state.ctx
        log_ctx = run_state.log_ctx
        q = run_queue

        etype = getattr(event, "type", None)

        if etype == "tool_execution_start":
            args = getattr(event, "args", {}) or {}
            label = args.get("label") or getattr(event, "tool_name", "")
            tool_call_id = getattr(event, "tool_call_id", "")
            tool_name = getattr(event, "tool_name", "")

            run_state.pending_tools[tool_call_id] = {
                "toolName": tool_name,
                "args": args,
                "startTime": time.time(),
            }

            log.log_tool_start(log_ctx, tool_name, label, args)
            q.enqueue(lambda: ctx.respond(f"_→ {label}_", False), "tool label")

        elif etype == "tool_execution_end":
            tool_call_id = getattr(event, "tool_call_id", "")
            tool_name = getattr(event, "tool_name", "")
            result = getattr(event, "result", None)
            is_error = getattr(event, "is_error", False)

            result_str = _extract_tool_result_text(result)
            pending = run_state.pending_tools.pop(tool_call_id, None)
            duration_ms = (time.time() - pending["startTime"]) * 1000 if pending else 0

            if is_error:
                log.log_tool_error(log_ctx, tool_name, duration_ms, result_str)
            else:
                log.log_tool_success(log_ctx, tool_name, duration_ms, result_str)

            label = pending["args"].get("label") if pending and pending.get("args") else None
            args_formatted = (
                _format_tool_args_for_slack(pending["args"])
                if pending
                else "(args not found)"
            )
            duration_s = f"{duration_ms / 1000:.1f}"
            thread_msg = f"*{'✗' if is_error else '✓'} {tool_name}*"
            if label:
                thread_msg += f": {label}"
            thread_msg += f" ({duration_s}s)\n"
            if args_formatted:
                thread_msg += f"```\n{args_formatted}\n```\n"
            thread_msg += f"*Result:*\n```\n{result_str}\n```"

            q.enqueue_message(thread_msg, "thread", "tool result thread", False)

            if is_error:
                q.enqueue(
                    lambda: ctx.respond(f"_Error: {_truncate(result_str, 200)}_", False),
                    "tool error",
                )

        elif etype == "message_start":
            msg = getattr(event, "message", None)
            if msg and getattr(msg, "role", None) == "assistant":
                log.log_response_start(log_ctx)

        elif etype == "message_end":
            msg = getattr(event, "message", None)
            if msg and getattr(msg, "role", None) == "assistant":
                stop = getattr(msg, "stop_reason", None) or getattr(msg, "stopReason", None)
                if stop:
                    run_state.stop_reason = stop
                err = getattr(msg, "error_message", None) or getattr(msg, "errorMessage", None)
                if err:
                    run_state.error_message = err

                usage = getattr(msg, "usage", None)
                if usage:
                    u = run_state.total_usage
                    u["input"] = u.get("input", 0) + getattr(usage, "input", 0)
                    u["output"] = u.get("output", 0) + getattr(usage, "output", 0)
                    u["cacheRead"] = u.get("cacheRead", 0) + getattr(usage, "cache_read", getattr(usage, "cacheRead", 0))
                    u["cacheWrite"] = u.get("cacheWrite", 0) + getattr(usage, "cache_write", getattr(usage, "cacheWrite", 0))
                    cost = getattr(usage, "cost", None) or {}
                    if isinstance(cost, dict):
                        uc = u.setdefault("cost", {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0})
                        uc["input"] += cost.get("input", 0)
                        uc["output"] += cost.get("output", 0)
                        uc["cacheRead"] += cost.get("cacheRead", 0)
                        uc["cacheWrite"] += cost.get("cacheWrite", 0)
                        uc["total"] += cost.get("total", 0)

                content = getattr(msg, "content", []) or []
                thinking_parts: list[str] = []
                text_parts: list[str] = []
                for part in content:
                    ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                    if ptype == "thinking":
                        thinking_parts.append(part.get("thinking", "") if isinstance(part, dict) else getattr(part, "thinking", ""))
                    elif ptype == "text":
                        text_parts.append(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", ""))

                text = "\n".join(text_parts)

                for thinking in thinking_parts:
                    log.log_thinking(log_ctx, thinking)
                    q.enqueue_message(f"_{thinking}_", "main", "thinking main")
                    q.enqueue_message(f"_{thinking}_", "thread", "thinking thread", False)

                if text.strip():
                    log.log_response(log_ctx, text)
                    q.enqueue_message(text, "main", "response main")
                    q.enqueue_message(text, "thread", "response thread", False)

        elif etype == "auto_compaction_start":
            reason = getattr(event, "reason", "")
            log.log_info(f"Auto-compaction started (reason: {reason})")
            q.enqueue(lambda: ctx.respond("_Compacting context..._", False), "compaction start")

        elif etype == "auto_compaction_end":
            result = getattr(event, "result", None)
            if result:
                tokens = getattr(result, "tokens_before", getattr(result, "tokensBefore", 0))
                log.log_info(f"Auto-compaction complete: {tokens} tokens compacted")
            elif getattr(event, "aborted", False):
                log.log_info("Auto-compaction aborted")

        elif etype == "auto_retry_start":
            attempt = getattr(event, "attempt", 0)
            max_attempts = getattr(event, "max_attempts", getattr(event, "maxAttempts", 0))
            err_msg = getattr(event, "error_message", getattr(event, "errorMessage", ""))
            log.log_warning(f"Retrying ({attempt}/{max_attempts})", err_msg)
            q.enqueue(
                lambda: ctx.respond(f"_Retrying ({attempt}/{max_attempts})..._", False),
                "retry",
            )

    session.subscribe(_on_event)

    # ── Runner implementation ────────────────────────────────────────

    class _Runner:
        async def run(
            self,
            ctx: SlackContext,
            store: ChannelStore,
            pending_messages: list[PendingMessage] | None = None,
        ) -> dict[str, Any]:
            nonlocal run_queue

            os.makedirs(channel_dir, exist_ok=True)

            synced = sync_log_to_session_manager(
                session_manager, channel_dir, ctx.message.ts
            )
            if synced > 0:
                log.log_info(f"[{channel_id}] Synced {synced} messages from log.jsonl")

            reloaded = session_manager.build_session_context()
            if reloaded.messages:
                agent.replace_messages(reloaded.messages)
                log.log_info(
                    f"[{channel_id}] Reloaded {len(reloaded.messages)} messages from context"
                )

            mem = _get_memory(channel_dir)
            sk = _load_mom_skills(channel_dir, workspace_path)
            sys_prompt = _build_system_prompt(
                workspace_path,
                channel_id,
                mem,
                sandbox_config,
                ctx.channels,
                ctx.users,
                sk,
            )
            session.agent.set_system_prompt(sys_prompt)

            set_upload_function(
                lambda fp, title=None: ctx.upload_file(
                    _translate_to_host_path(fp, channel_dir, workspace_path, channel_id),
                    title,
                )
            )

            # Reset per-run state
            run_state.ctx = ctx
            run_state.log_ctx = LogContext(
                channel_id=ctx.message.channel,
                user_name=ctx.message.user_name,
                channel_name=ctx.channel_name,
            )
            run_state.pending_tools = {}
            run_state.total_usage = {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            }
            run_state.stop_reason = "stop"
            run_state.error_message = None

            run_queue = _RunQueue(ctx)

            log.log_info(f"Context sizes - system: {len(sys_prompt)} chars, memory: {len(mem)} chars")
            log.log_info(f"Channels: {len(ctx.channels)}, Users: {len(ctx.users)}")

            # Build user message with timestamp
            now = datetime.now().astimezone()
            ts_str = now.strftime("%Y-%m-%d %H:%M:%S%z")
            # Insert colon in timezone offset: +0100 -> +01:00
            if len(ts_str) > 5 and (ts_str[-5] == "+" or ts_str[-5] == "-"):
                ts_str = ts_str[:-2] + ":" + ts_str[-2:]

            user_message = f"[{ts_str}] [{ctx.message.user_name or 'unknown'}]: {ctx.message.text}"

            image_attachments: list[dict[str, Any]] = []
            non_image_paths: list[str] = []

            for a in ctx.message.attachments or []:
                local = a.get("local", "")
                full_path = f"{workspace_path}/{local}"
                mime_type = _get_image_mime_type(local)

                if mime_type and os.path.exists(full_path):
                    try:
                        data = base64.b64encode(Path(full_path).read_bytes()).decode("ascii")
                        image_attachments.append(
                            {"type": "image", "mimeType": mime_type, "data": data}
                        )
                    except Exception:
                        non_image_paths.append(full_path)
                else:
                    non_image_paths.append(full_path)

            if non_image_paths:
                user_message += (
                    "\n\n<slack_attachments>\n"
                    + "\n".join(non_image_paths)
                    + "\n</slack_attachments>"
                )

            # Debug: write context to last_prompt.jsonl
            debug_ctx = {
                "systemPrompt": sys_prompt,
                "messages": [str(m) for m in session.messages],
                "newUserMessage": user_message,
                "imageAttachmentCount": len(image_attachments),
            }
            with open(
                os.path.join(channel_dir, "last_prompt.jsonl"), "w", encoding="utf-8"
            ) as fh:
                json.dump(debug_ctx, fh, indent=2)

            prompt_kwargs: dict[str, Any] = {}
            if image_attachments:
                prompt_kwargs["images"] = image_attachments

            await session.prompt(user_message, **prompt_kwargs)
            await run_queue.wait()

            # Handle error
            if run_state.stop_reason == "error" and run_state.error_message:
                try:
                    await ctx.replace_message("_Sorry, something went wrong_")
                    await ctx.respond_in_thread(f"_Error: {run_state.error_message}_")
                except Exception as exc:
                    log.log_warning("Failed to post error message", str(exc))
            else:
                messages = session.messages
                last_assistant = None
                for m in reversed(messages):
                    if getattr(m, "role", None) == "assistant":
                        last_assistant = m
                        break

                final_text = ""
                if last_assistant:
                    content = getattr(last_assistant, "content", []) or []
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
                        elif hasattr(c, "type") and c.type == "text":
                            parts.append(getattr(c, "text", ""))
                    final_text = "\n".join(parts)

                if final_text.strip() == "[SILENT]" or final_text.strip().startswith("[SILENT]"):
                    try:
                        await ctx.delete_message()
                        log.log_info("Silent response - deleted message and thread")
                    except Exception as exc:
                        log.log_warning("Failed to delete message for silent response", str(exc))
                elif final_text.strip():
                    try:
                        main_text = (
                            final_text[:_SLACK_MAX_LENGTH - 50] + "\n\n_(see thread for full response)_"
                            if len(final_text) > _SLACK_MAX_LENGTH
                            else final_text
                        )
                        await ctx.replace_message(main_text)
                    except Exception as exc:
                        log.log_warning("Failed to replace message with final text", str(exc))

            # Usage summary
            cost = run_state.total_usage.get("cost", {})
            if cost.get("total", 0) > 0:
                msgs = session.messages
                last_asst = None
                for m in reversed(msgs):
                    if getattr(m, "role", None) == "assistant":
                        sr = getattr(m, "stop_reason", None) or getattr(m, "stopReason", None)
                        if sr != "aborted":
                            last_asst = m
                            break

                context_tokens = 0
                if last_asst:
                    u = getattr(last_asst, "usage", None)
                    if u:
                        context_tokens = (
                            getattr(u, "input", 0)
                            + getattr(u, "output", 0)
                            + getattr(u, "cache_read", getattr(u, "cacheRead", 0))
                            + getattr(u, "cache_write", getattr(u, "cacheWrite", 0))
                        )
                context_window = getattr(_model, "context_window", None) or getattr(_model, "contextWindow", None) or 200000

                summary = log.log_usage_summary(
                    run_state.log_ctx, run_state.total_usage, context_tokens, context_window  # type: ignore[arg-type]
                )
                run_queue.enqueue(lambda: ctx.respond_in_thread(summary), "usage summary")
                await run_queue.wait()

            # Clear run state
            run_state.ctx = None
            run_state.log_ctx = None
            run_queue = None

            return {
                "stopReason": run_state.stop_reason,
                "errorMessage": run_state.error_message,
            }

        def abort(self) -> None:
            session.abort()

    return _Runner()  # type: ignore[return-value]
