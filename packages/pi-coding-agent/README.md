# pi-coding-agent

CLI coding agent with file tools, session persistence, context compaction, and an extension system. This is the application layer that wires `pi-ai` (LLM streaming), `pi-agent` (agent loop), and `pi-tui` (terminal UI) into a working coding assistant.

## Running

```bash
# Interactive mode
pi

# With a specific model
pi -m claude-opus-4-6
pi -m openai/gpt-5.2
pi -m google/gemini-3-pro-preview

# Print mode (non-interactive, single prompt)
pi --print "Explain this error: ..."

# With reasoning
pi --thinking high

# Continue the most recent session
pi --continue

# Resume a specific session
pi --session ~/.pi/sessions/abc123.jsonl
```

## Architecture

```
CLI (cli.py)
  |
AgentSession (session/)
  |-- Agent (pi-agent)
  |-- SessionManager (sessions.py)
  |-- SettingsManager (settings.py)
  |-- ExtensionRunner (extensions/)
  |-- ModelRegistry (resolver.py)
  |-- SystemPrompt (prompt.py)
  |-- Compaction (compaction/)
  |
Tools (tools/)
  |-- bash, read, write, edit, grep, find, ls
```

### AgentSession

The central orchestrator. It wires together:

- An `Agent` instance (from `pi-agent`) with model, tools, and system prompt
- Session persistence (JSONL files with full message history)
- Extension lifecycle (load, execute hooks, hot-reload)
- Model resolution (fuzzy matching, provider/model splitting)
- Context compaction (summarize old messages when context window fills up)

### Tools

Each tool implements the `AgentTool` interface from `pi-agent`:

| Tool | Description |
|------|-------------|
| `bash` | Execute shell commands with timeout and output capture |
| `read` | Read file contents with line range support |
| `write` | Write or create files |
| `edit` | Search-and-replace within files |
| `grep` | Search file contents with regex |
| `find` | Find files by glob pattern |
| `ls` | List directory contents |

Tools are registered during session setup. The agent decides which tools to call based on the system prompt and user request.

### Extensions

Extensions are Python modules loaded from a directory. They can hook into the agent lifecycle:

```python
# ~/.pi/extensions/my_extension.py
def on_agent_start(event, session):
    """Called when the agent starts processing."""
    pass

def on_turn_end(event, session):
    """Called after each turn (LLM response + tool execution)."""
    pass

def on_agent_end(event, session):
    """Called when the agent finishes."""
    pass
```

Load extensions:

```bash
pi --extension ./my_extension.py
```

### Session persistence

Sessions are stored as JSONL files. Each line is a message (user, assistant, tool_result) serialized as JSON. Sessions can be resumed with `--continue` or `--session <path>`.

### Context compaction

When the conversation approaches the model's context window limit, older messages are summarized into a compact form. This preserves the essential context while freeing space for new messages. The compaction module handles:

- Detecting when compaction is needed
- Summarizing message blocks
- Replacing original messages with summaries
- Preserving tool call/result pairs that are still relevant

### Model resolution

The resolver supports flexible model specification:

```bash
pi -m claude-opus      # Fuzzy matches to claude-opus-4-6
pi -m gpt-5            # Fuzzy matches to gpt-5.2
pi -m openai/gpt-4.1   # Explicit provider/model
pi -m google/gemini-3-pro-preview
```

## File structure

```
src/pi/coding/
    cli.py                   CLI entry point
    core/
        prompt.py            System prompt construction
        resolver.py          Model registry with fuzzy matching
        sessions.py          Session file management
        settings.py          User settings (JSON config)
        compaction/
            compact.py       Compaction orchestration
            summarize.py     Message summarization
            utils.py         Compaction utilities
        extensions/
            loader.py        Extension discovery and loading
            runner.py        Extension lifecycle execution
            types.py         Extension hook types
            wrapper.py       Extension error isolation
        session/
            __init__.py      AgentSession orchestrator
            compaction.py    Session compaction integration
            events.py        Session-level events
            models.py        Model switching within session
            navigation.py    Message history navigation
        tools/
            __init__.py      Tool registration
            bash.py          Shell command execution
            edit.py          File editing (search and replace)
            find.py          File finding (glob)
            grep.py          Content search (regex)
            ls.py            Directory listing
            read.py          File reading
            write.py         File writing
```
