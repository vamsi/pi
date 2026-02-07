# pi-agent

Stateful agent runtime that orchestrates LLM calls, tool execution, steering, and follow-up message processing. Built on top of `pi-ai` for the LLM layer.

The agent loop handles the mechanics of multi-turn conversations -- you supply the model, tools, and message conversion logic. The loop handles streaming, tool dispatch, retry, and message accumulation.

## Core concepts

**Agent** -- Stateful wrapper that holds the model, system prompt, tools, messages, and thinking level. Provides `prompt()`, `continue_()`, `abort()`, and event subscription.

**Agent loop** -- The inner engine (`agent_loop`, `agent_loop_continue`). Runs turns in a while-loop: stream an LLM response, execute any tool calls, check for steering/follow-up messages, repeat until the model stops calling tools.

**Steering** -- Messages injected mid-execution. If a steering message arrives while tools are running, remaining tool calls are skipped and the steering message is appended to context.

**Follow-ups** -- Messages queued for after the current run completes. The loop picks them up and starts a new turn automatically.

## Usage

### Minimal agent with tools

```python
import asyncio
from pi.ai import get_model
from pi.ai.types import TextContent
from pi.agent import Agent, AgentTool, AgentToolResult

async def execute_greet(tool_call_id, arguments, cancel_event, on_update):
    name = arguments.get("name", "world")
    return AgentToolResult(
        content=[TextContent(text=f"Hello, {name}!")]
    )

async def main():
    agent = Agent()
    agent.set_model(get_model("anthropic", "claude-sonnet-4-5"))
    agent.set_system_prompt("You have a greet tool. Use it when asked to greet someone.")
    agent.set_tools([
        AgentTool(
            name="greet",
            description="Greet a person by name.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person to greet"}
                },
                "required": ["name"],
            },
            execute=execute_greet,
        ),
    ])

    # Subscribe to events
    def on_event(event):
        if event.type == "message_update":
            ae = event.assistant_message_event
            if ae.type == "text_delta":
                print(ae.delta, end="", flush=True)
        elif event.type == "tool_execution_end":
            print(f"\n[tool result: {event.result.content[0].text}]")
        elif event.type == "agent_end":
            print("\n--- done ---")

    agent.subscribe(on_event)

    await agent.prompt("Please greet Alice.")

asyncio.run(main())
```

### Using the loop directly (without Agent)

The `Agent` class is a convenience wrapper. For full control, use `agent_loop` directly:

```python
import asyncio
from pi.ai import get_model
from pi.ai.types import UserMessage
from pi.agent import (
    agent_loop,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
)

async def main():
    model = get_model("anthropic", "claude-sonnet-4-5")

    context = AgentContext(
        system_prompt="Be helpful.",
        messages=[],
        tools=[],
    )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [m for m in msgs if m.role in ("user", "assistant", "tool_result")],
        reasoning="medium",
    )

    prompts = [UserMessage(role="user", content="What is 2+2?", timestamp=0)]

    event_stream = agent_loop(prompts, context, config)

    async for event in event_stream:
        if event.type == "message_update":
            ae = event.assistant_message_event
            if ae.type == "text_delta":
                print(ae.delta, end="")
        elif event.type == "agent_end":
            print()

asyncio.run(main())
```

### Thinking/reasoning

```python
agent.set_thinking_level("high")
await agent.prompt("Solve this step by step: ...")
```

The agent passes the thinking level through to `stream_simple()`, which maps it to the appropriate provider-specific format (token budgets for Anthropic, effort levels for OpenAI, thinking levels for Google).

Valid levels: `"off"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`.

### Steering and follow-ups

```python
# Queue a message to inject mid-execution
agent.steer(UserMessage(role="user", content="Actually, stop and do this instead.", timestamp=0))

# Queue a message for after the current run finishes
agent.follow_up(UserMessage(role="user", content="Now summarize what you did.", timestamp=0))
```

### Custom stream function

Override how the agent calls the LLM:

```python
from pi.ai.stream import stream_simple

async def my_stream_fn(model, context, options):
    # Add logging, modify context, etc.
    print(f"Calling {model.name} with {len(context.messages)} messages")
    return stream_simple(model, context, options)

agent = Agent(stream_fn=my_stream_fn)
```

### Custom message conversion

The agent loop works with `AgentMessage` (which is `Message` by default). If your application uses custom message types, provide a converter:

```python
def convert_to_llm(messages):
    """Strip application-specific fields before sending to LLM."""
    return [
        msg for msg in messages
        if msg.role in ("user", "assistant", "tool_result")
    ]

agent = Agent(convert_to_llm=convert_to_llm)
```

## Event types

The agent loop emits these events via `agent.subscribe(fn)` or by iterating the stream from `agent_loop()`:

| Event | When |
|-------|------|
| `AgentStartEvent` | Agent loop begins |
| `TurnStartEvent` | New turn (LLM call + tool execution) begins |
| `MessageStartEvent` | A message (user, assistant, or tool result) starts |
| `MessageUpdateEvent` | Streaming update with inner `assistant_message_event` |
| `MessageEndEvent` | Message complete |
| `ToolExecutionStartEvent` | Tool execution begins (has `tool_call_id`, `tool_name`, `args`) |
| `ToolExecutionUpdateEvent` | Partial tool result (for long-running tools) |
| `ToolExecutionEndEvent` | Tool execution complete (has `result`, `is_error`) |
| `TurnEndEvent` | Turn complete (has final `message` and `tool_results`) |
| `AgentEndEvent` | Agent loop finished (has all `messages` produced) |

### MessageUpdateEvent detail

`MessageUpdateEvent.assistant_message_event` contains the raw `pi-ai` streaming event (`TextDeltaEvent`, `ThinkingDeltaEvent`, `ToolCallEndEvent`, etc.). This lets you render streaming output without duplicating the event type hierarchy.

## Tool interface

```python
@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]     # JSON Schema
    label: str = ""                # Display label (optional)
    execute: Callable | None       # async (tool_call_id, arguments, cancel_event, on_update) -> AgentToolResult
```

The `execute` function signature:

```python
async def my_tool(
    tool_call_id: str,
    arguments: dict[str, Any],
    cancel_event: asyncio.Event | None,
    on_update: Callable[[AgentToolResult], None],
) -> AgentToolResult:
    ...
```

- `cancel_event` -- Set when the user aborts. Check `cancel_event.is_set()` in long-running tools.
- `on_update` -- Call with partial results to emit `ToolExecutionUpdateEvent` during execution.

## File structure

```
src/pi/agent/
    __init__.py     Public exports
    agent.py        Agent class (stateful wrapper)
    loop.py         agent_loop, agent_loop_continue, tool execution
    types.py        AgentTool, AgentState, AgentContext, AgentLoopConfig, events
```

## Architecture notes

- The agent loop transforms `AgentMessage[]` to `Message[]` only at the LLM call boundary (via `convert_to_llm`). This means your application can carry arbitrary metadata on messages without polluting the LLM context.
- Tool arguments are validated against JSON Schema before execution. Invalid arguments produce an error result sent back to the LLM.
- The loop respects the `cancel_event` throughout: before LLM calls, during streaming, and between tool calls.
- When a steering message arrives during tool execution, remaining tool calls are skipped (with a "Skipped due to queued user message" error result), and the steering message is appended to context for the next turn.
