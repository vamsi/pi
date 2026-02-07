# pi-ai

Unified LLM streaming API across 9 provider backends. Every provider exposes the same interface: pass a `Model` and `Context`, get back an `AssistantMessageEventStream` you can async-iterate over. Provider differences (message formats, auth, SSE parsing, retry logic) are handled internally.

## Supported providers

| API identifier | SDK | Provider examples |
|----------------|-----|-------------------|
| `anthropic-messages` | `anthropic` | Anthropic |
| `openai-completions` | `openai` | OpenAI, GitHub Copilot, OpenRouter, Groq, xAI, Cerebras, Mistral, HuggingFace |
| `openai-responses` | `openai` | OpenAI (GPT-5+), GitHub Copilot (GPT-5+) |
| `openai-codex-responses` | `httpx` (raw HTTP) | OpenAI Codex (ChatGPT backend) |
| `azure-openai-responses` | `openai` (Azure) | Azure OpenAI |
| `google-generative-ai` | `google-genai` | Google (Gemini via API key) |
| `google-vertex` | `google-genai` | Google Vertex AI (ADC auth) |
| `google-gemini-cli` | `httpx` (raw HTTP) | Google Cloud Code Assist, Antigravity |
| `bedrock-converse-stream` | `boto3` | Amazon Bedrock |

65 models are pre-registered across 6 provider families (Anthropic, OpenAI, Google, Bedrock, Codex, GitHub Copilot).

## Installation

```bash
# Core (includes anthropic, openai, httpx)
pip install pi-ai

# With optional provider SDKs
pip install "pi-ai[google]"     # adds google-genai
pip install "pi-ai[bedrock]"    # adds boto3
pip install "pi-ai[mistral]"    # adds mistralai
```

## Usage

### One-shot completion

```python
import asyncio
from pi.ai import get_model, complete_simple
from pi.ai.types import Context, UserMessage, SimpleStreamOptions

async def main():
    model = get_model("anthropic", "claude-sonnet-4-5")
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            UserMessage(
                role="user",
                content="Explain the builder pattern in three sentences.",
                timestamp=0,
            )
        ],
    )

    result = await complete_simple(
        model,
        context,
        SimpleStreamOptions(reasoning="medium"),
    )

    for block in result.content:
        if block.type == "text":
            print(block.text)

    print(f"Tokens: {result.usage.input} in, {result.usage.output} out")
    print(f"Cost: ${result.usage.cost.total:.4f}")

asyncio.run(main())
```

### Streaming with token-level deltas

```python
import asyncio
from pi.ai import get_model, stream_simple
from pi.ai.types import Context, UserMessage, SimpleStreamOptions

async def main():
    model = get_model("openai", "gpt-4.1")
    context = Context(
        messages=[UserMessage(role="user", content="Write a haiku.", timestamp=0)]
    )

    event_stream = stream_simple(model, context, SimpleStreamOptions())
    async for event in event_stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "done":
            print()

asyncio.run(main())
```

### Streaming with tool calls

```python
import asyncio
from pi.ai import get_model, stream_simple
from pi.ai.types import Context, UserMessage, Tool, SimpleStreamOptions

async def main():
    model = get_model("anthropic", "claude-sonnet-4-5")
    context = Context(
        system_prompt="Use tools when appropriate.",
        messages=[
            UserMessage(role="user", content="What is the weather in Tokyo?", timestamp=0)
        ],
        tools=[
            Tool(
                name="get_weather",
                description="Get current weather for a city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            )
        ],
    )

    event_stream = stream_simple(model, context, SimpleStreamOptions())
    async for event in event_stream:
        if event.type == "text_delta":
            print(event.delta, end="")
        elif event.type == "toolcall_end":
            tc = event.tool_call
            print(f"\n[Tool call: {tc.name}({tc.arguments})]")
        elif event.type == "done":
            print(f"\nStop reason: {event.reason}")

asyncio.run(main())
```

### Using thinking/reasoning

```python
from pi.ai import get_model, stream_simple
from pi.ai.types import Context, UserMessage, SimpleStreamOptions

model = get_model("anthropic", "claude-opus-4-6")
options = SimpleStreamOptions(reasoning="high")

event_stream = stream_simple(model, context, options)
async for event in event_stream:
    if event.type == "thinking_delta":
        # Model's chain-of-thought (visible with reasoning-capable models)
        print(f"[thinking] {event.delta}", end="")
    elif event.type == "text_delta":
        print(event.delta, end="")
```

Reasoning levels: `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (xhigh only supported on select models like Claude Opus 4.6 and GPT-5.2+).

### Switching providers at runtime

The same `Context` works across all providers. Switch models without changing application code:

```python
from pi.ai import get_model, complete_simple

# Same context, different providers
for provider, model_id in [
    ("anthropic", "claude-sonnet-4-5"),
    ("openai", "gpt-4.1"),
    ("google", "gemini-2.5-pro"),
]:
    model = get_model(provider, model_id)
    result = await complete_simple(model, context)
    print(f"{model.name}: {result.usage.cost.total:.4f}")
```

### Provider-specific options (advanced)

When you need provider-specific control beyond what `SimpleStreamOptions` offers, use the provider's stream function directly:

```python
from pi.ai.providers.anthropic import stream_anthropic, AnthropicOptions

event_stream = stream_anthropic(
    model,
    context,
    AnthropicOptions(
        temperature=0.7,
        max_tokens=4096,
        thinking_level="high",
        thinking_budget=8192,
        cache_retention="long",
    ),
)
```

Each provider has its own options class with provider-specific fields. See the docstrings in `providers/*.py`.

## Core types

### Model

```python
class Model(BaseModel):
    id: str                    # "claude-sonnet-4-5", "gpt-4.1", etc.
    name: str                  # Human-readable name
    api: str                   # API protocol: "anthropic-messages", "openai-responses", etc.
    provider: str              # Provider family: "anthropic", "openai", "google", etc.
    base_url: str              # API endpoint URL
    reasoning: bool            # Whether the model supports chain-of-thought
    input: list[str]           # Supported input types: ["text"] or ["text", "image"]
    cost: ModelCost            # Per-million-token pricing
    context_window: int        # Maximum input context size
    max_tokens: int            # Maximum output tokens
    headers: dict | None       # Default headers (used by GitHub Copilot)
    compat: ... | None         # Provider-specific compatibility flags
```

### Context

```python
class Context(BaseModel):
    system_prompt: str | None
    messages: list[Message]     # UserMessage | AssistantMessage | ToolResultMessage
    tools: list[Tool] | None
```

### Event stream

Every `stream()` or `stream_simple()` call returns an `AssistantMessageEventStream`, which is an async iterator yielding typed events:

| Event | Fields | When |
|-------|--------|------|
| `StartEvent` | `partial` | Stream begins |
| `TextStartEvent` | `content_index`, `partial` | New text block |
| `TextDeltaEvent` | `content_index`, `delta`, `partial` | Text token |
| `TextEndEvent` | `content_index`, `content`, `partial` | Text block complete |
| `ThinkingStartEvent` | `content_index`, `partial` | Thinking block begins |
| `ThinkingDeltaEvent` | `content_index`, `delta`, `partial` | Thinking token |
| `ThinkingEndEvent` | `content_index`, `content`, `partial` | Thinking block complete |
| `ToolCallStartEvent` | `content_index`, `partial` | Tool call begins |
| `ToolCallDeltaEvent` | `content_index`, `delta`, `partial` | Tool call args streaming |
| `ToolCallEndEvent` | `content_index`, `tool_call`, `partial` | Tool call complete |
| `DoneEvent` | `reason`, `message` | Successful completion |
| `ErrorEvent` | `reason`, `error` | Error or abort |

Every event carries a `partial` field with the full `AssistantMessage` accumulated so far. You can also `await event_stream.result()` to get the final `AssistantMessage` directly.

## Registry

### Model registry

```python
from pi.ai import get_model, get_models, get_providers, register_models

# List all provider families that have models registered
get_providers()
# -> ["anthropic", "openai", "google", "amazon-bedrock", "openai-codex", "github-copilot"]

# List all models for a provider
get_models("openai")
# -> [Model(id="gpt-5.2", ...), Model(id="gpt-5.1", ...), ...]

# Look up a specific model
model = get_model("anthropic", "claude-opus-4-6")

# Register custom models
from pi.ai.types import Model, ModelCost

register_models("my-provider", {
    "my-model-v1": Model(
        id="my-model-v1",
        name="My Model v1",
        api="openai-completions",     # Which protocol to use
        provider="my-provider",
        baseUrl="https://my-api.example.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1.0, output=3.0),
        contextWindow=128000,
        maxTokens=8192,
    ),
})
```

### API provider registry

```python
from pi.ai import get_api_provider, register_api_provider, ApiProvider

# Look up a provider implementation
provider = get_api_provider("openai-responses")
# provider.stream(model, context, options)
# provider.stream_simple(model, context, simple_options)

# Register a custom provider
register_api_provider(ApiProvider(
    api="my-custom-api",
    stream=my_stream_function,
    stream_simple=my_simple_stream_function,
))
```

## Writing a custom provider

Every provider implements one function signature:

```python
def stream_my_provider(
    model: Model,
    context: Context,
    options: MyProviderOptions | None = None,
) -> AssistantMessageEventStream:
```

Here is a minimal but complete provider:

```python
import asyncio
import time
from dataclasses import dataclass
from pi.ai.events import AssistantMessageEventStream
from pi.ai.types import (
    AssistantMessage, Context, DoneEvent, ErrorEvent, Model,
    StartEvent, TextContent, TextDeltaEvent, TextEndEvent,
    TextStartEvent, Usage,
)


@dataclass
class EchoProviderOptions:
    prefix: str = "Echo: "


def stream_echo(
    model: Model,
    context: Context,
    options: EchoProviderOptions | None = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()

    async def _run():
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        )

        try:
            # Get last user message
            last_msg = context.messages[-1] if context.messages else None
            user_text = ""
            if last_msg and last_msg.role == "user":
                if isinstance(last_msg.content, str):
                    user_text = last_msg.content
                else:
                    user_text = " ".join(
                        c.text for c in last_msg.content if c.type == "text"
                    )

            prefix = (options.prefix if options else "Echo: ")
            reply = prefix + user_text

            event_stream.push(StartEvent(partial=output))

            # Add a text content block
            output.content.append(TextContent(text=""))
            event_stream.push(TextStartEvent(content_index=0, partial=output))

            # Stream character by character (or by chunk)
            for char in reply:
                output.content[0].text += char
                event_stream.push(
                    TextDeltaEvent(content_index=0, delta=char, partial=output)
                )

            event_stream.push(
                TextEndEvent(content_index=0, content=reply, partial=output)
            )

            event_stream.push(DoneEvent(reason="stop", message=output))
            event_stream.end()

        except Exception as e:
            output.stop_reason = "error"
            output.error_message = str(e)
            event_stream.push(ErrorEvent(reason="error", error=output))
            event_stream.end()

    event_stream._background_task = asyncio.ensure_future(_run())
    return event_stream
```

Register it:

```python
from pi.ai import register_api_provider, ApiProvider, register_models
from pi.ai.types import Model, ModelCost

register_api_provider(ApiProvider(
    api="echo",
    stream=stream_echo,
    stream_simple=stream_echo,  # Same function works for simple API
))

register_models("echo-provider", {
    "echo-v1": Model(
        id="echo-v1",
        name="Echo v1",
        api="echo",
        provider="echo-provider",
        baseUrl="",
        contextWindow=100000,
        maxTokens=100000,
    ),
})
```

Use it:

```python
from pi.ai import get_model, complete_simple
from pi.ai.types import Context, UserMessage

model = get_model("echo-provider", "echo-v1")
result = await complete_simple(model, Context(
    messages=[UserMessage(role="user", content="hello world", timestamp=0)]
))
print(result.content[0].text)  # "Echo: hello world"
```

### Provider conventions

When implementing a provider, follow these patterns (all built-in providers do):

1. Create an `AssistantMessageEventStream` at the top of the function.
2. Define an inner `async def _run()` that does all work.
3. Set `event_stream._background_task = asyncio.ensure_future(_run())`.
4. Return the event stream immediately (non-blocking).
5. Inside `_run()`, push events in order: `StartEvent` -> content events -> `DoneEvent`/`ErrorEvent`.
6. Wrap the body of `_run()` in try/except. On error, set `stop_reason` to `"error"` or `"aborted"` and push `ErrorEvent`.
7. Always call `event_stream.end()` in both success and error paths.

### The two-tier API

Each provider exposes two functions:

- `stream_<provider>(model, context, provider_options)` -- full control with provider-specific options.
- `stream_simple_<provider>(model, context, simple_options)` -- accepts `SimpleStreamOptions` with a `reasoning` level. Internally maps the reasoning level to provider-specific parameters (thinking budgets, effort levels, etc.) and delegates to the full function.

The simple API exists so that application code can set `reasoning="high"` without knowing whether the provider uses token budgets (Anthropic, Google 2.x), effort strings (OpenAI), or thinking levels (Google 3.x).

## File structure

```
src/pi/ai/
    __init__.py              Public API surface, auto-registers providers and models
    types.py                 Pydantic models (Message, Model, Context, events, etc.)
    events.py                EventStream and AssistantMessageEventStream
    stream.py                Top-level dispatch: stream(), complete(), stream_simple()
    models.py                Model registry (register, get, cost calculation)
    models_builtin.py        65 pre-registered models across 6 providers
    registry.py              API provider registry
    env.py                   Environment variable API key resolution
    utils/
        json.py              Streaming JSON parser
        overflow.py          Text overflow utilities
        validation.py        JSON Schema argument validation
    providers/
        anthropic.py         Anthropic Messages API
        openai_completions.py OpenAI-compatible Chat Completions (15+ sub-providers)
        openai_responses.py  OpenAI Responses API
        openai_codex_responses.py  OpenAI Codex (raw HTTP, JWT, SSE)
        azure_openai_responses.py  Azure OpenAI Responses
        google.py            Google Generative AI (Gemini via API key)
        google_vertex.py     Google Vertex AI (ADC auth)
        google_gemini_cli.py Google Cloud Code Assist / Antigravity (raw HTTP, OAuth)
        amazon_bedrock.py    Amazon Bedrock Converse Stream
        openai_shared.py     Shared utilities for OpenAI Responses providers
        google_shared.py     Shared utilities for Google providers
        options.py           Reasoning budget calculation
        transform.py         Cross-provider message normalization
        builtins.py          Provider registration on import
```

## Environment variables

API keys are resolved from environment variables per provider:

| Provider | Variable | Fallback |
|----------|----------|----------|
| `anthropic` | `ANTHROPIC_API_KEY` | `PI_API_KEY` |
| `openai` | `OPENAI_API_KEY` | |
| `google` | `GOOGLE_API_KEY` | |
| `amazon-bedrock` | (uses AWS credentials chain) | |
| `openai-codex` | `OPENAI_CODEX_API_KEY` | |
| `github-copilot` | `GITHUB_COPILOT_TOKEN` | |
| `groq` | `GROQ_API_KEY` | |
| `xai` | `XAI_API_KEY` | |
| `mistral` | `MISTRAL_API_KEY` | |

You can also pass `api_key` directly in stream options, which takes precedence over environment variables.
