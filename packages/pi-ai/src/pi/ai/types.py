"""Core types for the unified LLM API.

All types use Pydantic models for validation and serialization.
snake_case naming throughout, with camelCase aliases for JSONL compatibility.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- API and Provider identifiers ---

KnownApi = Literal[
    "openai-completions",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
]

Api = str  # KnownApi or custom string

KnownProvider = Literal[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "huggingface",
    "opencode",
    "kimi-coding",
]

Provider = str  # KnownProvider or custom string

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

CacheRetention = Literal["none", "short", "long"]

StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]

# --- Content blocks ---


class TextContent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = Field(default=None, alias="textSignature")


class ThinkingContent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")


class ImageContent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mime_type: str = Field(alias="mimeType")


class ToolCall(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["tool_call"] = Field(default="tool_call", alias="toolCall")
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought_signature: str | None = Field(default=None, alias="thoughtSignature")


# --- Usage tracking ---


class UsageCost(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: UsageCost = Field(default_factory=UsageCost)


# --- Messages ---

UserContentItem = TextContent | ImageContent
AssistantContentItem = TextContent | ThinkingContent | ToolCall
ToolResultContentItem = TextContent | ImageContent


class UserMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user"] = "user"
    content: str | list[UserContentItem]
    timestamp: int  # Unix timestamp in milliseconds


class AssistantMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContentItem] = Field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = Field(default="stop", alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    timestamp: int = 0  # Unix timestamp in milliseconds


class ToolResultMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["tool_result"] = Field(default="tool_result", alias="toolResult")
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[ToolResultContentItem] = Field(default_factory=list)
    details: Any = None
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int = 0  # Unix timestamp in milliseconds


Message = UserMessage | AssistantMessage | ToolResultMessage

# --- Tool definition ---


class Tool(BaseModel):
    """Tool definition with JSON Schema parameters."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# --- Context ---


class Context(BaseModel):
    """Full context for an LLM call: system prompt, messages, and tools."""

    model_config = ConfigDict(populate_by_name=True)

    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] | None = None


# --- Model ---


class ModelCost(BaseModel):
    """Cost per million tokens."""

    model_config = ConfigDict(populate_by_name=True)

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class OpenAICompletionsCompat(BaseModel):
    """Compatibility settings for OpenAI-compatible completions APIs."""

    model_config = ConfigDict(populate_by_name=True)

    supports_store: bool | None = Field(default=None, alias="supportsStore")
    supports_developer_role: bool | None = Field(default=None, alias="supportsDeveloperRole")
    supports_reasoning_effort: bool | None = Field(default=None, alias="supportsReasoningEffort")
    supports_usage_in_streaming: bool | None = Field(default=None, alias="supportsUsageInStreaming")
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = Field(
        default=None, alias="maxTokensField"
    )
    requires_tool_result_name: bool | None = Field(default=None, alias="requiresToolResultName")
    requires_assistant_after_tool_result: bool | None = Field(default=None, alias="requiresAssistantAfterToolResult")
    requires_thinking_as_text: bool | None = Field(default=None, alias="requiresThinkingAsText")
    requires_mistral_tool_ids: bool | None = Field(default=None, alias="requiresMistralToolIds")
    thinking_format: Literal["openai", "zai", "qwen"] | None = Field(default=None, alias="thinkingFormat")
    supports_strict_mode: bool | None = Field(default=None, alias="supportsStrictMode")


class OpenAIResponsesCompat(BaseModel):
    """Compatibility settings for OpenAI Responses APIs."""

    pass


class Model(BaseModel):
    """Model definition for the unified model system."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    api: str
    provider: str
    base_url: str = Field(alias="baseUrl")
    reasoning: bool = False
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = Field(default=0, alias="contextWindow")
    max_tokens: int = Field(default=0, alias="maxTokens")
    headers: dict[str, str] | None = None
    compat: OpenAICompletionsCompat | OpenAIResponsesCompat | None = None


# --- Stream options ---


class ThinkingBudgets(BaseModel):
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


class StreamOptions(BaseModel):
    """Options for streaming LLM calls."""

    model_config = ConfigDict(populate_by_name=True)

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    api_key: str | None = Field(default=None, alias="apiKey")
    cache_retention: CacheRetention = Field(default="short", alias="cacheRetention")
    session_id: str | None = Field(default=None, alias="sessionId")
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = Field(default=None, alias="maxRetryDelayMs")
    on_payload: Any | None = Field(default=None, exclude=True)  # callback, not serialized


class SimpleStreamOptions(StreamOptions):
    """Stream options with reasoning level for simple API."""

    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = Field(default=None, alias="thinkingBudgets")


# --- Streaming events ---


class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class TextDeltaEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class TextEndEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    partial: AssistantMessage


class ToolCallDeltaEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")
    tool_call: ToolCall = Field(alias="toolCall")
    partial: AssistantMessage


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "tool_use"]
    message: AssistantMessage


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)
