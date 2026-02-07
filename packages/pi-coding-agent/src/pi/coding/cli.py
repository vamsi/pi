"""CLI entry point for the pi coding agent.

Enhanced CLI that uses AgentSession with model resolution,
session management, and extension support.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from pi.agent.agent import Agent
from pi.agent.types import AgentState, MessageEndEvent
from pi.ai.types import TextContent
from pi.coding.core.resolver import ModelRegistry, find_initial_model
from pi.coding.core.session import AgentSession, AgentSessionConfig
from pi.coding.core.sessions import SessionManager
from pi.coding.core.settings import SettingsManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pi",
        description="AI coding agent with tool use",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send (print mode)")
    parser.add_argument("-m", "--model", help="Model ID or pattern (e.g., 'claude-opus', 'openai/gpt-5')")
    parser.add_argument("-p", "--provider", help="Provider name")
    parser.add_argument("--api-key", help="API key (or set via env)")
    parser.add_argument("--print", dest="print_mode", action="store_true", help="Print mode (non-interactive)")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue most recent session")
    parser.add_argument("--session", help="Resume a specific session file")
    parser.add_argument("--thinking", help="Thinking level (off, minimal, low, medium, high, xhigh)")
    parser.add_argument("--no-extensions", action="store_true", help="Disable extensions")
    parser.add_argument("--extension", action="append", dest="extensions", help="Additional extension path")
    return parser.parse_args()


def _setup_registry() -> ModelRegistry:
    """Create and populate the model registry with built-in providers."""

    registry = ModelRegistry()
    # Built-in models are registered by pi.ai on import
    # Register from the global providers that pi.ai sets up
    from pi.ai.registry import get_all_models

    for model in get_all_models():
        registry.register(model)

    return registry


def _resolve_model(
    args: argparse.Namespace,
    registry: ModelRegistry,
    settings: SettingsManager,
    is_continue: bool,
) -> tuple[Any, str]:
    """Resolve the model from CLI args, settings, and registry.

    Returns (model, thinking_level).
    """
    result = find_initial_model(
        models=registry.get_all(),
        cli_provider=args.provider,
        cli_model=args.model,
        settings=settings,
        is_continue=is_continue,
    )

    if not result.model:
        print(f"Error: {result.fallback_message or 'No models available'}", file=sys.stderr)
        sys.exit(1)

    thinking_level = args.thinking or result.thinking_level
    return result.model, thinking_level


async def run_print_mode(prompt: str, args: argparse.Namespace) -> None:
    """Run in print mode: send prompt and print response using AgentSession."""
    settings = SettingsManager.create(args.cwd)

    # Try to set up registry with models, fall back to manual model creation
    try:
        registry = _setup_registry()
        model, thinking_level = _resolve_model(args, registry, settings, is_continue=False)
    except Exception:
        # Fallback: create model directly from args
        from pi.ai.types import Model, ModelCost

        model_id = args.model or "claude-sonnet-4-5-20250929"
        provider = args.provider or "anthropic"
        model = Model(
            id=model_id,
            name=model_id,
            api="anthropic-messages",
            provider=provider,
            base_url="https://api.anthropic.com",
            reasoning=False,
            input=["text"],
            cost=ModelCost(),
            context_window=200000,
            max_tokens=8192,
        )
        thinking_level = args.thinking or "off"
        registry = None

    # Create session manager
    if args.session:
        sm = SessionManager.open(args.session)
    elif args.continue_session:
        sm = SessionManager.continue_recent(args.cwd)
    else:
        sm = SessionManager.in_memory(args.cwd)

    # Create agent
    agent = Agent(initial_state=AgentState(model=model))
    if thinking_level and thinking_level != "off":
        agent.set_thinking_level(thinking_level)

    # Load extensions unless disabled
    extensions = None
    extension_runner = None
    if not args.no_extensions:
        try:
            from pi.coding.core.extensions.loader import discover_and_load_extensions

            configured = settings.get_extension_paths()
            extra = args.extensions or []
            all_paths = configured + extra
            if all_paths:
                extensions, _errors = discover_and_load_extensions(
                    configured_paths=all_paths,
                    cwd=args.cwd,
                )
                if extensions:
                    from pi.coding.core.extensions.runner import ExtensionRunner

                    extension_runner = ExtensionRunner(extensions, args.cwd)
        except Exception:
            pass  # Extensions are optional

    # Build AgentSession
    config = AgentSessionConfig(
        agent=agent,
        session_manager=sm,
        settings_manager=settings,
        cwd=args.cwd,
        model_registry=registry if registry else None,
        extension_runner=extension_runner,
        extensions=extensions,
    )
    session = AgentSession(config)

    # Subscribe to events for output
    def on_event(event: Any) -> None:
        if isinstance(event, MessageEndEvent):
            msg = event.message
            if hasattr(msg, "content") and hasattr(msg, "role") and msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, TextContent):
                        print(block.text, end="")
                print()

    session.subscribe(on_event)

    # Send prompt
    await session.prompt(prompt)
    await session.agent.wait_for_idle()

    # Clean up
    session.dispose()


def main() -> None:
    args = parse_args()

    if args.prompt or args.print_mode:
        prompt = args.prompt or ""
        if not prompt:
            prompt = sys.stdin.read()
        asyncio.run(run_print_mode(prompt, args))
    else:
        print("Interactive mode not yet implemented. Use --print mode or pass a prompt.")
        print("Usage: pi 'your prompt here'")
        sys.exit(1)


if __name__ == "__main__":
    main()
