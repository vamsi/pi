"""Tests for system prompt builder."""

from __future__ import annotations

from pi.coding.core.prompt import (
    ContextFile,
    build_system_prompt,
)


class TestBuildSystemPrompt:
    """Tests for the build_system_prompt function."""

    def test_default_prompt_contains_tools(self) -> None:
        prompt = build_system_prompt(cwd="/test")
        assert "Available tools:" in prompt
        assert "read: Read file contents" in prompt
        assert "bash: Execute bash commands" in prompt
        assert "edit: Make surgical edits" in prompt
        assert "write: Create or overwrite files" in prompt

    def test_default_prompt_contains_guidelines(self) -> None:
        prompt = build_system_prompt(cwd="/test")
        assert "Guidelines:" in prompt
        assert "Be concise in your responses" in prompt

    def test_default_prompt_contains_cwd(self) -> None:
        prompt = build_system_prompt(cwd="/my/project")
        assert "Current working directory: /my/project" in prompt

    def test_default_prompt_contains_datetime(self) -> None:
        prompt = build_system_prompt(cwd="/test")
        assert "Current date and time:" in prompt

    def test_selected_tools_filters(self) -> None:
        prompt = build_system_prompt(selected_tools=["read", "bash"], cwd="/test")
        assert "read: Read file contents" in prompt
        assert "bash: Execute bash commands" in prompt
        assert "edit:" not in prompt
        assert "write:" not in prompt

    def test_unknown_tools_filtered(self) -> None:
        prompt = build_system_prompt(selected_tools=["read", "magic_tool"], cwd="/test")
        assert "read: Read file contents" in prompt
        assert "magic_tool" not in prompt

    def test_no_tools_shows_none(self) -> None:
        prompt = build_system_prompt(selected_tools=[], cwd="/test")
        assert "(none)" in prompt

    def test_custom_prompt_replaces_default(self) -> None:
        prompt = build_system_prompt(custom_prompt="You are a helpful bot.", cwd="/test")
        assert prompt.startswith("You are a helpful bot.")
        assert "Available tools:" not in prompt

    def test_custom_prompt_includes_cwd(self) -> None:
        prompt = build_system_prompt(custom_prompt="Custom.", cwd="/my/dir")
        assert "Current working directory: /my/dir" in prompt

    def test_custom_prompt_includes_datetime(self) -> None:
        prompt = build_system_prompt(custom_prompt="Custom.", cwd="/test")
        assert "Current date and time:" in prompt

    def test_append_system_prompt(self) -> None:
        prompt = build_system_prompt(append_system_prompt="Extra instructions here.", cwd="/test")
        assert "Extra instructions here." in prompt

    def test_append_with_custom_prompt(self) -> None:
        prompt = build_system_prompt(
            custom_prompt="Base prompt.",
            append_system_prompt="Appended text.",
            cwd="/test",
        )
        assert "Base prompt." in prompt
        assert "Appended text." in prompt

    def test_context_files(self) -> None:
        files = [
            ContextFile(path="AGENTS.md", content="Agent rules here"),
            ContextFile(path=".pi/settings.json", content="Settings content"),
        ]
        prompt = build_system_prompt(context_files=files, cwd="/test")
        assert "# Project Context" in prompt
        assert "## AGENTS.md" in prompt
        assert "Agent rules here" in prompt
        assert "## .pi/settings.json" in prompt
        assert "Settings content" in prompt

    def test_context_files_with_custom_prompt(self) -> None:
        files = [ContextFile(path="README.md", content="Read me")]
        prompt = build_system_prompt(custom_prompt="Custom base.", context_files=files, cwd="/test")
        assert "Custom base." in prompt
        assert "## README.md" in prompt
        assert "Read me" in prompt

    def test_guidelines_bash_only(self) -> None:
        prompt = build_system_prompt(selected_tools=["bash"], cwd="/test")
        assert "Use bash for file operations like ls, rg, find" in prompt

    def test_guidelines_bash_with_exploration_tools(self) -> None:
        prompt = build_system_prompt(selected_tools=["bash", "grep", "find"], cwd="/test")
        assert "Prefer grep/find/ls tools over bash" in prompt

    def test_guidelines_read_and_edit(self) -> None:
        prompt = build_system_prompt(selected_tools=["read", "edit"], cwd="/test")
        assert "Use read to examine files before editing" in prompt
        assert "Use edit for precise changes" in prompt

    def test_guidelines_write(self) -> None:
        prompt = build_system_prompt(selected_tools=["write"], cwd="/test")
        assert "Use write only for new files or complete rewrites" in prompt

    def test_guidelines_output_with_edit_or_write(self) -> None:
        prompt = build_system_prompt(selected_tools=["edit"], cwd="/test")
        assert "output plain text directly" in prompt

    def test_expert_coding_assistant_intro(self) -> None:
        prompt = build_system_prompt(cwd="/test")
        assert "You are an expert coding assistant" in prompt

    def test_custom_tools_note(self) -> None:
        prompt = build_system_prompt(cwd="/test")
        assert "custom tools depending on the project" in prompt

    def test_all_seven_tools(self) -> None:
        all_tools = ["read", "bash", "edit", "write", "grep", "find", "ls"]
        prompt = build_system_prompt(selected_tools=all_tools, cwd="/test")
        for tool in all_tools:
            assert f"- {tool}:" in prompt
