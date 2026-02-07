"""Hierarchical settings manager with JSON persistence.

Supports three-level precedence: global > project > CLI overrides.
Tracks per-field modifications for safe concurrent file updates.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_DIR_NAME = ".pi"


# --- Settings schema ---


@dataclass
class CompactionSettings:
    """Controls session history compaction."""

    enabled: bool | None = None
    reserve_tokens: int | None = None
    keep_recent_tokens: int | None = None


@dataclass
class BranchSummarySettings:
    """Controls branch summarization."""

    reserve_tokens: int | None = None


@dataclass
class RetrySettings:
    """Controls API retry behavior."""

    enabled: bool | None = None
    max_retries: int | None = None
    base_delay_ms: int | None = None
    max_delay_ms: int | None = None


@dataclass
class TerminalSettings:
    """Terminal display options."""

    show_images: bool | None = None
    clear_on_shrink: bool | None = None


@dataclass
class ImageSettings:
    """Image handling options."""

    auto_resize: bool | None = None
    block_images: bool | None = None


@dataclass
class ThinkingBudgetsSettings:
    """Custom token budgets per thinking level."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


@dataclass
class MarkdownSettings:
    """Markdown formatting options."""

    code_block_indent: int | None = None


# Type alias for package source specification
PackageSource = str | dict[str, Any]


def _settings_defaults() -> dict[str, Any]:
    """Default settings values."""
    return {
        "defaultProvider": None,
        "defaultModel": None,
        "defaultThinkingLevel": None,
        "steeringMode": None,
        "followUpMode": None,
        "theme": None,
        "hideThinkingBlock": None,
        "shellPath": None,
        "shellCommandPrefix": None,
        "quietStartup": None,
        "collapseChangelog": None,
        "enableSkillCommands": None,
        "enabledModels": None,
        "doubleEscapeAction": None,
        "showHardwareCursor": None,
        "editorPaddingX": None,
        "autocompleteMaxVisible": None,
        "lastChangelogVersion": None,
        "compaction": None,
        "branchSummary": None,
        "retry": None,
        "terminal": None,
        "images": None,
        "thinking": None,
        "markdown": None,
        "packages": None,
        "extensions": None,
        "skills": None,
        "promptTemplates": None,
        "themes": None,
    }


# --- Deep merge ---


def deep_merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base settings.

    For nested dicts, merge recursively. For primitives and arrays,
    override value wins completely.
    """
    result = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict) and not isinstance(value, list):
            result[key] = deep_merge_settings(result[key], value)
        else:
            result[key] = value
    return result


# --- Migrations ---


def _migrate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Apply all settings migrations."""
    # Migration 1: queueMode -> steeringMode
    if "queueMode" in settings and "steeringMode" not in settings:
        settings["steeringMode"] = settings.pop("queueMode")
    elif "queueMode" in settings:
        del settings["queueMode"]

    # Migration 2: skills object format -> array format
    skills_val = settings.get("skills")
    if isinstance(skills_val, dict) and "customDirectories" in skills_val:
        settings["skills"] = skills_val["customDirectories"]

    return settings


# --- SettingsManager ---


class SettingsManager:
    """Manages hierarchical settings with JSON file persistence.

    Three-level precedence:
        CLI overrides > project settings > global settings

    Use factory methods (create, in_memory) instead of calling constructor directly.
    """

    def __init__(
        self,
        *,
        settings_path: str | None,
        project_settings_path: str | None,
        initial_settings: dict[str, Any],
        persist: bool = True,
        load_error: Exception | None = None,
    ) -> None:
        self._settings_path = settings_path
        self._project_settings_path = project_settings_path
        self._global_settings = dict(initial_settings)
        self._persist = persist
        self._load_error = load_error
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}

        # Merge global + project settings
        project = self._load_project_settings() if self._project_settings_path else {}
        self._settings = deep_merge_settings(self._global_settings, project)

    # --- Factory methods ---

    @classmethod
    def create(cls, cwd: str, agent_dir: str | None = None) -> SettingsManager:
        """Create a settings manager with file persistence."""
        adir = agent_dir or _default_agent_dir()
        settings_path = os.path.join(adir, "settings.json")
        project_settings_path = os.path.join(cwd, CONFIG_DIR_NAME, "settings.json")

        settings, error = _load_from_file(settings_path)
        return cls(
            settings_path=settings_path,
            project_settings_path=project_settings_path,
            initial_settings=settings,
            persist=True,
            load_error=error,
        )

    @classmethod
    def in_memory(cls, settings: dict[str, Any] | None = None) -> SettingsManager:
        """Create an in-memory settings manager for testing."""
        return cls(
            settings_path=None,
            project_settings_path=None,
            initial_settings=settings or {},
            persist=False,
        )

    # --- Core operations ---

    def reload(self) -> None:
        """Reload all settings from disk."""
        if self._settings_path:
            self._global_settings, self._load_error = _load_from_file(self._settings_path)
        self._modified_fields.clear()
        self._modified_nested_fields.clear()

        project = self._load_project_settings() if self._project_settings_path else {}
        self._settings = deep_merge_settings(self._global_settings, project)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply CLI-level overrides on top of merged settings."""
        self._settings = deep_merge_settings(self._settings, overrides)

    def get_global_settings(self) -> dict[str, Any]:
        """Get a deep copy of the raw global settings."""
        return deepcopy(self._global_settings)

    def get_project_settings(self) -> dict[str, Any]:
        """Reload and return project-level settings."""
        return self._load_project_settings()

    @property
    def settings(self) -> dict[str, Any]:
        """Current merged settings (read-only view)."""
        return self._settings

    # --- Modification tracking ---

    def _mark_modified(self, field_name: str, nested_key: str | None = None) -> None:
        """Track which fields have been modified in this session."""
        self._modified_fields.add(field_name)
        if nested_key:
            if field_name not in self._modified_nested_fields:
                self._modified_nested_fields[field_name] = set()
            self._modified_nested_fields[field_name].add(nested_key)

    # --- Persistence ---

    def _save(self) -> None:
        """Write only modified fields to global settings file, preserving external changes."""
        if self._persist and self._settings_path:
            # Don't overwrite corrupted files
            if self._load_error:
                return

            # Re-read to capture external changes
            current_file, _ = _load_from_file(self._settings_path)
            merged: dict[str, Any] = dict(current_file)

            for field_name in self._modified_fields:
                value = self._global_settings.get(field_name)
                nested_keys = self._modified_nested_fields.get(field_name)

                if nested_keys and isinstance(value, dict):
                    # Merge only modified nested keys
                    if field_name not in merged or not isinstance(merged[field_name], dict):
                        merged[field_name] = {}
                    for nk in nested_keys:
                        merged[field_name][nk] = value.get(nk)
                else:
                    merged[field_name] = value

            # Remove None values at top level
            merged = {k: v for k, v in merged.items() if v is not None}

            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            Path(self._settings_path).write_text(
                json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        # Re-merge after save
        project = self._load_project_settings() if self._project_settings_path else {}
        self._settings = deep_merge_settings(self._global_settings, project)

    def _save_project_settings(self, settings: dict[str, Any]) -> None:
        """Write project-level settings file."""
        if not self._project_settings_path:
            return
        os.makedirs(os.path.dirname(self._project_settings_path), exist_ok=True)
        Path(self._project_settings_path).write_text(
            json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Re-merge
        self._settings = deep_merge_settings(self._global_settings, settings)

    def _load_project_settings(self) -> dict[str, Any]:
        """Load project-level settings from disk."""
        if not self._project_settings_path:
            return {}
        settings, _ = _load_from_file(self._project_settings_path)
        return settings

    # --- Getters: Provider & Model ---

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    # --- Setters: Provider & Model ---

    def set_default_provider(self, provider: str) -> None:
        self._global_settings["defaultProvider"] = provider
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_model(self, model_id: str) -> None:
        self._global_settings["defaultModel"] = model_id
        self._mark_modified("defaultModel")
        self._save()

    def set_default_model_and_provider(self, model_id: str, provider: str) -> None:
        self._global_settings["defaultModel"] = model_id
        self._global_settings["defaultProvider"] = provider
        self._mark_modified("defaultModel")
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_thinking_level(self, level: str) -> None:
        self._global_settings["defaultThinkingLevel"] = level
        self._mark_modified("defaultThinkingLevel")
        self._save()

    # --- Getters: Modes ---

    def get_steering_mode(self) -> str | None:
        return self._settings.get("steeringMode")

    def get_follow_up_mode(self) -> str | None:
        return self._settings.get("followUpMode")

    # --- Setters: Modes ---

    def set_steering_mode(self, mode: str) -> None:
        self._global_settings["steeringMode"] = mode
        self._mark_modified("steeringMode")
        self._save()

    def set_follow_up_mode(self, mode: str) -> None:
        self._global_settings["followUpMode"] = mode
        self._mark_modified("followUpMode")
        self._save()

    # --- Getters: UI ---

    def get_theme(self) -> str | None:
        return self._settings.get("theme")

    def get_hide_thinking_block(self) -> bool | None:
        return self._settings.get("hideThinkingBlock")

    def get_quiet_startup(self) -> bool:
        return self._settings.get("quietStartup") or False

    def get_collapse_changelog(self) -> bool | None:
        return self._settings.get("collapseChangelog")

    def get_show_hardware_cursor(self) -> bool:
        val = self._settings.get("showHardwareCursor")
        if val is not None:
            return val
        return os.environ.get("PI_HARDWARE_CURSOR") == "1"

    def get_editor_padding_x(self) -> int:
        return self._settings.get("editorPaddingX") or 0

    def get_autocomplete_max_visible(self) -> int:
        return self._settings.get("autocompleteMaxVisible") or 10

    # --- Setters: UI ---

    def set_theme(self, theme: str) -> None:
        self._global_settings["theme"] = theme
        self._mark_modified("theme")
        self._save()

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._global_settings["hideThinkingBlock"] = hide
        self._mark_modified("hideThinkingBlock")
        self._save()

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings["quietStartup"] = quiet
        self._mark_modified("quietStartup")
        self._save()

    def set_show_hardware_cursor(self, show: bool) -> None:
        self._global_settings["showHardwareCursor"] = show
        self._mark_modified("showHardwareCursor")
        self._save()

    def set_editor_padding_x(self, padding: int) -> None:
        self._global_settings["editorPaddingX"] = max(0, min(3, int(padding)))
        self._mark_modified("editorPaddingX")
        self._save()

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        self._global_settings["autocompleteMaxVisible"] = max(3, min(20, int(max_visible)))
        self._mark_modified("autocompleteMaxVisible")
        self._save()

    # --- Getters: Shell ---

    def get_shell_path(self) -> str | None:
        return self._settings.get("shellPath")

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    # --- Setters: Shell ---

    def set_shell_path(self, path: str) -> None:
        self._global_settings["shellPath"] = path
        self._mark_modified("shellPath")
        self._save()

    # --- Getters: Compaction ---

    def get_compaction_enabled(self) -> bool:
        compaction = self._settings.get("compaction")
        if isinstance(compaction, dict):
            val = compaction.get("enabled")
            return val if val is not None else True
        return True

    def get_compaction_settings(self) -> CompactionSettings:
        compaction = self._settings.get("compaction") or {}
        return CompactionSettings(
            enabled=compaction.get("enabled"),
            reserve_tokens=compaction.get("reserveTokens"),
            keep_recent_tokens=compaction.get("keepRecentTokens"),
        )

    # --- Setters: Compaction ---

    def set_compaction_enabled(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("compaction"), dict):
            self._global_settings["compaction"] = {}
        self._global_settings["compaction"]["enabled"] = enabled
        self._mark_modified("compaction", "enabled")
        self._save()

    # --- Getters: Retry ---

    def get_retry_enabled(self) -> bool:
        retry = self._settings.get("retry")
        if isinstance(retry, dict):
            val = retry.get("enabled")
            return val if val is not None else True
        return True

    def get_retry_settings(self) -> RetrySettings:
        retry = self._settings.get("retry") or {}
        return RetrySettings(
            enabled=retry.get("enabled"),
            max_retries=retry.get("maxRetries", 3),
            base_delay_ms=retry.get("baseDelayMs", 2000),
            max_delay_ms=retry.get("maxDelayMs", 60000),
        )

    # --- Setters: Retry ---

    def set_retry_enabled(self, enabled: bool) -> None:
        if not isinstance(self._global_settings.get("retry"), dict):
            self._global_settings["retry"] = {}
        self._global_settings["retry"]["enabled"] = enabled
        self._mark_modified("retry", "enabled")
        self._save()

    # --- Getters: Terminal ---

    def get_show_images(self) -> bool:
        terminal = self._settings.get("terminal")
        if isinstance(terminal, dict):
            val = terminal.get("showImages")
            return val if val is not None else True
        return True

    def get_clear_on_shrink(self) -> bool:
        terminal = self._settings.get("terminal")
        if isinstance(terminal, dict):
            val = terminal.get("clearOnShrink")
            if val is not None:
                return val
        return os.environ.get("PI_CLEAR_ON_SHRINK") == "1"

    # --- Setters: Terminal ---

    def set_show_images(self, show: bool) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["showImages"] = show
        self._mark_modified("terminal", "showImages")
        self._save()

    def set_clear_on_shrink(self, clear: bool) -> None:
        if not isinstance(self._global_settings.get("terminal"), dict):
            self._global_settings["terminal"] = {}
        self._global_settings["terminal"]["clearOnShrink"] = clear
        self._mark_modified("terminal", "clearOnShrink")
        self._save()

    # --- Getters: Images ---

    def get_image_auto_resize(self) -> bool:
        images = self._settings.get("images")
        if isinstance(images, dict):
            val = images.get("autoResize")
            return val if val is not None else True
        return True

    def get_block_images(self) -> bool:
        images = self._settings.get("images")
        if isinstance(images, dict):
            return images.get("blockImages") or False
        return False

    # --- Setters: Images ---

    def set_image_auto_resize(self, resize: bool) -> None:
        if not isinstance(self._global_settings.get("images"), dict):
            self._global_settings["images"] = {}
        self._global_settings["images"]["autoResize"] = resize
        self._mark_modified("images", "autoResize")
        self._save()

    def set_block_images(self, block: bool) -> None:
        if not isinstance(self._global_settings.get("images"), dict):
            self._global_settings["images"] = {}
        self._global_settings["images"]["blockImages"] = block
        self._mark_modified("images", "blockImages")
        self._save()

    # --- Getters: Thinking ---

    def get_thinking_budgets(self) -> ThinkingBudgetsSettings:
        thinking = self._settings.get("thinking") or {}
        return ThinkingBudgetsSettings(
            minimal=thinking.get("minimal"),
            low=thinking.get("low"),
            medium=thinking.get("medium"),
            high=thinking.get("high"),
        )

    # --- Getters: Markdown ---

    def get_code_block_indent(self) -> int:
        md = self._settings.get("markdown")
        if isinstance(md, dict):
            return md.get("codeBlockIndent") or 0
        return 0

    # --- Getters: Branch Summary ---

    def get_branch_summary_settings(self) -> BranchSummarySettings:
        bs = self._settings.get("branchSummary") or {}
        return BranchSummarySettings(reserve_tokens=bs.get("reserveTokens"))

    # --- Getters: Features ---

    def get_enable_skill_commands(self) -> bool | None:
        return self._settings.get("enableSkillCommands")

    def get_enabled_models(self) -> list[str] | None:
        return self._settings.get("enabledModels")

    def get_double_escape_action(self) -> str | None:
        return self._settings.get("doubleEscapeAction")

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    # --- Setters: Features ---

    def set_last_changelog_version(self, version: str) -> None:
        self._global_settings["lastChangelogVersion"] = version
        self._mark_modified("lastChangelogVersion")
        self._save()

    # --- Getters: Packages / Extensions / Skills / Prompts / Themes ---

    def get_packages(self) -> list[PackageSource]:
        return list(self._settings.get("packages") or [])

    def get_extension_paths(self) -> list[str]:
        return list(self._settings.get("extensions") or [])

    def get_skill_paths(self) -> list[str]:
        return list(self._settings.get("skills") or [])

    def get_prompt_template_paths(self) -> list[str]:
        return list(self._settings.get("promptTemplates") or [])

    def get_theme_paths(self) -> list[str]:
        return list(self._settings.get("themes") or [])

    # --- Setters: Packages / Extensions / Skills / Prompts / Themes (global) ---

    def set_packages(self, packages: list[PackageSource]) -> None:
        self._global_settings["packages"] = packages
        self._mark_modified("packages")
        self._save()

    def set_extension_paths(self, paths: list[str]) -> None:
        self._global_settings["extensions"] = paths
        self._mark_modified("extensions")
        self._save()

    def set_skill_paths(self, paths: list[str]) -> None:
        self._global_settings["skills"] = paths
        self._mark_modified("skills")
        self._save()

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._global_settings["promptTemplates"] = paths
        self._mark_modified("promptTemplates")
        self._save()

    def set_theme_paths(self, paths: list[str]) -> None:
        self._global_settings["themes"] = paths
        self._mark_modified("themes")
        self._save()

    # --- Setters: Packages / Extensions / Skills / Prompts / Themes (project) ---

    def set_project_packages(self, packages: list[PackageSource]) -> None:
        project = self._load_project_settings()
        project["packages"] = packages
        self._save_project_settings(project)

    def set_project_extension_paths(self, paths: list[str]) -> None:
        project = self._load_project_settings()
        project["extensions"] = paths
        self._save_project_settings(project)

    def set_project_skill_paths(self, paths: list[str]) -> None:
        project = self._load_project_settings()
        project["skills"] = paths
        self._save_project_settings(project)

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        project = self._load_project_settings()
        project["promptTemplates"] = paths
        self._save_project_settings(project)

    def set_project_theme_paths(self, paths: list[str]) -> None:
        project = self._load_project_settings()
        project["themes"] = paths
        self._save_project_settings(project)


# --- File I/O helpers ---


def _load_from_file(path: str) -> tuple[dict[str, Any], Exception | None]:
    """Load settings from a JSON file. Returns (settings, error)."""
    if not os.path.exists(path):
        return {}, None
    try:
        content = Path(path).read_text(encoding="utf-8")
        settings = json.loads(content)
        return _migrate_settings(settings), None
    except (OSError, json.JSONDecodeError) as e:
        return {}, e


def _default_agent_dir() -> str:
    """Default agent data directory (~/.pi)."""
    return os.path.join(os.path.expanduser("~"), ".pi")
