"""Tests for the hierarchical settings manager."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pi.coding.core.settings import (
    SettingsManager,
    _migrate_settings,
    deep_merge_settings,
)

# --- Deep merge ---


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    overrides = {"b": 3, "c": 4}
    result = deep_merge_settings(base, overrides)
    assert result == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested():
    base = {"compaction": {"enabled": True, "reserveTokens": 1000}}
    overrides = {"compaction": {"enabled": False}}
    result = deep_merge_settings(base, overrides)
    assert result == {"compaction": {"enabled": False, "reserveTokens": 1000}}


def test_deep_merge_none_values_skipped():
    base = {"a": 1, "b": 2}
    overrides = {"a": None, "c": 3}
    result = deep_merge_settings(base, overrides)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_deep_merge_array_replacement():
    base = {"packages": ["a", "b"]}
    overrides = {"packages": ["c"]}
    result = deep_merge_settings(base, overrides)
    assert result == {"packages": ["c"]}


# --- Migrations ---


def test_migrate_queue_mode_to_steering():
    settings = {"queueMode": "one-at-a-time"}
    migrated = _migrate_settings(settings)
    assert "queueMode" not in migrated
    assert migrated["steeringMode"] == "one-at-a-time"


def test_migrate_skills_object_to_array():
    settings = {"skills": {"customDirectories": ["/path/to/skills"]}}
    migrated = _migrate_settings(settings)
    assert migrated["skills"] == ["/path/to/skills"]


def test_migrate_no_change():
    settings = {"defaultModel": "claude-opus-4-6"}
    migrated = _migrate_settings(settings)
    assert migrated == {"defaultModel": "claude-opus-4-6"}


# --- In-memory settings manager ---


def test_in_memory_defaults():
    mgr = SettingsManager.in_memory()
    assert mgr.get_default_provider() is None
    assert mgr.get_default_model() is None
    assert mgr.get_compaction_enabled() is True
    assert mgr.get_retry_enabled() is True
    assert mgr.get_show_images() is True
    assert mgr.get_quiet_startup() is False


def test_in_memory_with_initial():
    mgr = SettingsManager.in_memory({"defaultProvider": "anthropic", "defaultModel": "claude"})
    assert mgr.get_default_provider() == "anthropic"
    assert mgr.get_default_model() == "claude"


def test_set_default_model_and_provider():
    mgr = SettingsManager.in_memory()
    mgr.set_default_model_and_provider("claude-opus-4-6", "anthropic")
    assert mgr.get_default_model() == "claude-opus-4-6"
    assert mgr.get_default_provider() == "anthropic"


def test_set_thinking_level():
    mgr = SettingsManager.in_memory()
    mgr.set_default_thinking_level("high")
    assert mgr.get_default_thinking_level() == "high"


def test_set_steering_mode():
    mgr = SettingsManager.in_memory()
    mgr.set_steering_mode("one-at-a-time")
    assert mgr.get_steering_mode() == "one-at-a-time"


def test_set_theme():
    mgr = SettingsManager.in_memory()
    mgr.set_theme("dark")
    assert mgr.get_theme() == "dark"


# --- Nested settings ---


def test_compaction_settings():
    mgr = SettingsManager.in_memory({"compaction": {"enabled": False, "reserveTokens": 5000}})
    assert not mgr.get_compaction_enabled()
    settings = mgr.get_compaction_settings()
    assert settings.enabled is False
    assert settings.reserve_tokens == 5000


def test_set_compaction_enabled():
    mgr = SettingsManager.in_memory()
    mgr.set_compaction_enabled(False)
    assert not mgr.get_compaction_enabled()


def test_retry_settings():
    mgr = SettingsManager.in_memory({"retry": {"maxRetries": 5}})
    settings = mgr.get_retry_settings()
    assert settings.max_retries == 5
    assert settings.base_delay_ms == 2000  # default


def test_set_retry_enabled():
    mgr = SettingsManager.in_memory()
    mgr.set_retry_enabled(False)
    assert not mgr.get_retry_enabled()


def test_terminal_settings():
    mgr = SettingsManager.in_memory({"terminal": {"showImages": False}})
    assert not mgr.get_show_images()


def test_set_terminal_settings():
    mgr = SettingsManager.in_memory()
    mgr.set_show_images(False)
    assert not mgr.get_show_images()
    mgr.set_clear_on_shrink(True)
    assert mgr.get_clear_on_shrink()


def test_image_settings():
    mgr = SettingsManager.in_memory({"images": {"autoResize": False, "blockImages": True}})
    assert not mgr.get_image_auto_resize()
    assert mgr.get_block_images()


def test_set_image_settings():
    mgr = SettingsManager.in_memory()
    mgr.set_image_auto_resize(False)
    assert not mgr.get_image_auto_resize()
    mgr.set_block_images(True)
    assert mgr.get_block_images()


# --- Value clamping ---


def test_editor_padding_clamped():
    mgr = SettingsManager.in_memory()
    mgr.set_editor_padding_x(10)
    assert mgr.get_editor_padding_x() == 3  # max

    mgr.set_editor_padding_x(-5)
    assert mgr.get_editor_padding_x() == 0  # min


def test_autocomplete_max_visible_clamped():
    mgr = SettingsManager.in_memory()
    mgr.set_autocomplete_max_visible(100)
    assert mgr.get_autocomplete_max_visible() == 20  # max

    mgr.set_autocomplete_max_visible(1)
    assert mgr.get_autocomplete_max_visible() == 3  # min


# --- Packages / Extensions ---


def test_packages():
    mgr = SettingsManager.in_memory({"packages": ["pkg1", "pkg2"]})
    assert mgr.get_packages() == ["pkg1", "pkg2"]

    mgr.set_packages(["pkg3"])
    assert mgr.get_packages() == ["pkg3"]


def test_extension_paths():
    mgr = SettingsManager.in_memory({"extensions": ["/ext1"]})
    assert mgr.get_extension_paths() == ["/ext1"]


def test_skill_paths():
    mgr = SettingsManager.in_memory()
    assert mgr.get_skill_paths() == []
    mgr.set_skill_paths(["/skills"])
    assert mgr.get_skill_paths() == ["/skills"]


# --- File persistence ---


def test_create_and_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi")
        cwd = os.path.join(tmpdir, "project")
        os.makedirs(cwd, exist_ok=True)

        mgr = SettingsManager.create(cwd, agent_dir)
        mgr.set_default_model_and_provider("claude-opus-4-6", "anthropic")

        # Verify file was written
        settings_path = os.path.join(agent_dir, "settings.json")
        assert os.path.exists(settings_path)

        content = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        assert content["defaultModel"] == "claude-opus-4-6"
        assert content["defaultProvider"] == "anthropic"


def test_project_settings_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi")
        cwd = os.path.join(tmpdir, "project")
        project_pi = os.path.join(cwd, ".pi")
        os.makedirs(project_pi, exist_ok=True)

        # Write global settings
        os.makedirs(agent_dir, exist_ok=True)
        Path(os.path.join(agent_dir, "settings.json")).write_text(
            json.dumps({"defaultModel": "global-model", "theme": "dark"}),
            encoding="utf-8",
        )

        # Write project settings
        Path(os.path.join(project_pi, "settings.json")).write_text(
            json.dumps({"defaultModel": "project-model"}),
            encoding="utf-8",
        )

        mgr = SettingsManager.create(cwd, agent_dir)
        # Project overrides global for defaultModel
        assert mgr.get_default_model() == "project-model"
        # Global theme is preserved
        assert mgr.get_theme() == "dark"


def test_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi")
        cwd = os.path.join(tmpdir, "project")
        os.makedirs(cwd, exist_ok=True)

        mgr = SettingsManager.create(cwd, agent_dir)
        mgr.set_default_model("model-1")
        assert mgr.get_default_model() == "model-1"

        # External change
        settings_path = os.path.join(agent_dir, "settings.json")
        Path(settings_path).write_text(
            json.dumps({"defaultModel": "model-2"}),
            encoding="utf-8",
        )

        mgr.reload()
        assert mgr.get_default_model() == "model-2"


def test_apply_overrides():
    mgr = SettingsManager.in_memory({"defaultModel": "base"})
    mgr.apply_overrides({"defaultModel": "override"})
    assert mgr.get_default_model() == "override"


def test_project_packages():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi")
        cwd = os.path.join(tmpdir, "project")
        os.makedirs(cwd, exist_ok=True)

        mgr = SettingsManager.create(cwd, agent_dir)
        mgr.set_project_packages(["pkg1", "pkg2"])

        # Project settings file should exist
        project_settings = os.path.join(cwd, ".pi", "settings.json")
        assert os.path.exists(project_settings)
        content = json.loads(Path(project_settings).read_text(encoding="utf-8"))
        assert content["packages"] == ["pkg1", "pkg2"]


def test_modification_tracking_preserves_external_changes():
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_dir = os.path.join(tmpdir, ".pi")
        cwd = os.path.join(tmpdir, "project")
        os.makedirs(cwd, exist_ok=True)

        # Write initial settings
        os.makedirs(agent_dir, exist_ok=True)
        Path(os.path.join(agent_dir, "settings.json")).write_text(
            json.dumps({"theme": "dark", "quietStartup": True}),
            encoding="utf-8",
        )

        mgr = SettingsManager.create(cwd, agent_dir)

        # Externally modify theme
        Path(os.path.join(agent_dir, "settings.json")).write_text(
            json.dumps({"theme": "light", "quietStartup": True}),
            encoding="utf-8",
        )

        # Modify only defaultModel through the manager
        mgr.set_default_model("claude")

        # External theme change should be preserved
        content = json.loads(Path(os.path.join(agent_dir, "settings.json")).read_text(encoding="utf-8"))
        assert content["theme"] == "light"
        assert content["defaultModel"] == "claude"
