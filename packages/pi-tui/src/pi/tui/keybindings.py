"""Editor keybindings manager."""

from __future__ import annotations

from typing import Literal

from pi.tui.keys import KeyId, matches_key

EditorAction = Literal[
    # Cursor movement
    "cursorUp",
    "cursorDown",
    "cursorLeft",
    "cursorRight",
    "cursorWordLeft",
    "cursorWordRight",
    "cursorLineStart",
    "cursorLineEnd",
    "jumpForward",
    "jumpBackward",
    "pageUp",
    "pageDown",
    # Deletion
    "deleteCharBackward",
    "deleteCharForward",
    "deleteWordBackward",
    "deleteWordForward",
    "deleteToLineStart",
    "deleteToLineEnd",
    # Text input
    "newLine",
    "submit",
    "tab",
    # Selection/autocomplete
    "selectUp",
    "selectDown",
    "selectPageUp",
    "selectPageDown",
    "selectConfirm",
    "selectCancel",
    # Clipboard
    "copy",
    # Kill ring
    "yank",
    "yankPop",
    # Undo
    "undo",
    # Tool output
    "expandTools",
    # Session
    "toggleSessionPath",
    "toggleSessionSort",
    "renameSession",
    "deleteSession",
    "deleteSessionNoninvasive",
]

EditorKeybindingsConfig = dict[EditorAction, KeyId | list[KeyId]]

DEFAULT_EDITOR_KEYBINDINGS: dict[EditorAction, KeyId | list[KeyId]] = {
    # Cursor movement
    "cursorUp": "up",
    "cursorDown": "down",
    "cursorLeft": ["left", "ctrl+b"],
    "cursorRight": ["right", "ctrl+f"],
    "cursorWordLeft": ["alt+left", "ctrl+left", "alt+b"],
    "cursorWordRight": ["alt+right", "ctrl+right", "alt+f"],
    "cursorLineStart": ["home", "ctrl+a"],
    "cursorLineEnd": ["end", "ctrl+e"],
    "jumpForward": "ctrl+]",
    "jumpBackward": "ctrl+alt+]",
    "pageUp": "pageUp",
    "pageDown": "pageDown",
    # Deletion
    "deleteCharBackward": "backspace",
    "deleteCharForward": ["delete", "ctrl+d"],
    "deleteWordBackward": ["ctrl+w", "alt+backspace"],
    "deleteWordForward": ["alt+d", "alt+delete"],
    "deleteToLineStart": "ctrl+u",
    "deleteToLineEnd": "ctrl+k",
    # Text input
    "newLine": "shift+enter",
    "submit": "enter",
    "tab": "tab",
    # Selection/autocomplete
    "selectUp": "up",
    "selectDown": "down",
    "selectPageUp": "pageUp",
    "selectPageDown": "pageDown",
    "selectConfirm": "enter",
    "selectCancel": ["escape", "ctrl+c"],
    # Clipboard
    "copy": "ctrl+c",
    # Kill ring
    "yank": "ctrl+y",
    "yankPop": "alt+y",
    # Undo
    "undo": "ctrl+-",
    # Tool output
    "expandTools": "ctrl+o",
    # Session
    "toggleSessionPath": "ctrl+p",
    "toggleSessionSort": "ctrl+s",
    "renameSession": "ctrl+r",
    "deleteSession": "ctrl+d",
    "deleteSessionNoninvasive": "ctrl+backspace",
}


class EditorKeybindingsManager:
    """Manages keybindings for the editor."""

    def __init__(
        self, config: EditorKeybindingsConfig | None = None
    ) -> None:
        self._action_to_keys: dict[EditorAction, list[KeyId]] = {}
        self._build_maps(config or {})

    def _build_maps(self, config: EditorKeybindingsConfig) -> None:
        self._action_to_keys.clear()

        # Start with defaults
        for action, keys in DEFAULT_EDITOR_KEYBINDINGS.items():
            key_array = keys if isinstance(keys, list) else [keys]
            self._action_to_keys[action] = list(key_array)

        # Override with user config
        for action, keys in config.items():
            key_array = keys if isinstance(keys, list) else [keys]
            self._action_to_keys[action] = list(key_array)

    def matches(self, data: str, action: EditorAction) -> bool:
        """Check if input matches a specific action."""
        keys = self._action_to_keys.get(action)
        if not keys:
            return False
        for key in keys:
            if matches_key(data, key):
                return True
        return False

    def get_keys(self, action: EditorAction) -> list[KeyId]:
        """Get keys bound to an action."""
        return self._action_to_keys.get(action, [])

    def set_config(self, config: EditorKeybindingsConfig) -> None:
        """Update configuration."""
        self._build_maps(config)


_global_editor_keybindings: EditorKeybindingsManager | None = None


def get_editor_keybindings() -> EditorKeybindingsManager:
    global _global_editor_keybindings
    if _global_editor_keybindings is None:
        _global_editor_keybindings = EditorKeybindingsManager()
    return _global_editor_keybindings


def set_editor_keybindings(manager: EditorKeybindingsManager) -> None:
    global _global_editor_keybindings
    _global_editor_keybindings = manager
