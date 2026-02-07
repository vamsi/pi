"""pi-tui: Terminal UI framework with differential rendering."""

# Autocomplete support
from pi.tui.autocomplete import (
    AutocompleteItem,
    AutocompleteProvider,
    CombinedAutocompleteProvider,
    SlashCommand,
)

# Components (re-exported from components package)
from pi.tui.components import (
    Box,
    CancellableLoader,
    DefaultTextStyle,
    Editor,
    EditorOptions,
    EditorTheme,
    Image,
    ImageOptions,
    ImageTheme,
    Input,
    Loader,
    Markdown,
    MarkdownTheme,
    SelectItem,
    SelectList,
    SelectListTheme,
    SettingItem,
    SettingsList,
    SettingsListOptions,
    SettingsListTheme,
    Spacer,
    Text,
    TruncatedText,
)

# Editor component interface
from pi.tui.editor_component import EditorComponent

# Fuzzy matching
from pi.tui.fuzzy import FuzzyMatch, fuzzy_filter, fuzzy_match

# Keybindings
from pi.tui.keybindings import (
    DEFAULT_EDITOR_KEYBINDINGS,
    EditorAction,
    EditorKeybindingsManager,
    get_editor_keybindings,
    set_editor_keybindings,
)

# Keyboard input handling
from pi.tui.keys import (
    Key,
    KeyEventType,
    KeyId,
    is_key_release,
    is_key_repeat,
    is_kitty_protocol_active,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)

# Input buffering
from pi.tui.stdin_buffer import StdinBuffer

# Terminal interface and implementations
from pi.tui.terminal import ProcessTerminal, Terminal

# Terminal image support
from pi.tui.terminal_image import (
    CellDimensions,
    ImageDimensions,
    ImageProtocol,
    ImageRenderOptions,
    TerminalCapabilities,
    allocate_image_id,
    calculate_image_rows,
    delete_all_kitty_images,
    delete_kitty_image,
    detect_capabilities,
    encode_iterm2,
    encode_kitty,
    get_capabilities,
    get_cell_dimensions,
    get_gif_dimensions,
    get_image_dimensions,
    get_jpeg_dimensions,
    get_png_dimensions,
    get_webp_dimensions,
    image_fallback,
    render_image,
    reset_capabilities_cache,
    set_cell_dimensions,
)

# Core TUI
from pi.tui.tui import (
    CURSOR_MARKER,
    Component,
    Container,
    Focusable,
    OverlayAnchor,
    OverlayHandle,
    OverlayMargin,
    OverlayOptions,
    SizeValue,
    TUI,
    is_focusable,
)

# Utilities
from pi.tui.utils import truncate_to_width, visible_width, wrap_text_with_ansi

__all__ = [
    # Autocomplete
    "AutocompleteItem",
    "AutocompleteProvider",
    "CombinedAutocompleteProvider",
    "SlashCommand",
    # Components
    "Box",
    "CancellableLoader",
    "DefaultTextStyle",
    "Editor",
    "EditorOptions",
    "EditorTheme",
    "Image",
    "ImageOptions",
    "ImageTheme",
    "Input",
    "Loader",
    "Markdown",
    "MarkdownTheme",
    "SelectItem",
    "SelectList",
    "SelectListTheme",
    "SettingItem",
    "SettingsList",
    "SettingsListOptions",
    "SettingsListTheme",
    "Spacer",
    "Text",
    "TruncatedText",
    # Editor component interface
    "EditorComponent",
    # Fuzzy
    "FuzzyMatch",
    "fuzzy_filter",
    "fuzzy_match",
    # Keybindings
    "DEFAULT_EDITOR_KEYBINDINGS",
    "EditorAction",
    "EditorKeybindingsManager",
    "get_editor_keybindings",
    "set_editor_keybindings",
    # Keys
    "Key",
    "KeyEventType",
    "KeyId",
    "is_key_release",
    "is_key_repeat",
    "is_kitty_protocol_active",
    "matches_key",
    "parse_key",
    "set_kitty_protocol_active",
    # Stdin buffer
    "StdinBuffer",
    # Terminal
    "ProcessTerminal",
    "Terminal",
    # Terminal image
    "CellDimensions",
    "ImageDimensions",
    "ImageProtocol",
    "ImageRenderOptions",
    "TerminalCapabilities",
    "allocate_image_id",
    "calculate_image_rows",
    "delete_all_kitty_images",
    "delete_kitty_image",
    "detect_capabilities",
    "encode_iterm2",
    "encode_kitty",
    "get_capabilities",
    "get_cell_dimensions",
    "get_gif_dimensions",
    "get_image_dimensions",
    "get_jpeg_dimensions",
    "get_png_dimensions",
    "get_webp_dimensions",
    "image_fallback",
    "render_image",
    "reset_capabilities_cache",
    "set_cell_dimensions",
    # TUI core
    "CURSOR_MARKER",
    "Component",
    "Container",
    "Focusable",
    "OverlayAnchor",
    "OverlayHandle",
    "OverlayMargin",
    "OverlayOptions",
    "SizeValue",
    "TUI",
    "is_focusable",
    # Utilities
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]
