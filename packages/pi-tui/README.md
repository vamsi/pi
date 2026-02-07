# pi-tui

Terminal UI framework with differential rendering. Manages raw terminal output, cursor positioning, ANSI escape sequences, keyboard input parsing, overlays, components, and efficient screen updates.

This is not a widget library that wraps curses. It works directly with escape sequences and maintains its own rendering pipeline. Components produce lines of text. The TUI tracks what changed between frames and writes only the diff.

## Core abstractions

### Terminal

The `Terminal` protocol defines the rendering surface. `ProcessTerminal` is the real implementation that writes to stdout. Tests use a `VirtualTerminal` that captures output.

```python
from pi.tui import ProcessTerminal

term = ProcessTerminal()
term.write("Hello\n")
term.flush()
print(term.width, term.height)
```

### TUI

`TUI` owns the render loop. You give it a `Terminal`, set a root `Component`, and call `render()`.

```python
from pi.tui import TUI, Text, ProcessTerminal

term = ProcessTerminal()
tui = TUI(term)
tui.set_root(Text("Hello, terminal."))
tui.render()
```

### Component

The `Component` protocol requires one method:

```python
class Component(Protocol):
    def render(self, width: int) -> list[str]:
        """Return lines of text to display, each fitting within `width` columns."""
        ...
```

Lines can contain ANSI escape codes. The framework handles visible width calculation correctly even with escapes, wide characters (CJK), and grapheme clusters.

### Container

Group components vertically:

```python
from pi.tui import Container, Text, Spacer

container = Container()
container.add(Text("Header"))
container.add(Spacer(1))
container.add(Text("Body content here."))
```

### Focusable

Components that accept keyboard input implement the `Focusable` protocol:

```python
class Focusable(Protocol):
    def handle_key(self, key: Key) -> bool:
        """Handle a key press. Return True if consumed."""
        ...

    def get_cursor_position(self) -> tuple[int, int] | None:
        """Return (row, col) for cursor placement, or None to hide cursor."""
        ...
```

## Built-in components

| Component | Purpose |
|-----------|---------|
| `Text` | Static text (supports ANSI, word wrap) |
| `TruncatedText` | Text truncated to fit width with ellipsis |
| `Box` | Bordered container with optional title |
| `Input` | Single-line text input with cursor, selection, history |
| `Editor` | Multi-line text editor with undo, kill ring, word movement |
| `Markdown` | Renders Markdown to terminal (code blocks, lists, bold, italic, links) |
| `SelectList` | Scrollable list with keyboard selection |
| `SettingsList` | Key-value settings with inline editing |
| `Loader` | Animated loading spinner |
| `CancellableLoader` | Loader with cancel callback |
| `Image` | Terminal image rendering (Kitty, iTerm2 protocols) |
| `Spacer` | Empty vertical space |

## Keyboard input

### Parsing

The `parse_key` function converts raw escape sequences into structured `Key` objects:

```python
from pi.tui import parse_key

key = parse_key("\x1b[A")    # Up arrow
key = parse_key("\r")        # Enter
key = parse_key("a")         # Letter 'a'
key = parse_key("\x1ba")     # Alt+a
key = parse_key("\x01")      # Ctrl+a
```

`Key` has fields: `id` (the key identifier), `shift`, `alt`, `ctrl`, `meta`, `event_type` (press, repeat, release).

### Matching

```python
from pi.tui import matches_key

if matches_key(key, "enter"):
    submit()
elif matches_key(key, "ctrl+c"):
    abort()
elif matches_key(key, "alt+backspace"):
    delete_word()
```

### StdinBuffer

Async stdin reader that accumulates raw bytes and yields parsed keys:

```python
from pi.tui import StdinBuffer

buffer = StdinBuffer()
buffer.feed(raw_bytes)
keys = buffer.process()
for key in keys:
    handle(key)
```

## Overlays

The TUI supports floating overlays anchored to positions in the main content:

```python
from pi.tui import TUI, OverlayOptions, OverlayAnchor, OverlayMargin

handle = tui.show_overlay(
    component=my_dropdown,
    options=OverlayOptions(
        anchor=OverlayAnchor.CURSOR,
        margin=OverlayMargin(top=1, left=0),
        max_height=10,
    ),
)

# Later:
tui.hide_overlay(handle)
```

## Text utilities

```python
from pi.tui import visible_width, truncate_to_width, wrap_text_with_ansi

# Correct width calculation with ANSI escapes and wide characters
visible_width("\x1b[31mhello\x1b[0m")  # 5, not 14

# Truncate to terminal width
truncate_to_width("long string...", 10)  # "long stri..."

# Word wrap preserving ANSI escape codes
lines = wrap_text_with_ansi("long text with \x1b[1mbold\x1b[0m words", width=20)
```

## Fuzzy matching

```python
from pi.tui import fuzzy_match, fuzzy_filter

match = fuzzy_match("abc", "a_big_cat")
# FuzzyMatch(score=..., indices=[0, 2, 6])

results = fuzzy_filter("usr", ["user", "username", "cursor", "bus"])
# Sorted by match score
```

## Keybindings

```python
from pi.tui import get_editor_keybindings, set_editor_keybindings, EditorAction

bindings = get_editor_keybindings()

# Rebind an action
bindings[EditorAction.SUBMIT] = ["ctrl+enter"]
bindings[EditorAction.NEWLINE] = ["enter"]
set_editor_keybindings(bindings)
```

## Image rendering

Supports Kitty and iTerm2 inline image protocols:

```python
from pi.tui import render_image, ImageRenderOptions

lines = render_image(
    image_bytes,
    ImageRenderOptions(max_width=80, max_height=24),
)
```

The framework auto-detects terminal capabilities and falls back to a text placeholder when image protocols are unavailable.

## File structure

```
src/pi/tui/
    __init__.py           Public exports
    tui.py                TUI engine, Component/Container/Focusable protocols, rendering
    terminal.py           Terminal protocol and ProcessTerminal implementation
    keys.py               Key parsing (escape sequences, Kitty protocol, modifiers)
    keybindings.py        Editor action keybinding management
    stdin_buffer.py       Async stdin buffering and key extraction
    autocomplete.py       Autocomplete provider framework
    editor_component.py   Editor component protocol
    fuzzy.py              Fuzzy string matching
    kill_ring.py          Emacs-style kill ring for cut/paste
    undo_stack.py         Undo/redo stack
    utils.py              visible_width, truncate, word wrap, ANSI utilities
    terminal_image.py     Image protocol detection and encoding
    components/
        text.py           Text, TruncatedText
        box.py            Box (bordered container)
        input.py          Single-line input
        editor.py         Multi-line editor
        markdown.py       Markdown renderer
        select_list.py    Scrollable selection list
        settings_list.py  Key-value settings editor
        loader.py         Loading spinner
        cancellable_loader.py  Loader with cancel
        image.py          Inline image component
        spacer.py         Vertical spacer
```

## Testing

709 tests covering all components, rendering, keyboard input, and edge cases:

```bash
uv run pytest packages/pi-tui/tests/ -v
```

Tests use a `VirtualTerminal` that captures output without needing a real terminal. This makes all rendering behavior fully testable.
