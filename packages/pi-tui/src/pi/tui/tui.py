"""Core TUI framework with differential rendering.

Provides the ``Component`` and ``Focusable`` protocols, a ``Container`` class
for composing child components, and the main ``TUI`` class that drives
rendering, input dispatch, overlay management, and hardware-cursor positioning
against a ``Terminal`` back-end.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

from pi.tui.keys import is_key_release, matches_key
from pi.tui.terminal_image import (
    CellDimensions,
    get_capabilities,
    is_image_line,
    set_cell_dimensions,
)
from pi.tui.utils import (
    extract_segments,
    slice_by_column,
    slice_with_width,
    visible_width,
)

if TYPE_CHECKING:
    from pi.tui.terminal import Terminal

# ---------------------------------------------------------------------------
# Re-export visible_width for convenience (mirrors the TS ``export``)
# ---------------------------------------------------------------------------

__all__ = [
    "Component",
    "Focusable",
    "is_focusable",
    "CURSOR_MARKER",
    "visible_width",
    "OverlayAnchor",
    "OverlayMargin",
    "SizeValue",
    "OverlayOptions",
    "OverlayHandle",
    "Container",
    "TUI",
]

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Component(Protocol):
    """A renderable terminal component.

    ``handle_input`` and ``wants_key_release`` are optional -- checked
    at call-sites via ``getattr`` / ``hasattr``.
    """

    def render(self, width: int) -> list[str]:
        """Render the component into a list of terminal lines."""
        ...

    def invalidate(self) -> None:
        """Mark the component as needing a re-render."""
        ...

    # NOTE: handle_input and wants_key_release are intentionally *not*
    # declared here.  In the TypeScript source they are optional interface
    # members (``handleInput?(data: string): void`` and
    # ``wantsKeyRelease?: boolean``).  Python protocols have no native
    # "optional member" syntax, so we use ``getattr`` / ``hasattr`` at
    # each call-site instead.


@runtime_checkable
class Focusable(Protocol):
    """A component that can receive focus."""

    focused: bool


def is_focusable(component: object | None) -> bool:
    """Type-guard: return ``True`` if *component* implements ``Focusable``.

    Mirrors the TypeScript ``isFocusable`` function.
    """
    return component is not None and hasattr(component, "focused")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURSOR_MARKER = "\x1b_pi:c\x07"

# Regex that matches the CURSOR_MARKER inside a string
_CURSOR_MARKER_RE = re.compile(re.escape(CURSOR_MARKER))

# Reset SGR sequence
_SEGMENT_RESET = "\x1b[0m"

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

OverlayAnchor = Literal[
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "top-center",
    "bottom-center",
    "left-center",
    "right-center",
]


class OverlayMargin(TypedDict, total=False):
    top: int
    right: int
    bottom: int
    left: int


# int  ->  exact number of columns/rows
# str  ->  percentage string like "50%"
SizeValue = Union[int, str]


def _parse_size_value(
    value: SizeValue | None, reference_size: int
) -> int | None:
    """Resolve a ``SizeValue`` against *reference_size*.

    * ``None``  -> ``None``
    * ``int``   -> returned as-is
    * ``"50%"`` -> ``math.floor(reference_size * 50 / 100)``
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.endswith("%"):
        try:
            pct = float(value[:-1])
            return math.floor(reference_size * pct / 100)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# OverlayOptions / OverlayHandle
# ---------------------------------------------------------------------------


class OverlayOptions(TypedDict, total=False):
    width: SizeValue
    min_width: int
    max_height: SizeValue
    anchor: OverlayAnchor
    offset_x: int
    offset_y: int
    row: SizeValue
    col: SizeValue
    margin: OverlayMargin | int
    visible: Callable[[int, int], bool]


class OverlayHandle:
    """Handle returned by :meth:`TUI.show_overlay`.

    Allows the caller to hide or query the overlay without holding a
    direct reference to the overlay stack entry.
    """

    def __init__(self, tui: TUI, component: object) -> None:
        self._tui = tui
        self._component = component

    def hide(self) -> None:
        """Remove the overlay from the stack."""
        self._tui.hide_overlay(self._component)  # type: ignore[arg-type]

    def set_hidden(self, hidden: bool) -> None:
        """Toggle visibility without removing from the stack."""
        for entry in self._tui._overlay_stack:
            if entry["component"] is self._component:
                entry["hidden"] = hidden
                self._tui.invalidate()
                return

    def is_hidden(self) -> bool:
        """Return ``True`` if the overlay is currently hidden."""
        for entry in self._tui._overlay_stack:
            if entry["component"] is self._component:
                return entry["hidden"]
        return True


# ---------------------------------------------------------------------------
# Overlay stack entry (internal)
# ---------------------------------------------------------------------------


class _OverlayEntry(TypedDict):
    component: object  # Component
    options: OverlayOptions | None
    pre_focus: object | None  # Component | None
    hidden: bool


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class Container:
    """A simple container that renders its children sequentially.

    Mirrors the TypeScript ``Container implements Component``.
    """

    def __init__(self) -> None:
        self.children: list[object] = []  # list[Component]

    def add_child(self, component: object) -> None:
        """Append *component* to the children list."""
        self.children.append(component)

    def remove_child(self, component: object) -> None:
        """Remove *component* from the children list (no-op if absent)."""
        try:
            self.children.remove(component)
        except ValueError:
            pass

    def clear(self) -> None:
        """Remove all children."""
        self.children.clear()

    def invalidate(self) -> None:
        """Invalidate every child."""
        for child in self.children:
            inv = getattr(child, "invalidate", None)
            if inv is not None:
                inv()

    def render(self, width: int) -> list[str]:
        """Render all children and concatenate their line output."""
        lines: list[str] = []
        for child in self.children:
            rendered = getattr(child, "render", None)
            if rendered is not None:
                lines.extend(rendered(width))
        return lines


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------


class TUI(Container):
    """Main TUI controller: rendering, input, overlays, cursor management.

    This is the core of the TUI framework.  It extends ``Container`` (so it
    can hold child components) and adds:

    * Differential rendering -- only changed lines are re-written.
    * Overlay management -- modal/non-modal panels composited on top.
    * Hardware cursor positioning.
    * Input dispatching to the focused component or topmost overlay.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
    ) -> None:
        super().__init__()

        self.terminal: Terminal = terminal

        # Previous render state (for differential updates)
        self._previous_lines: list[str] = []
        self._previous_width: int = 0

        # Focus
        self._focused_component: object | None = None  # Component | None

        # Debug callback
        self.on_debug: Callable[[], None] | None = None

        # Render scheduling
        self._render_requested: bool = False

        # Cursor tracking
        self._cursor_row: int = 0
        self._hardware_cursor_row: int = 0

        # Input buffering
        self._input_buffer: str = ""

        # Cell-size query
        self._cell_size_query_pending: bool = False

        # Hardware cursor visibility
        self._show_hardware_cursor: bool = (
            show_hardware_cursor
            if show_hardware_cursor is not None
            else os.environ.get("PI_HARDWARE_CURSOR") == "1"
        )

        # Whether to issue a full-screen clear when content shrinks
        self._clear_on_shrink: bool = (
            os.environ.get("PI_CLEAR_ON_SHRINK") == "1"
        )

        # High-water mark for rendered line count
        self._max_lines_rendered: int = 0

        # Viewport tracking (for scroll detection)
        self._previous_viewport_top: int = 0

        # Metrics
        self._full_redraw_count: int = 0

        # Lifecycle
        self._stopped: bool = False

        # Overlay stack
        self._overlay_stack: list[_OverlayEntry] = []

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def full_redraws(self) -> int:
        """Number of full (non-differential) redraws performed."""
        return self._full_redraw_count

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, value: bool) -> None:
        self._show_hardware_cursor = value
        self.invalidate()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_clear_on_shrink(self, value: bool) -> None:
        self._clear_on_shrink = value

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    def set_focus(self, component: object | None) -> None:
        """Set the focused component, unfocusing the previous one."""
        if self._focused_component is component:
            return

        # Unfocus previous
        prev = self._focused_component
        if is_focusable(prev):
            prev.focused = False  # type: ignore[union-attr]

        self._focused_component = component

        # Focus new
        if is_focusable(component):
            component.focused = True  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Overlay management
    # ------------------------------------------------------------------

    def show_overlay(
        self,
        component: object,
        options: OverlayOptions | None = None,
    ) -> OverlayHandle:
        """Push an overlay onto the stack and give it focus."""
        entry: _OverlayEntry = {
            "component": component,
            "options": options,
            "pre_focus": self._focused_component,
            "hidden": False,
        }
        self._overlay_stack.append(entry)
        self.set_focus(component)
        self.invalidate()
        return OverlayHandle(self, component)

    def hide_overlay(self, component: object) -> None:
        """Remove *component* from the overlay stack and restore focus."""
        idx: int | None = None
        for i, entry in enumerate(self._overlay_stack):
            if entry["component"] is component:
                idx = i
                break

        if idx is None:
            return

        entry = self._overlay_stack.pop(idx)
        pre_focus = entry["pre_focus"]

        # If the removed overlay was the focused component, restore focus
        if self._focused_component is component:
            if self._overlay_stack:
                topmost = self._get_topmost_visible_overlay()
                if topmost is not None:
                    self.set_focus(topmost)
                else:
                    self.set_focus(pre_focus)
            else:
                self.set_focus(pre_focus)

        self.invalidate()

    def has_overlay(self) -> bool:
        """Return ``True`` if there are any overlays on the stack."""
        return len(self._overlay_stack) > 0

    def is_overlay_visible(self, component: object) -> bool:
        """Return ``True`` if *component* is on the overlay stack and not hidden."""
        for entry in self._overlay_stack:
            if entry["component"] is component:
                return not entry["hidden"]
        return False

    def _get_topmost_visible_overlay(self) -> object | None:
        """Return the topmost non-hidden overlay component, or ``None``."""
        for entry in reversed(self._overlay_stack):
            if not entry["hidden"]:
                return entry["component"]
        return None

    # ------------------------------------------------------------------
    # Invalidation / lifecycle
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Schedule a re-render (no-op if stopped)."""
        if self._stopped:
            return
        self.request_render()

    def start(self) -> None:
        """Start (or restart) the TUI render loop."""
        self._stopped = False
        self.request_render()

    def query_cell_size(self) -> None:
        """Send a ``CSI 16 t`` query to determine the cell pixel dimensions.

        The response is parsed inside :meth:`handle_input` via
        :meth:`_parse_cell_size_response`.
        """
        if self._cell_size_query_pending:
            return
        self._cell_size_query_pending = True
        # CSI 16 t -- report cell size in pixels
        self.terminal.write("\x1b[16t")

    def stop(self) -> None:
        """Stop the TUI and move the terminal cursor below content."""
        self._stopped = True
        # Move cursor below the last rendered content
        lines_below = len(self._previous_lines) - self._cursor_row - 1
        if lines_below > 0:
            self.terminal.write(f"\x1b[{lines_below}B")
        self.terminal.write("\n")

    # ------------------------------------------------------------------
    # Render scheduling
    # ------------------------------------------------------------------

    def request_render(self) -> None:
        """Schedule a render on the next event-loop tick.

        Equivalent to ``process.nextTick(cb)`` in the TypeScript source.
        Multiple calls coalesce into a single render pass.
        """
        if self._render_requested:
            return
        self._render_requested = True
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon(self._do_render_tick)
        except RuntimeError:
            # No running event loop -- render synchronously
            self._do_render_tick()

    def _do_render_tick(self) -> None:
        """Callback fired from the event loop to execute a render."""
        self._render_requested = False
        if self._stopped:
            return
        self.do_render()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_input(self, data: str) -> None:
        """Dispatch terminal input to the appropriate component.

        * Cell-size responses are intercepted and parsed.
        * Key-release events are forwarded only to components that opt in.
        * ``ctrl+shift+alt+d`` triggers the debug callback.
        * Otherwise input goes to the topmost visible overlay, or failing
          that, to the focused component.
        """
        if self._stopped:
            return

        # Intercept cell-size response: ESC[6;<height>;<width>t
        if self._cell_size_query_pending and "\x1b[6;" in data:
            remaining = self._parse_cell_size_response(data)
            if remaining is None or remaining == "":
                return
            data = remaining

        # Key-release filtering
        if is_key_release(data):
            focused = self._focused_component
            if focused is not None and getattr(
                focused, "wants_key_release", False
            ):
                handler = getattr(focused, "handle_input", None)
                if callable(handler):
                    handler(data)
            return

        # Debug hook
        if matches_key(data, "ctrl+shift+alt+d"):
            if self.on_debug is not None:
                self.on_debug()
            return

        # Overlay dispatch
        topmost = self._get_topmost_visible_overlay()
        if topmost is not None:
            handler = getattr(topmost, "handle_input", None)
            if callable(handler):
                handler(data)
            return

        # Focused component dispatch
        if self._focused_component is not None:
            handler = getattr(self._focused_component, "handle_input", None)
            if callable(handler):
                handler(data)

    # ------------------------------------------------------------------
    # Cell-size response parsing
    # ------------------------------------------------------------------

    def _parse_cell_size_response(self, data: str) -> str | None:
        """Parse a cell-size response ``ESC[6;<h>;<w>t`` from *data*.

        On success, updates the global cell dimensions via
        ``set_cell_dimensions`` and returns any leftover data that was
        not part of the response.  Returns ``None`` when the entire
        string was consumed.  Returns the original *data* unchanged if
        no valid response is found.
        """
        m = re.search(r"\x1b\[6;(\d+);(\d+)t", data)
        if m is None:
            return data

        self._cell_size_query_pending = False
        height_px = int(m.group(1))
        width_px = int(m.group(2))

        if height_px > 0 and width_px > 0:
            set_cell_dimensions(
                CellDimensions(width_px=width_px, height_px=height_px)
            )

        before = data[: m.start()]
        after = data[m.end() :]
        remaining = before + after
        return remaining if remaining else None

    # ------------------------------------------------------------------
    # Overlay layout resolution
    # ------------------------------------------------------------------

    def _resolve_overlay_layout(
        self,
        options: OverlayOptions | None,
        term_width: int,
        term_height: int,
        content_height: int,
    ) -> dict[str, int]:
        """Compute an overlay's absolute position and clamped size.

        Returns a dict with keys ``"row"``, ``"col"``, ``"width"``,
        ``"height"``.
        """
        if options is None:
            options = {}

        # Normalise margin
        margin_raw = options.get("margin")
        if isinstance(margin_raw, int):
            margin: OverlayMargin = {
                "top": margin_raw,
                "right": margin_raw,
                "bottom": margin_raw,
                "left": margin_raw,
            }
        elif margin_raw is not None:
            margin = margin_raw
        else:
            margin = {}

        m_top = margin.get("top", 0)
        m_right = margin.get("right", 0)
        m_bottom = margin.get("bottom", 0)
        m_left = margin.get("left", 0)

        available_width = term_width - m_left - m_right
        available_height = term_height - m_top - m_bottom

        # --- Width ---
        width = _parse_size_value(options.get("width"), term_width)
        if width is None:
            width = available_width
        min_width = options.get("min_width")
        if min_width is not None and width < min_width:
            width = min_width
        width = max(1, min(width, term_width))

        # --- Height ---
        max_height_val = _parse_size_value(
            options.get("max_height"), term_height
        )
        height = content_height
        if max_height_val is not None and height > max_height_val:
            height = max_height_val
        height = max(1, min(height, max(1, available_height)))

        # --- Position ---
        explicit_row = _parse_size_value(options.get("row"), term_height)
        explicit_col = _parse_size_value(options.get("col"), term_width)

        anchor: OverlayAnchor = options.get("anchor", "center")  # type: ignore[assignment]
        offset_x: int = options.get("offset_x", 0)  # type: ignore[assignment]
        offset_y: int = options.get("offset_y", 0)  # type: ignore[assignment]

        if explicit_row is not None:
            row = explicit_row + offset_y
        else:
            row = (
                self._resolve_anchor_row(
                    anchor, term_height, height, m_top, m_bottom
                )
                + offset_y
            )

        if explicit_col is not None:
            col = explicit_col + offset_x
        else:
            col = (
                self._resolve_anchor_col(
                    anchor, term_width, width, m_left, m_right
                )
                + offset_x
            )

        # Clamp to terminal bounds
        row = max(0, min(row, term_height - 1))
        col = max(0, min(col, term_width - width))

        return {"row": row, "col": col, "width": width, "height": height}

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_anchor_row(
        anchor: OverlayAnchor,
        term_height: int,
        overlay_height: int,
        margin_top: int,
        margin_bottom: int,
    ) -> int:
        """Compute the starting row for an overlay based on its anchor."""
        if anchor in ("top-left", "top-right", "top-center"):
            return margin_top
        if anchor in ("bottom-left", "bottom-right", "bottom-center"):
            return term_height - overlay_height - margin_bottom
        # center, left-center, right-center
        available = term_height - margin_top - margin_bottom
        return margin_top + max(0, (available - overlay_height) // 2)

    @staticmethod
    def _resolve_anchor_col(
        anchor: OverlayAnchor,
        term_width: int,
        overlay_width: int,
        margin_left: int,
        margin_right: int,
    ) -> int:
        """Compute the starting column for an overlay based on its anchor."""
        if anchor in ("top-left", "bottom-left", "left-center"):
            return margin_left
        if anchor in ("top-right", "bottom-right", "right-center"):
            return term_width - overlay_width - margin_right
        # center, top-center, bottom-center
        available = term_width - margin_left - margin_right
        return margin_left + max(0, (available - overlay_width) // 2)

    # ------------------------------------------------------------------
    # Overlay compositing
    # ------------------------------------------------------------------

    def _composite_overlays(
        self,
        base_lines: list[str],
        term_width: int,
        term_height: int,
    ) -> list[str]:
        """Composite all visible overlays on top of *base_lines*.

        Returns a new list of lines with overlays merged in.
        """
        result = list(base_lines)

        # Pad to at least term_height lines so overlays near the bottom
        # have something to composite onto.
        while len(result) < term_height:
            result.append("")

        for entry in self._overlay_stack:
            if entry["hidden"]:
                continue

            component = entry["component"]
            options: OverlayOptions | None = entry["options"]

            # Visibility callback
            if options is not None:
                vis_fn = options.get("visible")
                if vis_fn is not None and not vis_fn(term_width, term_height):
                    continue

            # First pass: render at provisional width to learn content height
            provisional_layout = self._resolve_overlay_layout(
                options, term_width, term_height, term_height
            )
            overlay_width = provisional_layout["width"]

            render_fn = getattr(component, "render", None)
            if render_fn is None:
                continue
            overlay_lines: list[str] = render_fn(overlay_width)

            # Second pass: resolve with actual content height
            layout = self._resolve_overlay_layout(
                options, term_width, term_height, len(overlay_lines)
            )
            overlay_row = layout["row"]
            overlay_col = layout["col"]
            overlay_width = layout["width"]
            overlay_height = layout["height"]

            # Truncate to max height
            if len(overlay_lines) > overlay_height:
                overlay_lines = overlay_lines[:overlay_height]

            # Composite each overlay line
            for line_idx, overlay_line in enumerate(overlay_lines):
                target_row = overlay_row + line_idx
                if target_row < 0 or target_row >= len(result):
                    continue

                result[target_row] = self._composite_line_at(
                    result[target_row],
                    overlay_line,
                    overlay_col,
                    overlay_width,
                    term_width,
                )

        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_line_resets(line: str) -> str:
        """Append ``ESC[0m`` if the line contains SGR codes but lacks a
        trailing reset.  This prevents style leaking across composited
        segments.
        """
        if not line:
            return line
        # Quick check: if there is no CSI in the line, no reset needed
        if "\x1b[" not in line:
            return line
        if line.endswith(_SEGMENT_RESET):
            return line
        return line + _SEGMENT_RESET

    # ------------------------------------------------------------------

    def _composite_line_at(
        self,
        base_line: str,
        overlay_line: str,
        col: int,
        overlay_width: int,
        term_width: int,
    ) -> str:
        """Merge a single overlay line onto a base line at column *col*.

        The base line is split into a *before* segment ``[0, col)`` and
        an *after* segment ``[col + overlay_width, term_width)``.  The
        overlay occupies the gap in between.
        """
        after_start = col + overlay_width
        after_len = term_width - after_start
        if after_len < 0:
            after_len = 0

        # Extract the two flanking segments in a single pass
        before, after = extract_segments(
            base_line,
            col,
            after_start,
            after_len,
            strict_after=True,
        )

        # Ensure resets so styles don't leak
        before = self._apply_line_resets(before)

        # Pad the overlay to its declared width
        overlay_vis_w = visible_width(overlay_line)
        if overlay_vis_w < overlay_width:
            overlay_line = overlay_line + " " * (overlay_width - overlay_vis_w)
        overlay_line = self._apply_line_resets(overlay_line)

        parts: list[str] = []
        if before:
            parts.append(before)
        parts.append(overlay_line)
        if after:
            parts.append(after)

        return "".join(parts)

    # ------------------------------------------------------------------
    # Cursor extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cursor_position(
        lines: list[str],
    ) -> tuple[list[str], int, int]:
        """Find and remove the ``CURSOR_MARKER`` from *lines*.

        Returns ``(cleaned_lines, cursor_row, cursor_col)``.  If no
        marker is found the cursor defaults to ``(len(lines) - 1, 0)``.
        """
        cursor_row = max(0, len(lines) - 1)
        cursor_col = 0

        for row_idx, line in enumerate(lines):
            marker_pos = line.find(CURSOR_MARKER)
            if marker_pos != -1:
                # Visible column up to the marker
                before_marker = line[:marker_pos]
                cursor_col = visible_width(before_marker)
                cursor_row = row_idx

                # Remove the marker from the line
                cleaned = (
                    line[:marker_pos]
                    + line[marker_pos + len(CURSOR_MARKER) :]
                )
                # Produce a copy of the list to avoid mutating the caller's data
                lines = list(lines)
                lines[row_idx] = cleaned
                break

        return lines, cursor_row, cursor_col

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def do_render(self) -> None:  # noqa: C901 — inherently complex
        """Perform a differential (or full) render pass.

        This is the heart of the TUI.  It:

        1.  Renders the container children to produce *base_lines*.
        2.  Composites any visible overlays.
        3.  Extracts the logical cursor position.
        4.  Diffs against the previous frame and emits only the changed
            lines (or does a full redraw when the terminal width changed,
            the viewport scrolled, or content shrank).
        5.  Positions the terminal cursor (and optionally the hardware
            cursor) at the logical cursor location.
        """
        if self._stopped:
            return

        term_width: int = self.terminal.columns
        term_height: int = self.terminal.rows

        if term_width <= 0 or term_height <= 0:
            return

        # -- 1. Render base content -----------------------------------------
        base_lines = self.render(term_width)

        # -- 2. Composite overlays ------------------------------------------
        if self._overlay_stack:
            lines = self._composite_overlays(
                base_lines, term_width, term_height
            )
        else:
            lines = base_lines

        # Clamp to terminal height
        if len(lines) > term_height:
            lines = lines[:term_height]

        # -- 3. Extract cursor position -------------------------------------
        lines, cursor_row, cursor_col = self._extract_cursor_position(lines)

        # -- 4. Decide: full vs differential --------------------------------
        force_full = False

        if term_width != self._previous_width:
            force_full = True

        # Viewport shift (scrolling)
        viewport_top = 0
        if viewport_top != self._previous_viewport_top:
            force_full = True
            self._previous_viewport_top = viewport_top

        # Content shrank and the caller opted into clearing
        if self._clear_on_shrink and len(lines) < self._max_lines_rendered:
            force_full = True

        if force_full:
            self._full_redraw_count += 1

        # -- 5. Build output buffer -----------------------------------------
        out: list[str] = []

        # Navigate to row 0 of our content region
        if self._cursor_row > 0:
            out.append(f"\x1b[{self._cursor_row}A")
        out.append("\r")

        # Hide the hardware cursor while we paint
        if self._show_hardware_cursor:
            out.append("\x1b[?25l")

        num_new = len(lines)
        num_old = len(self._previous_lines)

        if force_full:
            # ---- Full redraw ----
            # Clear from the current position to the end of the screen
            out.append("\x1b[J")

            for i, line in enumerate(lines):
                if i > 0:
                    out.append("\n")

                if is_image_line(line):
                    # Image lines are emitted verbatim; no trailing clear
                    out.append(line)
                else:
                    out.append(line)
                    out.append("\x1b[K")
        else:
            # ---- Differential update ----
            total = max(num_new, num_old)
            for i in range(total):
                if i > 0:
                    out.append("\n")

                new_line = lines[i] if i < num_new else ""
                old_line = (
                    self._previous_lines[i] if i < num_old else ""
                )

                if i >= num_new:
                    # Line existed before but not now -- clear it
                    out.append("\r\x1b[K")
                elif new_line != old_line or is_image_line(new_line):
                    out.append("\r")
                    if is_image_line(new_line):
                        out.append(new_line)
                    else:
                        out.append(new_line)
                        out.append("\x1b[K")
                # else: line unchanged -- just skip (the "\n" above still
                # moves us to the next row)

        # -- 6. Update bookkeeping ------------------------------------------
        self._previous_lines = lines
        self._previous_width = term_width
        if num_new > self._max_lines_rendered:
            self._max_lines_rendered = num_new

        # -- 7. Position cursor at the logical cursor row/col ---------------
        last_written_row = max(0, (max(num_new, num_old) if not force_full else num_new) - 1)
        self._cursor_row = cursor_row

        # Move from last_written_row to cursor_row
        delta = last_written_row - cursor_row
        if delta > 0:
            out.append(f"\x1b[{delta}A")
        elif delta < 0:
            out.append(f"\x1b[{-delta}B")

        # Horizontal position
        out.append("\r")
        if cursor_col > 0:
            out.append(f"\x1b[{cursor_col}C")

        # -- 8. Hardware cursor ---------------------------------------------
        if self._show_hardware_cursor:
            self._position_hardware_cursor(out, cursor_row, cursor_col)

        # -- 9. Flush -------------------------------------------------------
        self.terminal.write("".join(out))

    # ------------------------------------------------------------------
    # Hardware cursor
    # ------------------------------------------------------------------

    def _position_hardware_cursor(
        self,
        out: list[str],
        cursor_row: int,
        cursor_col: int,
    ) -> None:
        """Append escape codes to *out* that make the hardware cursor
        visible and positioned at ``(cursor_row, cursor_col)``.

        Called at the end of :meth:`do_render` only when
        ``_show_hardware_cursor`` is ``True``.
        """
        # The software cursor has already been positioned; just un-hide.
        out.append("\x1b[?25h")
        self._hardware_cursor_row = cursor_row

    # ------------------------------------------------------------------
    # Debug dump
    # ------------------------------------------------------------------

    def _write_debug_dump(self) -> None:
        """Write the current render state to ``~/.pi/tui-debug/`` for
        post-mortem inspection.
        """
        try:
            debug_dir = os.path.join(os.path.expanduser("~"), ".pi", "tui-debug")
            os.makedirs(debug_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            dump_path = os.path.join(debug_dir, f"render-{ts}.txt")
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(
                    f"terminal: {self.terminal.columns}x{self.terminal.rows}\n"
                )
                f.write(f"cursor_row: {self._cursor_row}\n")
                f.write(f"hardware_cursor_row: {self._hardware_cursor_row}\n")
                f.write(f"show_hardware_cursor: {self._show_hardware_cursor}\n")
                f.write(f"clear_on_shrink: {self._clear_on_shrink}\n")
                f.write(f"full_redraws: {self._full_redraw_count}\n")
                f.write(f"max_lines_rendered: {self._max_lines_rendered}\n")
                f.write(f"overlay_count: {len(self._overlay_stack)}\n")
                f.write(f"stopped: {self._stopped}\n")
                f.write(f"\nprevious_lines ({len(self._previous_lines)}):\n")
                for i, line in enumerate(self._previous_lines):
                    f.write(f"  [{i:3d}] {line!r}\n")
                f.write(f"\noverlay_stack ({len(self._overlay_stack)}):\n")
                for i, entry in enumerate(self._overlay_stack):
                    comp = entry["component"]
                    hidden = entry["hidden"]
                    f.write(f"  [{i}] hidden={hidden} component={comp!r}\n")
        except OSError:
            pass
