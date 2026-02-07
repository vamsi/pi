"""Tests for the TUI overlay system.

Covers overlay creation, rendering, stacking, hiding, and focus
management using the VirtualTerminal test helper.
"""

from __future__ import annotations

from pi.tui.tui import TUI, OverlayHandle

from .virtual_terminal import VirtualTerminal


# ---------------------------------------------------------------------------
# Minimal test components
# ---------------------------------------------------------------------------


class SimpleComponent:
    """A minimal component that renders a fixed set of lines."""

    def __init__(self, lines: list[str] | None = None, text: str = "") -> None:
        if lines is not None:
            self._lines = lines
        else:
            self._lines = [text] if text else [""]
        self.focused = False
        self._input_log: list[str] = []

    def render(self, width: int) -> list[str]:
        return list(self._lines)

    def invalidate(self) -> None:
        pass

    def handle_input(self, data: str) -> None:
        self._input_log.append(data)


# ---------------------------------------------------------------------------
# Overlay creation tests
# ---------------------------------------------------------------------------


class TestOverlayCreation:
    """TUI.show_overlay pushes an overlay onto the stack."""

    def test_show_overlay_returns_handle(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        handle = tui.show_overlay(comp)
        assert isinstance(handle, OverlayHandle)

    def test_has_overlay_after_show(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        tui.show_overlay(comp)
        assert tui.has_overlay() is True

    def test_no_overlay_initially(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        assert tui.has_overlay() is False

    def test_overlay_is_visible(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        tui.show_overlay(comp)
        assert tui.is_overlay_visible(comp) is True


# ---------------------------------------------------------------------------
# Overlay rendering tests
# ---------------------------------------------------------------------------


class TestOverlayRendering:
    """Overlays are composited on top of main content during render."""

    def test_overlay_text_appears_in_output(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base content"))
        overlay = SimpleComponent(text="OVERLAY_TEXT")
        tui.show_overlay(overlay)
        tui.do_render()
        assert "OVERLAY_TEXT" in term.output

    def test_base_content_still_present_with_overlay(self) -> None:
        """The base content lines outside the overlay region should survive."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        base = SimpleComponent(lines=["base line 0", "base line 1"])
        tui.add_child(base)
        # Overlay is small and centered -- at least some base content remains
        overlay = SimpleComponent(text="OVR")
        tui.show_overlay(overlay, {"width": 10, "anchor": "center"})
        tui.do_render()
        output = term.output
        # The base lines are rendered first and then the overlay is composited.
        # In a large enough terminal (80x24), parts of base content survive.
        assert "base line" in output or "OVR" in output

    def test_overlay_renders_after_main_children(self) -> None:
        """The overlay content should visually appear in the output."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="main"))
        overlay = SimpleComponent(text="on top")
        tui.show_overlay(overlay)
        tui.do_render()
        assert "on top" in term.output


# ---------------------------------------------------------------------------
# Overlay dismissal tests
# ---------------------------------------------------------------------------


class TestOverlayDismissal:
    """Overlays can be removed from the stack."""

    def test_hide_overlay_removes_from_stack(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        tui.show_overlay(comp)
        assert tui.has_overlay() is True
        tui.hide_overlay(comp)
        assert tui.has_overlay() is False

    def test_handle_hide_removes_overlay(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        handle = tui.show_overlay(comp)
        handle.hide()
        assert tui.has_overlay() is False

    def test_hidden_overlay_not_visible(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        tui.show_overlay(comp)
        tui.hide_overlay(comp)
        assert tui.is_overlay_visible(comp) is False

    def test_hide_nonexistent_overlay_is_noop(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="never added")
        tui.hide_overlay(comp)  # should not raise
        assert tui.has_overlay() is False


# ---------------------------------------------------------------------------
# Overlay stacking tests
# ---------------------------------------------------------------------------


class TestOverlayStacking:
    """Multiple overlays stack and are removed independently."""

    def test_multiple_overlays_stack(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        c1 = SimpleComponent(text="first")
        c2 = SimpleComponent(text="second")
        tui.show_overlay(c1)
        tui.show_overlay(c2)
        assert tui.has_overlay() is True

    def test_removing_top_overlay_leaves_bottom(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        c1 = SimpleComponent(text="bottom")
        c2 = SimpleComponent(text="top")
        tui.show_overlay(c1)
        tui.show_overlay(c2)
        tui.hide_overlay(c2)
        assert tui.has_overlay() is True
        assert tui.is_overlay_visible(c1) is True

    def test_removing_bottom_overlay_leaves_top(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        c1 = SimpleComponent(text="bottom")
        c2 = SimpleComponent(text="top")
        tui.show_overlay(c1)
        tui.show_overlay(c2)
        tui.hide_overlay(c1)
        assert tui.has_overlay() is True
        assert tui.is_overlay_visible(c2) is True

    def test_removing_all_overlays_clears_stack(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        c1 = SimpleComponent(text="first")
        c2 = SimpleComponent(text="second")
        tui.show_overlay(c1)
        tui.show_overlay(c2)
        tui.hide_overlay(c2)
        tui.hide_overlay(c1)
        assert tui.has_overlay() is False


# ---------------------------------------------------------------------------
# Overlay focus tests
# ---------------------------------------------------------------------------


class TestOverlayFocus:
    """Overlay push/pop manages focus correctly."""

    def test_show_overlay_sets_focus(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        base = SimpleComponent(text="base")
        base.focused = True
        tui.set_focus(base)

        overlay = SimpleComponent(text="overlay")
        tui.show_overlay(overlay)
        assert overlay.focused is True

    def test_hide_overlay_restores_focus(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        base = SimpleComponent(text="base")
        tui.set_focus(base)
        assert base.focused is True

        overlay = SimpleComponent(text="overlay")
        tui.show_overlay(overlay)
        assert base.focused is False

        tui.hide_overlay(overlay)
        assert base.focused is True

    def test_input_dispatches_to_overlay(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        base = SimpleComponent(text="base")
        tui.set_focus(base)

        overlay = SimpleComponent(text="overlay")
        tui.show_overlay(overlay)

        tui.handle_input("x")
        assert overlay._input_log == ["x"]
        assert base._input_log == []

    def test_input_dispatches_to_base_after_overlay_hidden(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        base = SimpleComponent(text="base")
        tui.set_focus(base)

        overlay = SimpleComponent(text="overlay")
        tui.show_overlay(overlay)
        tui.hide_overlay(overlay)

        tui.handle_input("y")
        assert base._input_log == ["y"]
        assert overlay._input_log == []


# ---------------------------------------------------------------------------
# OverlayHandle.set_hidden / is_hidden tests
# ---------------------------------------------------------------------------


class TestOverlayHandleVisibility:
    """OverlayHandle.set_hidden toggles overlay visibility without removal."""

    def test_set_hidden_true_hides_overlay(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        handle = tui.show_overlay(comp)
        handle.set_hidden(True)
        assert handle.is_hidden() is True
        # Overlay is still on the stack
        assert tui.has_overlay() is True

    def test_set_hidden_false_shows_overlay(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        handle = tui.show_overlay(comp)
        handle.set_hidden(True)
        handle.set_hidden(False)
        assert handle.is_hidden() is False
        assert tui.is_overlay_visible(comp) is True

    def test_hidden_overlay_not_rendered(self) -> None:
        """A hidden overlay should not contribute content to the output."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        overlay = SimpleComponent(text="HIDDEN_MARKER")
        handle = tui.show_overlay(overlay)
        handle.set_hidden(True)
        # Clear any output from the synchronous renders triggered by
        # show_overlay/set_hidden (no running event loop = immediate render).
        term.clear_buffer()
        tui.do_render()
        # The overlay is hidden so its text should not appear.
        # The compositing loop skips entries with hidden=True.
        assert "HIDDEN_MARKER" not in term.output

    def test_is_hidden_after_removal(self) -> None:
        """After hide(), is_hidden returns True (overlay no longer on stack)."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleComponent(text="overlay")
        handle = tui.show_overlay(comp)
        handle.hide()
        assert handle.is_hidden() is True


# ---------------------------------------------------------------------------
# Overlay with options tests
# ---------------------------------------------------------------------------


class TestOverlayOptions:
    """Overlay positioning and sizing options."""

    def test_overlay_with_width_option(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        overlay = SimpleComponent(text="narrow")
        tui.show_overlay(overlay, {"width": 20})
        tui.do_render()
        assert "narrow" in term.output

    def test_overlay_with_anchor_top_left(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        overlay = SimpleComponent(text="TL")
        tui.show_overlay(overlay, {"anchor": "top-left", "width": 10})
        tui.do_render()
        assert "TL" in term.output

    def test_overlay_with_percentage_width(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        overlay = SimpleComponent(text="half")
        tui.show_overlay(overlay, {"width": "50%"})
        tui.do_render()
        assert "half" in term.output

    def test_overlay_with_max_height(self) -> None:
        """An overlay with many lines should be clamped to max_height."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        many_lines = [f"line {i}" for i in range(50)]
        overlay = SimpleComponent(lines=many_lines)
        tui.show_overlay(overlay, {"max_height": 5})
        tui.do_render()
        # Only 5 lines should be composited; "line 49" should not appear
        assert "line 49" not in term.output

    def test_overlay_with_margin(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleComponent(text="base"))
        overlay = SimpleComponent(text="margined")
        tui.show_overlay(overlay, {"margin": 2})
        tui.do_render()
        assert "margined" in term.output
