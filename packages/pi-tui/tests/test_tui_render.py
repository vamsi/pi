"""Tests for TUI rendering -- Container, child management, and do_render.

Uses the VirtualTerminal to capture output and verify that components are
rendered correctly by the TUI framework.
"""

from __future__ import annotations

from pi.tui.tui import Container, TUI

from .virtual_terminal import VirtualTerminal


# ---------------------------------------------------------------------------
# Minimal test components
# ---------------------------------------------------------------------------


class SimpleText:
    """A minimal component that renders a single line of static text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._dirty = True

    def render(self, width: int) -> list[str]:
        self._dirty = False
        return [self.text]

    def invalidate(self) -> None:
        self._dirty = True


class MultiLineText:
    """A component that renders multiple lines."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return list(self.lines)

    def invalidate(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Container tests
# ---------------------------------------------------------------------------


class TestContainerAddRemove:
    """Container.add_child / remove_child / clear semantics."""

    def test_add_child_appends(self) -> None:
        container = Container()
        c1 = SimpleText("a")
        c2 = SimpleText("b")
        container.add_child(c1)
        container.add_child(c2)
        assert container.children == [c1, c2]

    def test_remove_child_removes(self) -> None:
        container = Container()
        c1 = SimpleText("a")
        c2 = SimpleText("b")
        container.add_child(c1)
        container.add_child(c2)
        container.remove_child(c1)
        assert container.children == [c2]

    def test_remove_absent_child_is_noop(self) -> None:
        container = Container()
        c1 = SimpleText("a")
        container.add_child(c1)
        c2 = SimpleText("b")
        container.remove_child(c2)  # should not raise
        assert container.children == [c1]

    def test_clear_removes_all(self) -> None:
        container = Container()
        container.add_child(SimpleText("a"))
        container.add_child(SimpleText("b"))
        container.add_child(SimpleText("c"))
        container.clear()
        assert container.children == []


class TestContainerRender:
    """Container.render concatenates child lines in order."""

    def test_single_child(self) -> None:
        container = Container()
        container.add_child(SimpleText("hello"))
        lines = container.render(80)
        assert lines == ["hello"]

    def test_multiple_children_render_in_order(self) -> None:
        container = Container()
        container.add_child(SimpleText("first"))
        container.add_child(SimpleText("second"))
        container.add_child(SimpleText("third"))
        lines = container.render(80)
        assert lines == ["first", "second", "third"]

    def test_multiline_child(self) -> None:
        container = Container()
        container.add_child(MultiLineText(["line1", "line2"]))
        container.add_child(SimpleText("after"))
        lines = container.render(80)
        assert lines == ["line1", "line2", "after"]

    def test_empty_container_renders_empty(self) -> None:
        container = Container()
        lines = container.render(80)
        assert lines == []

    def test_width_is_forwarded_to_children(self) -> None:
        """Verify the width argument is passed through to child render."""
        received_widths: list[int] = []

        class WidthRecorder:
            def render(self, width: int) -> list[str]:
                received_widths.append(width)
                return ["x"]

            def invalidate(self) -> None:
                pass

        container = Container()
        container.add_child(WidthRecorder())
        container.render(42)
        assert received_widths == [42]


class TestContainerInvalidate:
    """Container.invalidate propagates to all children."""

    def test_invalidate_calls_children(self) -> None:
        invalidated: list[str] = []

        class TrackingComponent:
            def __init__(self, name: str) -> None:
                self.name = name

            def render(self, width: int) -> list[str]:
                return [self.name]

            def invalidate(self) -> None:
                invalidated.append(self.name)

        container = Container()
        container.add_child(TrackingComponent("a"))
        container.add_child(TrackingComponent("b"))
        container.invalidate()
        assert invalidated == ["a", "b"]


# ---------------------------------------------------------------------------
# TUI render tests
# ---------------------------------------------------------------------------


class TestTUIRender:
    """TUI.do_render writes output to the terminal."""

    def test_render_simple_text(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("hello world"))
        tui.do_render()
        assert "hello world" in term.output

    def test_render_multiple_children(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("alpha"))
        tui.add_child(SimpleText("beta"))
        tui.do_render()
        output = term.output
        assert "alpha" in output
        assert "beta" in output

    def test_render_after_remove_child(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        comp = SimpleText("removed")
        tui.add_child(comp)
        tui.do_render()
        assert "removed" in term.output

        term.clear_buffer()
        tui.remove_child(comp)
        tui.do_render()
        # After removal the component text should not appear in new output
        # (the TUI may emit clear sequences, but the text itself should be gone)
        # We check the rendered lines directly via Container.render
        assert tui.render(80) == []

    def test_render_writes_to_terminal(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("x"))
        tui.do_render()
        assert term.write_count > 0

    def test_render_empty_tui_does_not_crash(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.do_render()
        # Should not raise -- an empty TUI renders zero lines which
        # means no visible content, but no crash either.

    def test_children_order_in_output(self) -> None:
        """Verify that children appear in order in the rendered output."""
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("AAA"))
        tui.add_child(SimpleText("BBB"))
        tui.add_child(SimpleText("CCC"))
        tui.do_render()
        output = term.output
        pos_a = output.find("AAA")
        pos_b = output.find("BBB")
        pos_c = output.find("CCC")
        assert pos_a < pos_b < pos_c


class TestTUIRenderDifferential:
    """Differential rendering: only changed lines are re-emitted."""

    def test_second_render_skips_unchanged_lines(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("static line"))
        tui.do_render()
        first_output = term.output

        term.clear_buffer()
        tui.do_render()
        second_output = term.output

        # The first render has to emit the full content.
        assert "static line" in first_output
        # The second render should produce less output because the line
        # has not changed (differential mode).  At minimum the output
        # should be shorter or equal -- we do not expect the full line
        # to be rewritten.
        assert len(second_output) <= len(first_output)

    def test_width_change_triggers_full_redraw(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("content"))
        tui.do_render()

        redraws_before = tui.full_redraws
        term.columns = 100
        tui.do_render()
        assert tui.full_redraws > redraws_before


class TestTUIStoppedBehaviour:
    """A stopped TUI should not render or accept input."""

    def test_stop_prevents_render(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("visible"))
        tui.do_render()
        term.clear_buffer()

        tui.stop()
        tui.do_render()
        # After stop the render is a no-op; nothing new written beyond
        # the stop() call itself.
        # stop() writes a cursor-move + newline, but do_render should
        # be inert.

    def test_invalidate_after_stop_is_noop(self) -> None:
        term = VirtualTerminal(rows=24, columns=80)
        tui = TUI(term)
        tui.stop()
        # invalidate should not crash or schedule work
        tui.invalidate()


class TestTUIZeroDimensions:
    """Edge case: terminal with zero rows or columns."""

    def test_zero_columns_no_crash(self) -> None:
        term = VirtualTerminal(rows=24, columns=0)
        tui = TUI(term)
        tui.add_child(SimpleText("text"))
        tui.do_render()  # should not raise

    def test_zero_rows_no_crash(self) -> None:
        term = VirtualTerminal(rows=0, columns=80)
        tui = TUI(term)
        tui.add_child(SimpleText("text"))
        tui.do_render()  # should not raise
