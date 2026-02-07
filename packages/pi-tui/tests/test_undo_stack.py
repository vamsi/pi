"""Tests for pi.tui.undo_stack.UndoStack -- generic undo with deep-copy semantics."""

from __future__ import annotations

from pi.tui.undo_stack import UndoStack


class TestUndoStackPushAndPop:
    """Basic push and pop operations."""

    def test_push_then_pop_returns_value(self) -> None:
        stack: UndoStack[str] = UndoStack()
        stack.push("state1")
        assert stack.pop() == "state1"

    def test_pop_returns_in_lifo_order(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1

    def test_length_tracks_entries(self) -> None:
        stack: UndoStack[str] = UndoStack()
        assert stack.length == 0
        stack.push("a")
        assert stack.length == 1
        stack.push("b")
        assert stack.length == 2
        stack.pop()
        assert stack.length == 1


class TestUndoStackEmpty:
    """Behaviour on an empty stack."""

    def test_pop_empty_returns_none(self) -> None:
        stack: UndoStack[str] = UndoStack()
        assert stack.pop() is None

    def test_pop_after_exhaustion_returns_none(self) -> None:
        stack: UndoStack[int] = UndoStack()
        stack.push(42)
        stack.pop()
        assert stack.pop() is None


class TestUndoStackDeepCopy:
    """Push stores a deep clone so mutations to the original do not affect the stack."""

    def test_mutating_list_after_push_does_not_affect_stored(self) -> None:
        stack: UndoStack[list[int]] = UndoStack()
        data = [1, 2, 3]
        stack.push(data)

        data.append(4)  # mutate the original

        popped = stack.pop()
        assert popped == [1, 2, 3]

    def test_mutating_dict_after_push_does_not_affect_stored(self) -> None:
        stack: UndoStack[dict[str, int]] = UndoStack()
        data = {"a": 1, "b": 2}
        stack.push(data)

        data["c"] = 3  # mutate the original

        popped = stack.pop()
        assert popped == {"a": 1, "b": 2}

    def test_nested_structure_is_deeply_copied(self) -> None:
        stack: UndoStack[dict[str, list[int]]] = UndoStack()
        data = {"items": [1, 2, 3]}
        stack.push(data)

        data["items"].append(4)  # mutate nested list

        popped = stack.pop()
        assert popped is not None
        assert popped["items"] == [1, 2, 3]


class TestUndoStackClear:
    """The clear method empties the stack."""

    def test_clear_removes_all_entries(self) -> None:
        stack: UndoStack[str] = UndoStack()
        stack.push("a")
        stack.push("b")
        stack.push("c")
        assert stack.length == 3

        stack.clear()
        assert stack.length == 0
        assert stack.pop() is None

    def test_clear_on_empty_stack_is_safe(self) -> None:
        stack: UndoStack[str] = UndoStack()
        stack.clear()  # should not raise
        assert stack.length == 0
