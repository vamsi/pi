"""Tests for pi.tui.fuzzy -- fuzzy matching and filtering."""

from __future__ import annotations

from pi.tui.fuzzy import FuzzyMatch, fuzzy_filter, fuzzy_match


# ---------------------------------------------------------------------------
# fuzzy_match: basic matching behaviour
# ---------------------------------------------------------------------------


class TestFuzzyMatchBasic:
    """Core matching semantics for fuzzy_match."""

    def test_empty_query_matches_everything(self) -> None:
        result = fuzzy_match("", "anything at all")
        assert result.matches is True
        assert result.score == 0

    def test_empty_query_matches_empty_text(self) -> None:
        result = fuzzy_match("", "")
        assert result.matches is True
        assert result.score == 0

    def test_exact_match(self) -> None:
        result = fuzzy_match("hello", "hello")
        assert result.matches is True

    def test_exact_match_case_insensitive(self) -> None:
        result = fuzzy_match("Hello", "hello")
        assert result.matches is True

    def test_prefix_match(self) -> None:
        result = fuzzy_match("hel", "hello world")
        assert result.matches is True

    def test_non_consecutive_chars_match(self) -> None:
        """Characters appear in order but not adjacent in the text."""
        result = fuzzy_match("hlo", "hello")
        assert result.matches is True

    def test_no_match_returns_false(self) -> None:
        result = fuzzy_match("xyz", "hello")
        assert result.matches is False

    def test_query_longer_than_text_does_not_match(self) -> None:
        result = fuzzy_match("abcdef", "abc")
        assert result.matches is False

    def test_characters_must_appear_in_order(self) -> None:
        """'ba' should not match 'ab' because order matters."""
        result = fuzzy_match("ba", "ab")
        assert result.matches is False


# ---------------------------------------------------------------------------
# fuzzy_match: scoring
# ---------------------------------------------------------------------------


class TestFuzzyMatchScoring:
    """Score semantics -- lower score = better match."""

    def test_exact_prefix_scores_lower_than_scattered(self) -> None:
        prefix = fuzzy_match("ab", "abcdef")
        scattered = fuzzy_match("af", "abcdef")
        assert prefix.matches is True
        assert scattered.matches is True
        assert prefix.score < scattered.score

    def test_word_boundary_bonus_lowers_score(self) -> None:
        """Matching at a word boundary (after space/dash/etc.) gets a bonus."""
        boundary = fuzzy_match("w", "hello world")
        non_boundary = fuzzy_match("o", "hello world")
        # 'w' is at a word boundary (index 6, preceded by space) and gets -10 bonus.
        # 'o' at index 4 is not at a boundary.
        assert boundary.matches is True
        assert non_boundary.matches is True
        assert boundary.score < non_boundary.score

    def test_consecutive_chars_bonus(self) -> None:
        """Consecutive character matches receive a bonus (score reduction)."""
        consecutive = fuzzy_match("ab", "abxyz")
        non_consecutive = fuzzy_match("az", "abxyz")
        assert consecutive.matches is True
        assert non_consecutive.matches is True
        assert consecutive.score < non_consecutive.score

    def test_match_at_start_scores_better_than_later(self) -> None:
        """Earlier matches get lower positional penalty (i * 0.1)."""
        early = fuzzy_match("a", "abcdef")
        late = fuzzy_match("f", "abcdef")
        assert early.matches is True
        assert late.matches is True
        assert early.score < late.score


# ---------------------------------------------------------------------------
# fuzzy_match: alphanumeric swap
# ---------------------------------------------------------------------------


class TestFuzzyMatchAlphanumericSwap:
    """Alpha-numeric and numeric-alpha query rewriting."""

    def test_alpha_numeric_swap_matches(self) -> None:
        """Query 'abc123' should match text containing '123abc' via swap."""
        result = fuzzy_match("abc123", "123abc")
        assert result.matches is True

    def test_numeric_alpha_swap_matches(self) -> None:
        """Query '123abc' should match text containing 'abc123' via swap."""
        result = fuzzy_match("123abc", "abc123")
        assert result.matches is True

    def test_swap_incurs_penalty(self) -> None:
        """A swapped match should have a higher (worse) score than a direct match."""
        direct = fuzzy_match("abc123", "abc123")
        swapped = fuzzy_match("abc123", "123abc")
        assert direct.matches is True
        assert swapped.matches is True
        assert swapped.score > direct.score

    def test_swap_not_attempted_for_non_alphanum(self) -> None:
        """If query does not match alpha+digits or digits+alpha, no swap."""
        result = fuzzy_match("a1b2", "2b1a")
        # 'a1b2' is not pure letters+digits pattern, so no swap attempted.
        assert result.matches is False


# ---------------------------------------------------------------------------
# fuzzy_filter
# ---------------------------------------------------------------------------


class TestFuzzyFilter:
    """Filtering and sorting a list of items by fuzzy match quality."""

    def test_empty_query_returns_all_items(self) -> None:
        items = ["alpha", "beta", "gamma"]
        result = fuzzy_filter(items, "", get_text=str)
        assert result == items

    def test_whitespace_query_returns_all_items(self) -> None:
        items = ["alpha", "beta", "gamma"]
        result = fuzzy_filter(items, "   ", get_text=str)
        assert result == items

    def test_filters_non_matching_items(self) -> None:
        items = ["apple", "banana", "cherry"]
        result = fuzzy_filter(items, "app", get_text=str)
        assert "apple" in result
        assert "banana" not in result
        assert "cherry" not in result

    def test_sorts_by_score_best_first(self) -> None:
        items = ["xyzabc", "abcxyz", "aabbcc"]
        result = fuzzy_filter(items, "abc", get_text=str)
        # "abcxyz" has prefix match (best score), "aabbcc" has scattered match,
        # "xyzabc" matches starting later. Best match should come first.
        assert result[0] == "abcxyz"

    def test_space_separated_tokens_all_must_match(self) -> None:
        items = ["foo bar baz", "foo qux", "bar baz"]
        result = fuzzy_filter(items, "foo baz", get_text=str)
        # Both "foo" and "baz" must appear.
        assert "foo bar baz" in result
        assert "foo qux" not in result
        assert "bar baz" not in result

    def test_custom_get_text(self) -> None:
        """get_text extracts the string to match against from each item."""
        items = [{"name": "alice"}, {"name": "bob"}, {"name": "alicia"}]
        result = fuzzy_filter(items, "ali", get_text=lambda d: d["name"])
        names = [d["name"] for d in result]
        assert "alice" in names
        assert "alicia" in names
        assert "bob" not in names

    def test_returns_empty_list_when_nothing_matches(self) -> None:
        items = ["alpha", "beta", "gamma"]
        result = fuzzy_filter(items, "zzz", get_text=str)
        assert result == []
