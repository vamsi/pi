"""Fuzzy matching utilities.

Matches if all query characters appear in order (not necessarily consecutive).
Lower score = better match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

_WORD_BOUNDARY_RE = re.compile(r"[\s\-_./:]")
_ALPHA_NUM_RE = re.compile(r"^(?P<letters>[a-z]+)(?P<digits>[0-9]+)$")
_NUM_ALPHA_RE = re.compile(r"^(?P<digits>[0-9]+)(?P<letters>[a-z]+)$")


@dataclass
class FuzzyMatch:
    matches: bool
    score: float


def fuzzy_match(query: str, text: str) -> FuzzyMatch:
    query_lower = query.lower()
    text_lower = text.lower()

    def match_query(normalized_query: str) -> FuzzyMatch:
        if len(normalized_query) == 0:
            return FuzzyMatch(matches=True, score=0)

        if len(normalized_query) > len(text_lower):
            return FuzzyMatch(matches=False, score=0)

        query_index = 0
        score: float = 0
        last_match_index = -1
        consecutive_matches = 0

        for i in range(len(text_lower)):
            if query_index >= len(normalized_query):
                break
            if text_lower[i] == normalized_query[query_index]:
                is_word_boundary = i == 0 or bool(
                    _WORD_BOUNDARY_RE.match(text_lower[i - 1])
                )

                if last_match_index == i - 1:
                    consecutive_matches += 1
                    score -= consecutive_matches * 5
                else:
                    consecutive_matches = 0
                    if last_match_index >= 0:
                        score += (i - last_match_index - 1) * 2

                if is_word_boundary:
                    score -= 10

                score += i * 0.1

                last_match_index = i
                query_index += 1

        if query_index < len(normalized_query):
            return FuzzyMatch(matches=False, score=0)

        return FuzzyMatch(matches=True, score=score)

    primary_match = match_query(query_lower)
    if primary_match.matches:
        return primary_match

    alpha_numeric_match = _ALPHA_NUM_RE.match(query_lower)
    numeric_alpha_match = _NUM_ALPHA_RE.match(query_lower)

    if alpha_numeric_match:
        swapped_query = (
            alpha_numeric_match.group("digits") + alpha_numeric_match.group("letters")
        )
    elif numeric_alpha_match:
        swapped_query = (
            numeric_alpha_match.group("letters") + numeric_alpha_match.group("digits")
        )
    else:
        swapped_query = ""

    if not swapped_query:
        return primary_match

    swapped_match = match_query(swapped_query)
    if not swapped_match.matches:
        return primary_match

    return FuzzyMatch(matches=True, score=swapped_match.score + 5)


def fuzzy_filter(
    items: list[T], query: str, get_text: Callable[[T], str]
) -> list[T]:
    """Filter and sort items by fuzzy match quality (best matches first).

    Supports space-separated tokens: all tokens must match.
    """
    if not query.strip():
        return items

    tokens = [t for t in query.strip().split() if len(t) > 0]

    if not tokens:
        return items

    results: list[tuple[T, float]] = []

    for item in items:
        text = get_text(item)
        total_score: float = 0
        all_match = True

        for token in tokens:
            match = fuzzy_match(token, text)
            if match.matches:
                total_score += match.score
            else:
                all_match = False
                break

        if all_match:
            results.append((item, total_score))

    results.sort(key=lambda r: r[1])
    return [r[0] for r in results]
