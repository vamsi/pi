"""Autocomplete provider for slash commands, file paths, and fuzzy file search.

Handles @ file references, slash command completion, and path completion
using optional fd integration for fast, .gitignore-respecting file search.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from pi.tui.fuzzy import fuzzy_filter

PATH_DELIMITERS = frozenset({" ", "\t", '"', "'", "="})


def find_last_delimiter(text: str) -> int:
    """Find the last delimiter character in text, returning its index or -1."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] in PATH_DELIMITERS:
            return i
    return -1


def find_unclosed_quote_start(text: str) -> int | None:
    """Find the start position of an unclosed double-quote, or None."""
    in_quotes = False
    quote_start = -1

    for i in range(len(text)):
        if text[i] == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = i

    return quote_start if in_quotes else None


def is_token_start(text: str, index: int) -> bool:
    """Check if the given index is at the start of a token."""
    return index == 0 or text[index - 1] in PATH_DELIMITERS


def extract_quoted_prefix(text: str) -> str | None:
    """Extract the prefix starting from an unclosed quote, or None."""
    quote_start = find_unclosed_quote_start(text)
    if quote_start is None:
        return None

    if quote_start > 0 and text[quote_start - 1] == "@":
        if not is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1 :]

    if not is_token_start(text, quote_start):
        return None

    return text[quote_start:]


@dataclass
class ParsedPrefix:
    raw_prefix: str
    is_at_prefix: bool
    is_quoted_prefix: bool


def parse_path_prefix(prefix: str) -> ParsedPrefix:
    """Parse a prefix string to extract the raw path, and @ / quote flags."""
    if prefix.startswith('@"'):
        return ParsedPrefix(
            raw_prefix=prefix[2:], is_at_prefix=True, is_quoted_prefix=True
        )
    if prefix.startswith('"'):
        return ParsedPrefix(
            raw_prefix=prefix[1:], is_at_prefix=False, is_quoted_prefix=True
        )
    if prefix.startswith("@"):
        return ParsedPrefix(
            raw_prefix=prefix[1:], is_at_prefix=True, is_quoted_prefix=False
        )
    return ParsedPrefix(
        raw_prefix=prefix, is_at_prefix=False, is_quoted_prefix=False
    )


def build_completion_value(
    path: str,
    *,
    is_directory: bool,
    is_at_prefix: bool,
    is_quoted_prefix: bool,
) -> str:
    """Build the final completion value, adding quotes/@ as needed."""
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""

    if not needs_quotes:
        return f"{prefix}{path}"

    open_quote = f'{prefix}"'
    close_quote = '"'
    return f"{open_quote}{path}{close_quote}"


def walk_directory_with_fd(
    base_dir: str,
    fd_path: str,
    query: str,
    max_results: int,
) -> list[FdEntry]:
    """Use fd to walk a directory tree (fast, respects .gitignore).

    Returns a list of FdEntry with path and is_directory fields.
    """
    args = [
        fd_path,
        "--base-directory",
        base_dir,
        "--max-results",
        str(max_results),
        "--type",
        "f",
        "--type",
        "d",
        "--full-path",
        "--hidden",
        "--exclude",
        ".git",
        "--exclude",
        ".git/*",
        "--exclude",
        ".git/**",
    ]

    # Add query as pattern if provided
    if query:
        args.append(query)

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout:
        return []

    lines = [line for line in result.stdout.strip().split("\n") if line]
    results: list[FdEntry] = []

    for line in lines:
        normalized_path = line.rstrip("/")
        if (
            normalized_path == ".git"
            or normalized_path.startswith(".git/")
            or "/.git/" in normalized_path
        ):
            continue

        # fd outputs directories with trailing /
        is_directory = line.endswith("/")
        results.append(FdEntry(path=line, is_directory=is_directory))

    return results


@dataclass
class FdEntry:
    """An entry returned by fd: a file or directory path."""

    path: str
    is_directory: bool


@dataclass
class AutocompleteItem:
    """A single autocomplete suggestion."""

    value: str
    label: str
    description: str | None = None


@dataclass
class SlashCommand:
    """A registered slash command with optional argument completion."""

    name: str
    description: str | None = None
    get_argument_completions: Callable[[str], list[AutocompleteItem] | None] | None = (
        None
    )


@dataclass
class SuggestionResult:
    """The result of getSuggestions: matching items and the prefix they match."""

    items: list[AutocompleteItem]
    prefix: str


@dataclass
class CompletionResult:
    """The result of applyCompletion: updated lines and cursor position."""

    lines: list[str]
    cursor_line: int
    cursor_col: int


class AutocompleteProvider(Protocol):
    """Protocol for autocomplete providers."""

    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None:
        """Get autocomplete suggestions for current text/cursor position.

        Returns None if no suggestions are available.
        """
        ...

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> CompletionResult:
        """Apply the selected autocomplete item.

        Returns the new text and cursor position.
        """
        ...


class CombinedAutocompleteProvider:
    """Combined provider that handles slash commands, file paths, and @ references."""

    def __init__(
        self,
        commands: list[SlashCommand | AutocompleteItem] | None = None,
        base_path: str | None = None,
        fd_path: str | None = None,
    ) -> None:
        self._commands: list[SlashCommand | AutocompleteItem] = (
            commands if commands is not None else []
        )
        self._base_path: str = base_path if base_path is not None else os.getcwd()
        self._fd_path: str | None = fd_path

    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None:
        """Get autocomplete suggestions for the current text and cursor position."""
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        # Check for @ file reference (fuzzy search) - must be after a delimiter or at start
        at_prefix = self._extract_at_prefix(text_before_cursor)
        if at_prefix is not None:
            parsed = parse_path_prefix(at_prefix)
            suggestions = self._get_fuzzy_file_suggestions(
                parsed.raw_prefix,
                is_quoted_prefix=parsed.is_quoted_prefix,
            )
            if len(suggestions) == 0:
                return None

            return SuggestionResult(items=suggestions, prefix=at_prefix)

        # Check for slash commands
        if text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")

            if space_index == -1:
                # No space yet - complete command names with fuzzy matching
                prefix = text_before_cursor[1:]  # Remove the "/"

                @dataclass
                class _CmdItem:
                    name: str
                    label: str
                    description: str | None

                command_items: list[_CmdItem] = []
                for cmd in self._commands:
                    if isinstance(cmd, SlashCommand):
                        command_items.append(
                            _CmdItem(
                                name=cmd.name,
                                label=cmd.name,
                                description=cmd.description,
                            )
                        )
                    else:
                        command_items.append(
                            _CmdItem(
                                name=cmd.value,
                                label=cmd.label,
                                description=cmd.description,
                            )
                        )

                filtered_cmds = fuzzy_filter(
                    command_items, prefix, lambda item: item.name
                )
                filtered: list[AutocompleteItem] = []
                for item in filtered_cmds:
                    ac_item = AutocompleteItem(value=item.name, label=item.label)
                    if item.description:
                        ac_item.description = item.description
                    filtered.append(ac_item)

                if len(filtered) == 0:
                    return None

                return SuggestionResult(
                    items=filtered,
                    prefix=text_before_cursor,
                )
            else:
                # Space found - complete command arguments
                command_name = text_before_cursor[1:space_index]  # Command without "/"
                argument_text = text_before_cursor[
                    space_index + 1 :
                ]  # Text after space

                command: SlashCommand | AutocompleteItem | None = None
                for cmd in self._commands:
                    if isinstance(cmd, SlashCommand):
                        name = cmd.name
                    else:
                        name = cmd.value
                    if name == command_name:
                        command = cmd
                        break

                if command is None:
                    return None
                if not isinstance(command, SlashCommand):
                    return None
                if command.get_argument_completions is None:
                    return None

                argument_suggestions = command.get_argument_completions(argument_text)
                if not argument_suggestions or len(argument_suggestions) == 0:
                    return None

                return SuggestionResult(
                    items=argument_suggestions,
                    prefix=argument_text,
                )

        # Check for file paths - triggered by Tab or if we detect a path pattern
        path_match = self._extract_path_prefix(text_before_cursor, force_extract=False)

        if path_match is not None:
            suggestions = self._get_file_suggestions(path_match)
            if len(suggestions) == 0:
                return None

            # Check if we have an exact match that is a directory
            # In that case, we might want to return suggestions for the directory content instead
            # But only if the prefix ends with /
            if (
                len(suggestions) == 1
                and suggestions[0].value == path_match
                and not path_match.endswith("/")
            ):
                # Exact match found (e.g. user typed "src" and "src/" is the only match)
                # We still return it so user can select it and add /
                return SuggestionResult(items=suggestions, prefix=path_match)

            return SuggestionResult(items=suggestions, prefix=path_match)

        return None

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> CompletionResult:
        """Apply the selected autocomplete item, returning updated lines and cursor."""
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        before_prefix = current_line[: cursor_col - len(prefix)]
        after_cursor = current_line[cursor_col:]
        is_quoted_prefix = prefix.startswith('"') or prefix.startswith('@"')
        has_leading_quote_after_cursor = after_cursor.startswith('"')
        has_trailing_quote_in_item = item.value.endswith('"')
        adjusted_after_cursor = (
            after_cursor[1:]
            if (
                is_quoted_prefix
                and has_trailing_quote_in_item
                and has_leading_quote_after_cursor
            )
            else after_cursor
        )

        # Check if we're completing a slash command (prefix starts with "/" but NOT a file path)
        # Slash commands are at the start of the line and don't contain path separators after the first /
        is_slash_command = (
            prefix.startswith("/")
            and before_prefix.strip() == ""
            and "/" not in prefix[1:]
        )
        if is_slash_command:
            # This is a command name completion
            new_line = f"{before_prefix}/{item.value} {adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            return CompletionResult(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + len(item.value) + 2,  # +2 for "/" and space
            )

        # Check if we're completing a file attachment (prefix starts with "@")
        if prefix.startswith("@"):
            # This is a file attachment completion
            # Don't add space after directories so user can continue autocompleting
            is_directory = item.label.endswith("/")
            suffix = "" if is_directory else " "
            new_line = f"{before_prefix}{item.value}{suffix}{adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            has_trailing_quote = item.value.endswith('"')
            cursor_offset = (
                len(item.value) - 1
                if is_directory and has_trailing_quote
                else len(item.value)
            )

            return CompletionResult(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + cursor_offset + len(suffix),
            )

        # Check if we're in a slash command context (beforePrefix contains "/command ")
        text_before_cursor = current_line[:cursor_col]
        if "/" in text_before_cursor and " " in text_before_cursor:
            # This is likely a command argument completion
            new_line = before_prefix + item.value + adjusted_after_cursor
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            is_directory = item.label.endswith("/")
            has_trailing_quote = item.value.endswith('"')
            cursor_offset = (
                len(item.value) - 1
                if is_directory and has_trailing_quote
                else len(item.value)
            )

            return CompletionResult(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + cursor_offset,
            )

        # For file paths, complete the path
        new_line = before_prefix + item.value + adjusted_after_cursor
        new_lines = list(lines)
        new_lines[cursor_line] = new_line

        is_directory = item.label.endswith("/")
        has_trailing_quote = item.value.endswith('"')
        cursor_offset = (
            len(item.value) - 1
            if is_directory and has_trailing_quote
            else len(item.value)
        )

        return CompletionResult(
            lines=new_lines,
            cursor_line=cursor_line,
            cursor_col=len(before_prefix) + cursor_offset,
        )

    # ---- Private helpers ----

    def _extract_at_prefix(self, text: str) -> str | None:
        """Extract @ prefix for fuzzy file suggestions."""
        quoted_prefix = extract_quoted_prefix(text)
        if quoted_prefix is not None and quoted_prefix.startswith('@"'):
            return quoted_prefix

        last_delimiter_index = find_last_delimiter(text)
        token_start = 0 if last_delimiter_index == -1 else last_delimiter_index + 1

        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]

        return None

    def _extract_path_prefix(
        self, text: str, *, force_extract: bool = False
    ) -> str | None:
        """Extract a path-like prefix from the text before cursor."""
        quoted_prefix = extract_quoted_prefix(text)
        if quoted_prefix is not None:
            return quoted_prefix

        last_delimiter_index = find_last_delimiter(text)
        path_prefix = (
            text if last_delimiter_index == -1 else text[last_delimiter_index + 1 :]
        )

        # For forced extraction (Tab key), always return something
        if force_extract:
            return path_prefix

        # For natural triggers, return if it looks like a path, ends with /, starts with ~/, .
        # Only return empty string if the text looks like it's starting a path context
        if (
            "/" in path_prefix
            or path_prefix.startswith(".")
            or path_prefix.startswith("~/")
        ):
            return path_prefix

        # Return empty string only after a space (not for completely empty text)
        # Empty text should not trigger file suggestions - that's for forced Tab completion
        if path_prefix == "" and text.endswith(" "):
            return path_prefix

        return None

    def _expand_home_path(self, path: str) -> str:
        """Expand home directory (~/) to actual home path."""
        if path.startswith("~/"):
            expanded_path = os.path.join(str(Path.home()), path[2:])
            # Preserve trailing slash if original path had one
            if path.endswith("/") and not expanded_path.endswith("/"):
                return f"{expanded_path}/"
            return expanded_path
        elif path == "~":
            return str(Path.home())
        return path

    def _get_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        """Get file/directory suggestions for a given path prefix."""
        try:
            parsed = parse_path_prefix(prefix)
            raw_prefix = parsed.raw_prefix
            is_at_prefix = parsed.is_at_prefix
            is_quoted_prefix = parsed.is_quoted_prefix
            expanded_prefix = raw_prefix

            # Handle home directory expansion
            if expanded_prefix.startswith("~"):
                expanded_prefix = self._expand_home_path(expanded_prefix)

            is_root_prefix = raw_prefix in (
                "",
                "./",
                "../",
                "~",
                "~/",
                "/",
            ) or (is_at_prefix and raw_prefix == "")

            if is_root_prefix:
                # Complete from specified position
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = expanded_prefix
                else:
                    search_dir = os.path.join(self._base_path, expanded_prefix)
                search_prefix = ""
            elif raw_prefix.endswith("/"):
                # If prefix ends with /, show contents of that directory
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = expanded_prefix
                else:
                    search_dir = os.path.join(self._base_path, expanded_prefix)
                search_prefix = ""
            else:
                # Split into directory and file prefix
                dir_part = os.path.dirname(expanded_prefix)
                file_part = os.path.basename(expanded_prefix)
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = dir_part
                else:
                    search_dir = os.path.join(self._base_path, dir_part)
                search_prefix = file_part

            entries = list(os.scandir(search_dir))
            suggestions: list[AutocompleteItem] = []

            for entry in entries:
                if not entry.name.lower().startswith(search_prefix.lower()):
                    continue

                # Check if entry is a directory (or a symlink pointing to a directory)
                is_directory = entry.is_dir(follow_symlinks=False)
                if not is_directory and entry.is_symlink():
                    try:
                        full_path = os.path.join(search_dir, entry.name)
                        is_directory = os.path.isdir(full_path)
                    except OSError:
                        # Broken symlink or permission error - treat as file
                        pass

                name = entry.name
                display_prefix = raw_prefix

                if display_prefix.endswith("/"):
                    # If prefix ends with /, append entry to the prefix
                    relative_path = display_prefix + name
                elif "/" in display_prefix:
                    # Preserve ~/ format for home directory paths
                    if display_prefix.startswith("~/"):
                        home_relative_dir = display_prefix[2:]  # Remove ~/
                        dir_name = os.path.dirname(home_relative_dir)
                        relative_path = (
                            f"~/{name}"
                            if dir_name == "."
                            else f"~/{os.path.join(dir_name, name)}"
                        )
                    elif display_prefix.startswith("/"):
                        # Absolute path - construct properly
                        dir_name = os.path.dirname(display_prefix)
                        if dir_name == "/":
                            relative_path = f"/{name}"
                        else:
                            relative_path = f"{dir_name}/{name}"
                    else:
                        relative_path = os.path.join(
                            os.path.dirname(display_prefix), name
                        )
                else:
                    # For standalone entries, preserve ~/ if original prefix was ~/
                    if display_prefix.startswith("~"):
                        relative_path = f"~/{name}"
                    else:
                        relative_path = name

                path_value = f"{relative_path}/" if is_directory else relative_path
                value = build_completion_value(
                    path_value,
                    is_directory=is_directory,
                    is_at_prefix=is_at_prefix,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append(
                    AutocompleteItem(
                        value=value,
                        label=name + ("/" if is_directory else ""),
                    )
                )

            # Sort directories first, then alphabetically
            def _sort_key(item: AutocompleteItem) -> tuple[int, str]:
                is_dir = item.value.endswith("/")
                return (0 if is_dir else 1, item.label)

            suggestions.sort(key=_sort_key)

            return suggestions
        except OSError:
            # Directory doesn't exist or not accessible
            return []

    def _score_entry(
        self, file_path: str, query: str, is_directory: bool
    ) -> int:
        """Score an entry against the query (higher = better match).

        is_directory adds bonus to prioritize folders.
        """
        file_name = os.path.basename(file_path)
        lower_file_name = file_name.lower()
        lower_query = query.lower()

        score = 0

        # Exact filename match (highest)
        if lower_file_name == lower_query:
            score = 100
        # Filename starts with query
        elif lower_file_name.startswith(lower_query):
            score = 80
        # Substring match in filename
        elif lower_query in lower_file_name:
            score = 50
        # Substring match in full path
        elif lower_query in file_path.lower():
            score = 30

        # Directories get a bonus to appear first
        if is_directory and score > 0:
            score += 10

        return score

    def _get_fuzzy_file_suggestions(
        self,
        query: str,
        *,
        is_quoted_prefix: bool,
    ) -> list[AutocompleteItem]:
        """Fuzzy file search using fd (fast, respects .gitignore)."""
        if self._fd_path is None:
            # fd not available, return empty results
            return []

        try:
            entries = walk_directory_with_fd(
                self._base_path, self._fd_path, query, 100
            )

            # Score entries
            @dataclass
            class _ScoredEntry:
                path: str
                is_directory: bool
                score: int

            scored_entries: list[_ScoredEntry] = []
            for entry in entries:
                score = (
                    self._score_entry(entry.path, query, entry.is_directory)
                    if query
                    else 1
                )
                if score > 0:
                    scored_entries.append(
                        _ScoredEntry(
                            path=entry.path,
                            is_directory=entry.is_directory,
                            score=score,
                        )
                    )

            # Sort by score (descending) and take top 20
            scored_entries.sort(key=lambda e: e.score, reverse=True)
            top_entries = scored_entries[:20]

            # Build suggestions
            suggestions: list[AutocompleteItem] = []
            for se in top_entries:
                entry_path = se.path
                is_directory = se.is_directory

                # fd already includes trailing / for directories
                path_without_slash = (
                    entry_path[:-1] if is_directory else entry_path
                )
                entry_name = os.path.basename(path_without_slash)
                value = build_completion_value(
                    entry_path,
                    is_directory=is_directory,
                    is_at_prefix=True,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append(
                    AutocompleteItem(
                        value=value,
                        label=entry_name + ("/" if is_directory else ""),
                        description=path_without_slash,
                    )
                )

            return suggestions
        except Exception:
            return []

    def get_force_file_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> SuggestionResult | None:
        """Force file completion (called on Tab key) - always returns suggestions."""
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        # Don't trigger if we're typing a slash command at the start of the line
        stripped = text_before_cursor.strip()
        if stripped.startswith("/") and " " not in stripped:
            return None

        # Force extract path prefix - this will always return something
        path_match = self._extract_path_prefix(
            text_before_cursor, force_extract=True
        )
        if path_match is not None:
            suggestions = self._get_file_suggestions(path_match)
            if len(suggestions) == 0:
                return None

            return SuggestionResult(items=suggestions, prefix=path_match)

        return None

    def should_trigger_file_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
    ) -> bool:
        """Check if we should trigger file completion (called on Tab key)."""
        current_line = lines[cursor_line] if cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        # Don't trigger if we're typing a slash command at the start of the line
        stripped = text_before_cursor.strip()
        if stripped.startswith("/") and " " not in stripped:
            return False

        return True
