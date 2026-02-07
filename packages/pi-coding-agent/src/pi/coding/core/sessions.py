"""JSONL session manager with tree branching and lazy persistence.

Sessions are stored as append-only JSONL files where each line is a typed entry.
Entries form a tree structure via parent_id references, enabling branching
conversations with full history preservation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

# --- Constants ---

CURRENT_SESSION_VERSION = 3


# --- Entry types ---


class SessionHeader(BaseModel):
    """First line of a JSONL session file."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session"] = "session"
    version: int = CURRENT_SESSION_VERSION
    id: str
    timestamp: str
    cwd: str
    parent_session: str | None = Field(default=None, alias="parentSession")


class SessionEntryBase(BaseModel):
    """Common fields for all session entries."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    parent_id: str | None = Field(default=None, alias="parentId")
    timestamp: str = ""


class SessionMessageEntry(SessionEntryBase):
    """Wraps an agent message in the session."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["message"] = "message"
    message: dict[str, Any]  # Serialized AgentMessage


class ThinkingLevelChangeEntry(SessionEntryBase):
    """Records a change in thinking level."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking_level_change"] = Field(default="thinking_level_change", alias="thinkingLevelChange")
    thinking_level: str = Field(alias="thinkingLevel")


class ModelChangeEntry(SessionEntryBase):
    """Records a model/provider switch."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["model_change"] = Field(default="model_change", alias="modelChange")
    model_id: str = Field(alias="modelId")
    provider: str


class CompactionEntry(SessionEntryBase):
    """Compressed context summary, replacing discarded history."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str | None = Field(default=None, alias="firstKeptEntryId")
    tokens_before: int = Field(default=0, alias="tokensBefore")
    details: Any = None
    from_hook: bool = Field(default=False, alias="fromHook")


class BranchSummaryEntry(SessionEntryBase):
    """Summary of an abandoned branch path."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["branch_summary"] = Field(default="branch_summary", alias="branchSummary")
    summary: str
    branch_entry_ids: list[str] = Field(default_factory=list, alias="branchEntryIds")
    details: Any = None
    from_hook: bool = Field(default=False, alias="fromHook")


class CustomEntry(SessionEntryBase):
    """Extension-specific data. NOT included in LLM context."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["custom"] = "custom"
    role: str = ""
    data: Any = None


class LabelEntry(SessionEntryBase):
    """User-defined bookmark on an entry."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["label"] = "label"
    label: str
    target_id: str = Field(alias="targetId")


class SessionInfoEntry(SessionEntryBase):
    """Metadata like display name for the session."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["session_info"] = Field(default="session_info", alias="sessionInfo")
    name: str = ""


class CustomMessageEntry(SessionEntryBase):
    """Extension messages that participate in LLM context."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["custom_message"] = Field(default="custom_message", alias="customMessage")
    role: str
    content: list[dict[str, Any]] = Field(default_factory=list)


# Union of all entry types
SessionEntry = (
    SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | LabelEntry
    | SessionInfoEntry
    | CustomMessageEntry
)

FileEntry = SessionHeader | SessionEntry


# --- Tree node ---


@dataclass
class SessionTreeNode:
    """Tree structure for visualizing session history."""

    entry: SessionEntry
    children: list[SessionTreeNode] = field(default_factory=list)
    label: str | None = None


# --- Session context (resolved output) ---


@dataclass
class SessionContext:
    """Resolved output from walking the session tree."""

    messages: list[dict[str, Any]]
    thinking_level: str | None = None
    model_id: str | None = None
    provider: str | None = None


# --- Session info for listings ---


@dataclass
class SessionInfo:
    """Session metadata for display in session listings."""

    path: str
    id: str
    cwd: str
    name: str | None = None
    parent_session: str | None = None
    created: float = 0.0
    modified: float = 0.0
    message_count: int = 0
    first_user_text: str = ""
    all_text_preview: str = ""


# --- ID generation ---


def _generate_id(existing: set[str]) -> str:
    """Generate a short collision-checked ID."""
    for _ in range(100):
        candidate = uuid4().hex[:8]
        if candidate not in existing:
            return candidate
    return uuid4().hex


def _timestamp_now() -> str:
    """ISO timestamp string for entry timestamps."""
    return datetime.now(UTC).isoformat()


def _ms_now() -> int:
    """Unix timestamp in milliseconds."""
    return int(time.time() * 1000)


# --- Migrations ---


def _migrate_v1_to_v2(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add id/parentId tree structure (v1 was linear)."""
    prev_id: str | None = None
    existing_ids: set[str] = set()

    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue

        entry_id = _generate_id(existing_ids)
        existing_ids.add(entry_id)
        entry["id"] = entry_id
        entry["parentId"] = prev_id

        # Convert index-based compaction to id-based
        if entry.get("type") == "compaction" and "firstKeptEntryIndex" in entry:
            idx = entry.pop("firstKeptEntryIndex")
            # Find the entry at that index (skip header)
            non_header = [e for e in entries if e.get("type") != "session"]
            if 0 <= idx < len(non_header):
                target = non_header[idx]
                entry["firstKeptEntryId"] = target.get("id")

        prev_id = entry_id

    return entries


def _migrate_v2_to_v3(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename hookMessage role to custom."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "hookMessage":
            entry["type"] = "custom"
    return entries


def _migrate_to_current(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Apply all necessary migrations. Returns (entries, was_migrated)."""
    if not entries:
        return entries, False

    header = entries[0]
    version = header.get("version", 1)
    migrated = False

    if version < 2:
        entries = _migrate_v1_to_v2(entries)
        migrated = True
    if version < 3:
        entries = _migrate_v2_to_v3(entries)
        migrated = True

    return entries, migrated


# --- Parsing ---


def parse_session_entries(content: str) -> list[dict[str, Any]]:
    """Parse JSONL content into a list of entry dicts."""
    entries: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Skip malformed lines
    return entries


def load_entries_from_file(path: str) -> list[dict[str, Any]]:
    """Load and validate entries from a JSONL session file."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []

    entries = parse_session_entries(content)
    if not entries:
        return []

    # Validate header
    header = entries[0]
    if header.get("type") != "session":
        return []

    return entries


def is_valid_session_file(path: str) -> bool:
    """Check if a file appears to be a valid session file (peeks at first line)."""
    try:
        with Path(path).open(encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            header = json.loads(first_line)
            return header.get("type") == "session"
    except OSError:
        return False
    except json.JSONDecodeError:
        return False


def find_most_recent_session(session_dir: str) -> str | None:
    """Find the most recently modified valid session file in a directory."""
    try:
        files = sorted(
            (
                os.path.join(session_dir, f)
                for f in os.listdir(session_dir)
                if f.endswith(".jsonl") and is_valid_session_file(os.path.join(session_dir, f))
            ),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        return files[0] if files else None
    except OSError:
        return None


# --- SessionManager ---


class SessionManager:
    """Manages a JSONL session with tree-structured entries.

    Supports branching, compaction, lazy persistence, and migrations.
    Use factory methods (create, open, continue_recent, in_memory) instead
    of calling the constructor directly.
    """

    def __init__(
        self,
        *,
        session_id: str,
        session_dir: str,
        cwd: str,
        persist: bool = True,
        session_file: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._session_dir = session_dir
        self._cwd = cwd
        self._persist = persist
        self._session_file = session_file
        self._flushed = False
        self._file_entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._labels_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None

    # --- Properties ---

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_file(self) -> str | None:
        return self._session_file

    @property
    def session_dir(self) -> str:
        return self._session_dir

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    @property
    def entries(self) -> list[dict[str, Any]]:
        """All entries excluding the header."""
        return [e for e in self._file_entries if e.get("type") != "session"]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    # --- Factory methods ---

    @classmethod
    def create(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        """Create a new session with file persistence."""
        sdir = session_dir or cls._default_session_dir(cwd)
        session_id = uuid4().hex
        mgr = cls(session_id=session_id, session_dir=sdir, cwd=cwd, persist=True)
        mgr._new_session()
        return mgr

    @classmethod
    def open(cls, path: str, session_dir: str | None = None) -> SessionManager:
        """Open an existing session file."""
        entries = load_entries_from_file(path)
        if not entries:
            msg = f"Invalid or empty session file: {path}"
            raise ValueError(msg)

        # Apply migrations
        entries, migrated = _migrate_to_current(entries)

        header = entries[0]
        sdir = session_dir or str(Path(path).parent)
        session_id = header.get("id", uuid4().hex)

        mgr = cls(
            session_id=session_id,
            session_dir=sdir,
            cwd=header.get("cwd", ""),
            persist=True,
            session_file=path,
        )
        mgr._load_entries(entries)
        mgr._flushed = True  # Already on disk

        if migrated:
            mgr._rewrite_file()

        return mgr

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: str | None = None) -> SessionManager:
        """Resume the most recent session or create a new one."""
        sdir = session_dir or cls._default_session_dir(cwd)
        recent = find_most_recent_session(sdir)
        if recent:
            return cls.open(recent, sdir)
        return cls.create(cwd, sdir)

    @classmethod
    def in_memory(cls, cwd: str = "") -> SessionManager:
        """Create an in-memory session (no file persistence)."""
        session_id = uuid4().hex
        mgr = cls(session_id=session_id, session_dir="", cwd=cwd or os.getcwd(), persist=False)
        mgr._init_header()
        return mgr

    @classmethod
    def fork_from(cls, source_path: str, target_cwd: str, session_dir: str | None = None) -> SessionManager:
        """Copy a session to a different working directory."""
        source = cls.open(source_path)
        sdir = session_dir or cls._default_session_dir(target_cwd)
        new_mgr = cls.create(target_cwd, sdir)

        # Copy all entries from source
        for entry in source.entries:
            new_mgr._append_raw(dict(entry))

        return new_mgr

    # --- Session initialization ---

    def _new_session(self) -> str | None:
        """Initialize a new session with header, optionally create file."""
        self._init_header()

        if not self._persist:
            return None

        os.makedirs(self._session_dir, exist_ok=True)
        ts = _timestamp_now().replace(":", "-")
        filename = f"{ts}_{self._session_id[:16]}.jsonl"
        self._session_file = os.path.join(self._session_dir, filename)
        return self._session_file

    def _init_header(self) -> None:
        """Create the session header entry."""
        header: dict[str, Any] = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self._session_id,
            "timestamp": _timestamp_now(),
            "cwd": self._cwd,
        }
        self._file_entries = [header]

    def _load_entries(self, entries: list[dict[str, Any]]) -> None:
        """Load entries from parsed JSONL data."""
        self._file_entries = list(entries)
        self._by_id.clear()
        self._labels_by_id.clear()

        for entry in entries:
            if entry.get("type") == "session":
                continue
            entry_id = entry.get("id")
            if entry_id:
                self._by_id[entry_id] = entry
            if entry.get("type") == "label":
                self._labels_by_id[entry.get("targetId", "")] = entry.get("label", "")

        # Set leaf to the last entry
        non_header = self.entries
        self._leaf_id = non_header[-1]["id"] if non_header else None

    # --- Entry append ---

    def _append_raw(self, entry: dict[str, Any]) -> str:
        """Append a raw entry dict, assigning id and parentId."""
        existing_ids = set(self._by_id.keys())
        entry_id = entry.get("id") or _generate_id(existing_ids)
        entry["id"] = entry_id
        entry["parentId"] = self._leaf_id
        entry["timestamp"] = entry.get("timestamp") or _timestamp_now()

        self._file_entries.append(entry)
        self._by_id[entry_id] = entry
        self._leaf_id = entry_id

        self._persist_entry(entry)
        return entry_id

    def append_message(self, message: dict[str, Any]) -> str:
        """Append a message entry."""
        return self._append_raw({"type": "message", "message": message})

    def append_thinking_level_change(self, thinking_level: str) -> str:
        """Record a thinking level change."""
        return self._append_raw(
            {
                "type": "thinking_level_change",
                "thinkingLevel": thinking_level,
            }
        )

    def append_model_change(self, model_id: str, provider: str) -> str:
        """Record a model/provider switch."""
        return self._append_raw(
            {
                "type": "model_change",
                "modelId": model_id,
                "provider": provider,
            }
        )

    def append_compaction(
        self,
        summary: str,
        *,
        first_kept_entry_id: str | None = None,
        tokens_before: int = 0,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Append a compaction summary entry."""
        entry: dict[str, Any] = {
            "type": "compaction",
            "summary": summary,
        }
        if first_kept_entry_id is not None:
            entry["firstKeptEntryId"] = first_kept_entry_id
        if tokens_before:
            entry["tokensBefore"] = tokens_before
        if details is not None:
            entry["details"] = details
        if from_hook:
            entry["fromHook"] = True
        return self._append_raw(entry)

    def append_branch_summary(
        self,
        summary: str,
        branch_entry_ids: list[str] | None = None,
        *,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Append a branch summary entry."""
        entry: dict[str, Any] = {
            "type": "branch_summary",
            "summary": summary,
            "branchEntryIds": branch_entry_ids or [],
        }
        if details is not None:
            entry["details"] = details
        if from_hook:
            entry["fromHook"] = True
        return self._append_raw(entry)

    def append_custom_entry(self, role: str, data: Any = None) -> str:
        """Append an extension-specific entry (not in LLM context)."""
        return self._append_raw({"type": "custom", "role": role, "data": data})

    def append_custom_message(self, role: str, content: list[dict[str, Any]]) -> str:
        """Append an extension message that participates in LLM context."""
        return self._append_raw(
            {
                "type": "custom_message",
                "role": role,
                "content": content,
            }
        )

    def append_label(self, label: str, target_id: str) -> str:
        """Bookmark an entry with a label."""
        self._labels_by_id[target_id] = label
        return self._append_raw(
            {
                "type": "label",
                "label": label,
                "targetId": target_id,
            }
        )

    def set_session_name(self, name: str) -> str:
        """Set or update the session display name."""
        return self._append_raw(
            {
                "type": "session_info",
                "name": name,
            }
        )

    # --- Persistence ---

    def _persist_entry(self, entry: dict[str, Any]) -> None:
        """Write entry to file, using lazy flush strategy."""
        if not self._persist or not self._session_file:
            return

        # Check if we have an assistant message (triggers flush)
        has_assistant = any(
            e.get("type") == "message" and e.get("message", {}).get("role") == "assistant"
            for e in self._file_entries
            if e.get("type") != "session"
        )

        if not has_assistant:
            self._flushed = False
            return

        if not self._flushed:
            # Flush all entries at once
            os.makedirs(os.path.dirname(self._session_file), exist_ok=True)
            lines = [json.dumps(e, ensure_ascii=False) for e in self._file_entries]
            Path(self._session_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._flushed = True
        else:
            # Incremental append
            with Path(self._session_file).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite_file(self) -> None:
        """Rewrite the entire session file from in-memory entries."""
        if not self._persist or not self._session_file:
            return
        os.makedirs(os.path.dirname(self._session_file), exist_ok=True)
        lines = [json.dumps(e, ensure_ascii=False) for e in self._file_entries]
        Path(self._session_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._flushed = True

    def flush(self) -> None:
        """Force flush all entries to disk."""
        if self._persist and self._session_file:
            self._rewrite_file()

    # --- Tree operations ---

    def branch(self, branch_from_id: str) -> None:
        """Set the leaf to a specific entry, enabling branching."""
        if branch_from_id not in self._by_id:
            msg = f"Entry not found: {branch_from_id}"
            raise ValueError(msg)
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """Reset leaf to None, making next append a root entry."""
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str,
        summary: str,
        *,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        """Branch and append a summary of the abandoned path."""
        self.branch(branch_from_id)
        return self.append_branch_summary(summary, details=details, from_hook=from_hook)

    def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]:
        """Get ordered path from root to the given entry (or current leaf)."""
        target_id = from_id or self._leaf_id
        if target_id is None:
            return []

        path: list[dict[str, Any]] = []
        current_id: str | None = target_id

        while current_id is not None:
            entry = self._by_id.get(current_id)
            if entry is None:
                break
            path.append(entry)
            current_id = entry.get("parentId")

        path.reverse()
        return path

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Get an entry by ID."""
        return self._by_id.get(entry_id)

    def get_label(self, entry_id: str) -> str | None:
        """Get the label for an entry."""
        return self._labels_by_id.get(entry_id)

    def get_session_name(self) -> str | None:
        """Get the latest session display name."""
        name = None
        for entry in self.entries:
            if entry.get("type") == "session_info" and entry.get("name"):
                name = entry["name"]
        return name

    # --- Context building ---

    def build_session_context(self, from_id: str | None = None) -> SessionContext:
        """Walk the tree from root to leaf, building the resolved context.

        Handles compaction entries, thinking level changes, and model changes.
        """
        path = self.get_branch(from_id)
        messages: list[dict[str, Any]] = []
        thinking_level: str | None = None
        model_id: str | None = None
        provider: str | None = None

        for entry in path:
            entry_type = entry.get("type")

            if entry_type == "message":
                messages.append(entry["message"])

            elif entry_type == "compaction":
                # Compaction replaces earlier messages with a summary
                summary = entry.get("summary", "")
                first_kept_id = entry.get("firstKeptEntryId")

                if first_kept_id:
                    # Keep messages from firstKeptEntryId onward
                    # Insert summary as a user message before kept messages
                    messages = [{"role": "user", "content": summary, "timestamp": _ms_now()}]
                else:
                    messages = [{"role": "user", "content": summary, "timestamp": _ms_now()}]

            elif entry_type == "branch_summary":
                # Include branch summary as context
                summary = entry.get("summary", "")
                if summary:
                    messages.append({"role": "user", "content": summary, "timestamp": _ms_now()})

            elif entry_type == "thinking_level_change":
                thinking_level = entry.get("thinkingLevel")

            elif entry_type == "model_change":
                model_id = entry.get("modelId")
                provider = entry.get("provider")

            elif entry_type == "custom_message":
                # Extension messages participate in LLM context
                messages.append(
                    {
                        "role": entry.get("role", "user"),
                        "content": entry.get("content", []),
                        "timestamp": _ms_now(),
                    }
                )

        return SessionContext(
            messages=messages,
            thinking_level=thinking_level,
            model_id=model_id,
            provider=provider,
        )

    # --- Tree visualization ---

    def get_tree(self) -> list[SessionTreeNode]:
        """Build a tree structure from all entries for visualization."""
        nodes: dict[str, SessionTreeNode] = {}
        roots: list[SessionTreeNode] = []

        # Create nodes
        for entry in self.entries:
            entry_id = entry.get("id", "")
            label = self._labels_by_id.get(entry_id)
            node = SessionTreeNode(entry=entry, label=label)  # type: ignore[arg-type]
            nodes[entry_id] = node

        # Build parent-child relationships
        for entry in self.entries:
            entry_id = entry.get("id", "")
            parent_id = entry.get("parentId")
            node = nodes.get(entry_id)
            if node is None:
                continue

            if parent_id and parent_id in nodes:
                nodes[parent_id].children.append(node)
            else:
                roots.append(node)

        # Sort children by timestamp
        for node in nodes.values():
            node.children.sort(key=lambda n: n.entry.get("timestamp", "") if isinstance(n.entry, dict) else "")

        return roots

    def create_branched_session(self) -> str | None:
        """Create a new session file containing only the path from root to current leaf.

        Returns the new file path, or None if not persisting.
        """
        if not self._persist:
            return None

        path = self.get_branch()
        if not path:
            return None

        # Create new header
        header: dict[str, Any] = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": uuid4().hex,
            "timestamp": _timestamp_now(),
            "cwd": self._cwd,
            "parentSession": self._session_id,
        }

        # Rebuild entries with clean parent chain
        new_entries: list[dict[str, Any]] = [header]
        prev_id: str | None = None
        existing_ids: set[str] = set()

        for entry in path:
            new_entry = dict(entry)
            new_id = _generate_id(existing_ids)
            existing_ids.add(new_id)
            new_entry["id"] = new_id
            new_entry["parentId"] = prev_id
            new_entries.append(new_entry)
            prev_id = new_id

        # Recreate labels for entries in the new path
        old_to_new: dict[str, str] = {}
        for old_entry, new_entry in zip(path, new_entries[1:], strict=True):
            old_to_new[old_entry.get("id", "")] = new_entry["id"]

        for target_id, label in self._labels_by_id.items():
            if target_id in old_to_new:
                label_id = _generate_id(existing_ids)
                existing_ids.add(label_id)
                new_entries.append(
                    {
                        "type": "label",
                        "id": label_id,
                        "parentId": prev_id,
                        "timestamp": _timestamp_now(),
                        "label": label,
                        "targetId": old_to_new[target_id],
                    }
                )
                prev_id = label_id

        # Write new file
        os.makedirs(self._session_dir, exist_ok=True)
        ts = _timestamp_now().replace(":", "-")
        new_file = os.path.join(self._session_dir, f"{ts}_{header['id'][:16]}.jsonl")
        lines = [json.dumps(e, ensure_ascii=False) for e in new_entries]
        Path(new_file).write_text("\n".join(lines) + "\n", encoding="utf-8")

        return new_file

    # --- Session listing ---

    @staticmethod
    def list_sessions(
        cwd: str,
        session_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List sessions in a directory, sorted by modified time (newest first)."""
        sdir = session_dir or SessionManager._default_session_dir(cwd)
        if not os.path.isdir(sdir):
            return []

        files: list[str] = []
        try:
            for f in os.listdir(sdir):
                if f.endswith(".jsonl"):
                    full = os.path.join(sdir, f)
                    if is_valid_session_file(full):
                        files.append(full)
        except OSError:
            return []

        sessions: list[SessionInfo] = []
        for i, path in enumerate(files):
            if on_progress:
                on_progress(i + 1, len(files))
            info = _build_session_info(path)
            if info:
                sessions.append(info)

        sessions.sort(key=lambda s: s.modified, reverse=True)
        return sessions

    @staticmethod
    def list_all(
        agent_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List sessions across all project directories."""
        base = agent_dir or _default_agent_dir()
        sessions_root = os.path.join(base, "sessions")
        if not os.path.isdir(sessions_root):
            return []

        all_sessions: list[SessionInfo] = []
        try:
            for dirname in os.listdir(sessions_root):
                sdir = os.path.join(sessions_root, dirname)
                if os.path.isdir(sdir):
                    sessions = SessionManager.list_sessions("", session_dir=sdir, on_progress=on_progress)
                    all_sessions.extend(sessions)
        except OSError:
            pass

        all_sessions.sort(key=lambda s: s.modified, reverse=True)
        return all_sessions

    # --- Helpers ---

    @staticmethod
    def _default_session_dir(cwd: str) -> str:
        """Default session directory for a given working directory."""
        agent_dir = _default_agent_dir()
        # Encode cwd into a safe directory name
        encoded = cwd.replace("/", "--").replace("\\", "--").strip("-")
        return os.path.join(agent_dir, "sessions", encoded)


def _default_agent_dir() -> str:
    """Default agent data directory (~/.pi)."""
    return os.path.join(os.path.expanduser("~"), ".pi")


def _build_session_info(path: str) -> SessionInfo | None:
    """Extract session metadata from a JSONL file."""
    entries = load_entries_from_file(path)
    if not entries:
        return None

    header = entries[0]
    session_id = header.get("id", "")
    cwd = header.get("cwd", "")

    # Gather info
    name: str | None = None
    message_count = 0
    first_user_text = ""
    all_text_parts: list[str] = []

    for entry in entries[1:]:
        if entry.get("type") == "session_info" and entry.get("name"):
            name = entry["name"]
        elif entry.get("type") == "message":
            message_count += 1
            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(
                    item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
                )

            if text:
                if role == "user" and not first_user_text:
                    first_user_text = text[:200]
                all_text_parts.append(text[:100])

    try:
        stat = os.stat(path)
        created = stat.st_ctime
        modified = stat.st_mtime
    except OSError:
        created = modified = 0.0

    return SessionInfo(
        path=path,
        id=session_id,
        cwd=cwd,
        name=name,
        parent_session=header.get("parentSession"),
        created=created,
        modified=modified,
        message_count=message_count,
        first_user_text=first_user_text,
        all_text_preview=" | ".join(all_text_parts[:5]),
    )
