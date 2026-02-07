"""Keyboard input parsing and matching for terminal applications.

Handles kitty keyboard protocol, legacy terminal sequences, and modifier
key combinations. Provides a unified ``matches_key`` function that checks
whether raw terminal input corresponds to a named key identifier such as
``"ctrl+a"`` or ``"shift+enter"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

KeyId = str

# ---------------------------------------------------------------------------
# Global state: kitty keyboard protocol
# ---------------------------------------------------------------------------

_kitty_protocol_active: bool = False


def set_kitty_protocol_active(active: bool) -> None:
    global _kitty_protocol_active
    _kitty_protocol_active = active


def is_kitty_protocol_active() -> bool:
    return _kitty_protocol_active


# ---------------------------------------------------------------------------
# Key helper object
# ---------------------------------------------------------------------------


class Key:
    """Named key constants and modifier combinators."""

    # Special keys
    escape = "escape"
    esc = "esc"
    enter = "enter"
    tab = "tab"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    insert = "insert"
    clear = "clear"
    home = "home"
    end = "end"
    page_up = "pageUp"
    page_down = "pageDown"
    up = "up"
    down = "down"
    left = "left"
    right = "right"

    # Function keys
    f1 = "f1"
    f2 = "f2"
    f3 = "f3"
    f4 = "f4"
    f5 = "f5"
    f6 = "f6"
    f7 = "f7"
    f8 = "f8"
    f9 = "f9"
    f10 = "f10"
    f11 = "f11"
    f12 = "f12"

    # Symbol keys
    backtick = "`"
    hyphen = "-"
    equals = "="
    open_bracket = "["
    close_bracket = "]"
    backslash = "\\"
    semicolon = ";"
    quote = "'"
    comma = ","
    period = "."
    slash = "/"

    @staticmethod
    def ctrl(key: str) -> str:
        return f"ctrl+{key}"

    @staticmethod
    def shift(key: str) -> str:
        return f"shift+{key}"

    @staticmethod
    def alt(key: str) -> str:
        return f"alt+{key}"

    @staticmethod
    def ctrl_shift(key: str) -> str:
        return f"ctrl+shift+{key}"

    @staticmethod
    def ctrl_alt(key: str) -> str:
        return f"ctrl+alt+{key}"

    @staticmethod
    def shift_alt(key: str) -> str:
        return f"shift+alt+{key}"

    @staticmethod
    def ctrl_shift_alt(key: str) -> str:
        return f"ctrl+shift+alt+{key}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL_KEYS: set[str] = {
    "`", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/",
}

MODIFIERS: dict[str, int] = {
    "shift": 1,
    "alt": 2,
    "ctrl": 4,
}

LOCK_MASK = 64 + 128

CODEPOINTS: dict[str, int] = {
    "escape": 27,
    "tab": 9,
    "enter": 13,
    "space": 32,
    "backspace": 127,
    "kp_enter": 57414,
}

ARROW_CODEPOINTS: dict[str, int] = {
    "up": -1,
    "down": -2,
    "right": -3,
    "left": -4,
}

FUNCTIONAL_CODEPOINTS: dict[str, int] = {
    "delete": -10,
    "insert": -11,
    "pageUp": -12,
    "pageDown": -13,
    "home": -14,
    "end": -15,
}

# Legacy escape sequences -> key names
LEGACY_KEY_SEQUENCES: dict[str, str] = {
    "\x1b[A": "up",
    "\x1b[B": "down",
    "\x1b[C": "right",
    "\x1b[D": "left",
    "\x1b[H": "home",
    "\x1b[F": "end",
    "\x1bOA": "up",
    "\x1bOB": "down",
    "\x1bOC": "right",
    "\x1bOD": "left",
    "\x1bOH": "home",
    "\x1bOF": "end",
    "\x1b[2~": "insert",
    "\x1b[3~": "delete",
    "\x1b[5~": "pageUp",
    "\x1b[6~": "pageDown",
    "\x1bOP": "f1",
    "\x1bOQ": "f2",
    "\x1bOR": "f3",
    "\x1bOS": "f4",
    "\x1b[15~": "f5",
    "\x1b[17~": "f6",
    "\x1b[18~": "f7",
    "\x1b[19~": "f8",
    "\x1b[20~": "f9",
    "\x1b[21~": "f10",
    "\x1b[23~": "f11",
    "\x1b[24~": "f12",
    "\x1b[1~": "home",
    "\x1b[4~": "end",
    "\x1b[E": "clear",
}

LEGACY_SHIFT_SEQUENCES: dict[str, str] = {
    "\x1b[1;2A": "up",
    "\x1b[1;2B": "down",
    "\x1b[1;2C": "right",
    "\x1b[1;2D": "left",
    "\x1b[1;2H": "home",
    "\x1b[1;2F": "end",
    "\x1b[3;2~": "delete",
    "\x1b[5;2~": "pageUp",
    "\x1b[6;2~": "pageDown",
    "\x1b[2;2~": "insert",
    "\x1b[1;2P": "f1",
    "\x1b[1;2Q": "f2",
    "\x1b[1;2R": "f3",
    "\x1b[1;2S": "f4",
    "\x1b[15;2~": "f5",
    "\x1b[17;2~": "f6",
    "\x1b[18;2~": "f7",
    "\x1b[19;2~": "f8",
    "\x1b[20;2~": "f9",
    "\x1b[21;2~": "f10",
    "\x1b[23;2~": "f11",
    "\x1b[24;2~": "f12",
    "\x1bO2P": "f1",
    "\x1bO2Q": "f2",
    "\x1bO2R": "f3",
    "\x1bO2S": "f4",
    "\x1b[Z": "tab",
}

LEGACY_CTRL_SEQUENCES: dict[str, str] = {
    "\x1b[1;5A": "up",
    "\x1b[1;5B": "down",
    "\x1b[1;5C": "right",
    "\x1b[1;5D": "left",
    "\x1b[1;5H": "home",
    "\x1b[1;5F": "end",
    "\x1b[3;5~": "delete",
    "\x1b[5;5~": "pageUp",
    "\x1b[6;5~": "pageDown",
    "\x1b[2;5~": "insert",
    "\x1b[1;5P": "f1",
    "\x1b[1;5Q": "f2",
    "\x1b[1;5R": "f3",
    "\x1b[1;5S": "f4",
    "\x1b[15;5~": "f5",
    "\x1b[17;5~": "f6",
    "\x1b[18;5~": "f7",
    "\x1b[19;5~": "f8",
    "\x1b[20;5~": "f9",
    "\x1b[21;5~": "f10",
    "\x1b[23;5~": "f11",
    "\x1b[24;5~": "f12",
    "\x1bO5P": "f1",
    "\x1bO5Q": "f2",
    "\x1bO5R": "f3",
    "\x1bO5S": "f4",
}

LEGACY_ALT_SEQUENCES: dict[str, str] = {
    "\x1b[1;3A": "up",
    "\x1b[1;3B": "down",
    "\x1b[1;3C": "right",
    "\x1b[1;3D": "left",
    "\x1b[1;3H": "home",
    "\x1b[1;3F": "end",
    "\x1b[3;3~": "delete",
    "\x1b[5;3~": "pageUp",
    "\x1b[6;3~": "pageDown",
    "\x1b[2;3~": "insert",
    "\x1b[1;3P": "f1",
    "\x1b[1;3Q": "f2",
    "\x1b[1;3R": "f3",
    "\x1b[1;3S": "f4",
    "\x1b[15;3~": "f5",
    "\x1b[17;3~": "f6",
    "\x1b[18;3~": "f7",
    "\x1b[19;3~": "f8",
    "\x1b[20;3~": "f9",
    "\x1b[21;3~": "f10",
    "\x1b[23;3~": "f11",
    "\x1b[24;3~": "f12",
}

LEGACY_CTRL_SHIFT_SEQUENCES: dict[str, str] = {
    "\x1b[1;6A": "up",
    "\x1b[1;6B": "down",
    "\x1b[1;6C": "right",
    "\x1b[1;6D": "left",
    "\x1b[1;6H": "home",
    "\x1b[1;6F": "end",
    "\x1b[3;6~": "delete",
    "\x1b[5;6~": "pageUp",
    "\x1b[6;6~": "pageDown",
    "\x1b[2;6~": "insert",
    "\x1b[1;6P": "f1",
    "\x1b[1;6Q": "f2",
    "\x1b[1;6R": "f3",
    "\x1b[1;6S": "f4",
    "\x1b[15;6~": "f5",
    "\x1b[17;6~": "f6",
    "\x1b[18;6~": "f7",
    "\x1b[19;6~": "f8",
    "\x1b[20;6~": "f9",
    "\x1b[21;6~": "f10",
    "\x1b[23;6~": "f11",
    "\x1b[24;6~": "f12",
}

LEGACY_CTRL_ALT_SEQUENCES: dict[str, str] = {
    "\x1b[1;7A": "up",
    "\x1b[1;7B": "down",
    "\x1b[1;7C": "right",
    "\x1b[1;7D": "left",
    "\x1b[1;7H": "home",
    "\x1b[1;7F": "end",
    "\x1b[3;7~": "delete",
    "\x1b[5;7~": "pageUp",
    "\x1b[6;7~": "pageDown",
    "\x1b[2;7~": "insert",
    "\x1b[1;7P": "f1",
    "\x1b[1;7Q": "f2",
    "\x1b[1;7R": "f3",
    "\x1b[1;7S": "f4",
    "\x1b[15;7~": "f5",
    "\x1b[17;7~": "f6",
    "\x1b[18;7~": "f7",
    "\x1b[19;7~": "f8",
    "\x1b[20;7~": "f9",
    "\x1b[21;7~": "f10",
    "\x1b[23;7~": "f11",
    "\x1b[24;7~": "f12",
}

LEGACY_SHIFT_ALT_SEQUENCES: dict[str, str] = {
    "\x1b[1;4A": "up",
    "\x1b[1;4B": "down",
    "\x1b[1;4C": "right",
    "\x1b[1;4D": "left",
    "\x1b[1;4H": "home",
    "\x1b[1;4F": "end",
    "\x1b[3;4~": "delete",
    "\x1b[5;4~": "pageUp",
    "\x1b[6;4~": "pageDown",
    "\x1b[2;4~": "insert",
    "\x1b[1;4P": "f1",
    "\x1b[1;4Q": "f2",
    "\x1b[1;4R": "f3",
    "\x1b[1;4S": "f4",
    "\x1b[15;4~": "f5",
    "\x1b[17;4~": "f6",
    "\x1b[18;4~": "f7",
    "\x1b[19;4~": "f8",
    "\x1b[20;4~": "f9",
    "\x1b[21;4~": "f10",
    "\x1b[23;4~": "f11",
    "\x1b[24;4~": "f12",
}

LEGACY_CTRL_SHIFT_ALT_SEQUENCES: dict[str, str] = {
    "\x1b[1;8A": "up",
    "\x1b[1;8B": "down",
    "\x1b[1;8C": "right",
    "\x1b[1;8D": "left",
    "\x1b[1;8H": "home",
    "\x1b[1;8F": "end",
    "\x1b[3;8~": "delete",
    "\x1b[5;8~": "pageUp",
    "\x1b[6;8~": "pageDown",
    "\x1b[2;8~": "insert",
    "\x1b[1;8P": "f1",
    "\x1b[1;8Q": "f2",
    "\x1b[1;8R": "f3",
    "\x1b[1;8S": "f4",
    "\x1b[15;8~": "f5",
    "\x1b[17;8~": "f6",
    "\x1b[18;8~": "f7",
    "\x1b[19;8~": "f8",
    "\x1b[20;8~": "f9",
    "\x1b[21;8~": "f10",
    "\x1b[23;8~": "f11",
    "\x1b[24;8~": "f12",
}

# Reverse mapping: key name -> list of sequences (unmodified only)
LEGACY_SEQUENCE_KEY_IDS: dict[str, list[str]] = {}
for _seq, _key in LEGACY_KEY_SEQUENCES.items():
    LEGACY_SEQUENCE_KEY_IDS.setdefault(_key, []).append(_seq)

# Shifted key mapping for symbols (shift + base key)
SHIFTED_KEY_MAP: dict[str, str] = {
    "`": "~",
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}

# Reverse: shifted char -> base char
UNSHIFTED_KEY_MAP: dict[str, str] = {v: k for k, v in SHIFTED_KEY_MAP.items()}

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

KeyEventType = Literal["press", "repeat", "release"]

# ---------------------------------------------------------------------------
# ParsedKittySequence dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParsedKittySequence:
    codepoint: int
    shifted_key: Optional[int]
    base_layout_key: Optional[int]
    modifier: int
    event_type: int  # 1 = press, 2 = repeat, 3 = release


# ---------------------------------------------------------------------------
# Regex patterns for kitty protocol parsing
# ---------------------------------------------------------------------------

# CSI u format: \x1b[<codepoint>(:<shifted_key>(:<base_layout_key>))?(;<modifier>(:<event_type>))?u
_KITTY_CSI_U_RE = re.compile(
    r"\x1b\[(\d+)(?::(\d+)(?::(\d+))?)?(?:;(\d+)(?::(\d+))?)?u"
)

# Arrow keys with modifier: \x1b[1;<modifier>(:<event_type>)?[ABCD]
_KITTY_ARROW_RE = re.compile(
    r"\x1b\[1;(\d+)(?::(\d+))?([ABCD])"
)

# Functional keys with modifier: \x1b[<number>;<modifier>(:<event_type>)?~
_KITTY_FUNCTIONAL_RE = re.compile(
    r"\x1b\[(\d+);(\d+)(?::(\d+))?~"
)

# Home/End with modifier: \x1b[1;<modifier>(:<event_type>)?[HF]
_KITTY_HOME_END_RE = re.compile(
    r"\x1b\[1;(\d+)(?::(\d+))?([HF])"
)

# F1-F4 with modifier (SS3 style): \x1b[1;<modifier>(:<event_type>)?[PQRS]
_KITTY_F1_F4_RE = re.compile(
    r"\x1b\[1;(\d+)(?::(\d+))?([PQRS])"
)

# ---------------------------------------------------------------------------
# Release / repeat detection
# ---------------------------------------------------------------------------

_BRACKETED_PASTE_RE = re.compile(r"\x1b\[200~")

_RELEASE_PATTERNS = re.compile(
    r"(?::3u|;[^:]*:3~|;[^:]*:3[ABCDHF]|;[^:]*:3[PQRS])"
)

_REPEAT_PATTERNS = re.compile(
    r"(?::2u|;[^:]*:2~|;[^:]*:2[ABCDHF]|;[^:]*:2[PQRS])"
)


def is_key_release(data: str) -> bool:
    """Check if data contains a key release event pattern."""
    if _BRACKETED_PASTE_RE.search(data):
        return False
    return bool(_RELEASE_PATTERNS.search(data))


def is_key_repeat(data: str) -> bool:
    """Check if data contains a key repeat event pattern."""
    if _BRACKETED_PASTE_RE.search(data):
        return False
    return bool(_REPEAT_PATTERNS.search(data))


# ---------------------------------------------------------------------------
# Kitty sequence parsing
# ---------------------------------------------------------------------------

_ARROW_LETTER_TO_CODEPOINT: dict[str, int] = {
    "A": ARROW_CODEPOINTS["up"],
    "B": ARROW_CODEPOINTS["down"],
    "C": ARROW_CODEPOINTS["right"],
    "D": ARROW_CODEPOINTS["left"],
}

_FUNCTIONAL_NUMBER_TO_CODEPOINT: dict[int, int] = {
    3: FUNCTIONAL_CODEPOINTS["delete"],
    2: FUNCTIONAL_CODEPOINTS["insert"],
    5: FUNCTIONAL_CODEPOINTS["pageUp"],
    6: FUNCTIONAL_CODEPOINTS["pageDown"],
    1: FUNCTIONAL_CODEPOINTS["home"],
    4: FUNCTIONAL_CODEPOINTS["end"],
    # F-keys encoded as functional sequences
    15: -20,  # f5
    17: -21,  # f6
    18: -22,  # f7
    19: -23,  # f8
    20: -24,  # f9
    21: -25,  # f10
    23: -26,  # f11
    24: -27,  # f12
}

_HOME_END_LETTER_TO_CODEPOINT: dict[str, int] = {
    "H": FUNCTIONAL_CODEPOINTS["home"],
    "F": FUNCTIONAL_CODEPOINTS["end"],
}

_F1_F4_LETTER_TO_CODEPOINT: dict[str, int] = {
    "P": -16,  # f1
    "Q": -17,  # f2
    "R": -18,  # f3
    "S": -19,  # f4
}

# Codepoint -> key name for functional keys in kitty
_KITTY_CODEPOINT_TO_KEY: dict[int, str] = {
    FUNCTIONAL_CODEPOINTS["delete"]: "delete",
    FUNCTIONAL_CODEPOINTS["insert"]: "insert",
    FUNCTIONAL_CODEPOINTS["pageUp"]: "pageUp",
    FUNCTIONAL_CODEPOINTS["pageDown"]: "pageDown",
    FUNCTIONAL_CODEPOINTS["home"]: "home",
    FUNCTIONAL_CODEPOINTS["end"]: "end",
    ARROW_CODEPOINTS["up"]: "up",
    ARROW_CODEPOINTS["down"]: "down",
    ARROW_CODEPOINTS["right"]: "right",
    ARROW_CODEPOINTS["left"]: "left",
    -16: "f1",
    -17: "f2",
    -18: "f3",
    -19: "f4",
    -20: "f5",
    -21: "f6",
    -22: "f7",
    -23: "f8",
    -24: "f9",
    -25: "f10",
    -26: "f11",
    -27: "f12",
}

# F-key codepoints used by kitty CSI u encoding (57364+)
_KITTY_F_KEY_CODEPOINTS: dict[int, str] = {
    57364: "f1",
    57365: "f2",
    57366: "f3",
    57367: "f4",
    57368: "f5",
    57369: "f6",
    57370: "f7",
    57371: "f8",
    57372: "f9",
    57373: "f10",
    57374: "f11",
    57375: "f12",
}


def parse_kitty_sequence(data: str) -> ParsedKittySequence | None:
    """Parse a kitty keyboard protocol sequence from terminal input data.

    Returns a ``ParsedKittySequence`` or ``None`` if the data does not match
    any recognised kitty protocol pattern.
    """
    # CSI u format
    m = _KITTY_CSI_U_RE.match(data)
    if m:
        codepoint = int(m.group(1))
        shifted_key = int(m.group(2)) if m.group(2) else None
        base_layout_key = int(m.group(3)) if m.group(3) else None
        modifier_raw = int(m.group(4)) if m.group(4) else 1
        event_type = int(m.group(5)) if m.group(5) else 1
        return ParsedKittySequence(
            codepoint=codepoint,
            shifted_key=shifted_key,
            base_layout_key=base_layout_key,
            modifier=modifier_raw,
            event_type=event_type,
        )

    # Arrow keys: \x1b[1;<mod>(:<event>)?[ABCD]
    m = _KITTY_ARROW_RE.match(data)
    if m:
        modifier_raw = int(m.group(1))
        event_type = int(m.group(2)) if m.group(2) else 1
        letter = m.group(3)
        codepoint = _ARROW_LETTER_TO_CODEPOINT.get(letter, 0)
        return ParsedKittySequence(
            codepoint=codepoint,
            shifted_key=None,
            base_layout_key=None,
            modifier=modifier_raw,
            event_type=event_type,
        )

    # Functional keys: \x1b[<num>;<mod>(:<event>)?~
    m = _KITTY_FUNCTIONAL_RE.match(data)
    if m:
        number = int(m.group(1))
        modifier_raw = int(m.group(2))
        event_type = int(m.group(3)) if m.group(3) else 1
        codepoint = _FUNCTIONAL_NUMBER_TO_CODEPOINT.get(number, 0)
        return ParsedKittySequence(
            codepoint=codepoint,
            shifted_key=None,
            base_layout_key=None,
            modifier=modifier_raw,
            event_type=event_type,
        )

    # Home/End: \x1b[1;<mod>(:<event>)?[HF]
    m = _KITTY_HOME_END_RE.match(data)
    if m:
        modifier_raw = int(m.group(1))
        event_type = int(m.group(2)) if m.group(2) else 1
        letter = m.group(3)
        codepoint = _HOME_END_LETTER_TO_CODEPOINT.get(letter, 0)
        return ParsedKittySequence(
            codepoint=codepoint,
            shifted_key=None,
            base_layout_key=None,
            modifier=modifier_raw,
            event_type=event_type,
        )

    # F1-F4: \x1b[1;<mod>(:<event>)?[PQRS]
    m = _KITTY_F1_F4_RE.match(data)
    if m:
        modifier_raw = int(m.group(1))
        event_type = int(m.group(2)) if m.group(2) else 1
        letter = m.group(3)
        codepoint = _F1_F4_LETTER_TO_CODEPOINT.get(letter, 0)
        return ParsedKittySequence(
            codepoint=codepoint,
            shifted_key=None,
            base_layout_key=None,
            modifier=modifier_raw,
            event_type=event_type,
        )

    return None


# ---------------------------------------------------------------------------
# Kitty sequence matching helpers
# ---------------------------------------------------------------------------


def matches_kitty_sequence(
    data: str, expected_codepoint: int, expected_modifier: int
) -> bool:
    """Check if *data* is a kitty sequence matching the given codepoint and modifier."""
    parsed = parse_kitty_sequence(data)
    if parsed is None:
        return False

    # Strip lock bits from the modifier
    actual_mod = (parsed.modifier - 1) & ~LOCK_MASK
    expected_mod = expected_modifier

    if parsed.codepoint == expected_codepoint and actual_mod == expected_mod:
        return True

    # Fallback: check base_layout_key
    if (
        parsed.base_layout_key is not None
        and parsed.base_layout_key == expected_codepoint
        and actual_mod == expected_mod
    ):
        return True

    return False


def matches_modify_other_keys(
    data: str, expected_keycode: int, expected_modifier: int
) -> bool:
    """Match a CSI 27;<modifier>;<keycode>~ sequence (modifyOtherKeys format)."""
    m = re.match(r"\x1b\[27;(\d+);(\d+)~", data)
    if m is None:
        return False
    modifier = int(m.group(1))
    keycode = int(m.group(2))
    actual_mod = (modifier - 1) & ~LOCK_MASK
    return keycode == expected_keycode and actual_mod == expected_modifier


# ---------------------------------------------------------------------------
# Raw control character helper
# ---------------------------------------------------------------------------


def raw_ctrl_char(key: str) -> str | None:
    """Return the control character for a key, or ``None`` if not applicable.

    For example, ``raw_ctrl_char("a")`` returns ``"\\x01"``.
    """
    if len(key) != 1:
        return None
    code = ord(key.lower())
    if ord("a") <= code <= ord("z"):
        return chr(code & 0x1F)
    # Also handle some symbol control characters
    ctrl_map: dict[str, str] = {
        "[": chr(27),   # ESC
        "\\": chr(28),
        "]": chr(29),
        "^": chr(30),
        "_": chr(31),
        "@": chr(0),
        "?": chr(127),  # DEL
    }
    return ctrl_map.get(key)


# ---------------------------------------------------------------------------
# Key ID parsing
# ---------------------------------------------------------------------------


def parse_key_id(key_id: str) -> dict[str, object] | None:
    """Split a key identifier like ``"ctrl+shift+a"`` into its components.

    Returns a dict with:
    - ``modifiers``: int bitmask (shift=1, alt=2, ctrl=4)
    - ``key``: the base key string

    Returns ``None`` if the key_id is empty.
    """
    if not key_id:
        return None

    parts = key_id.split("+")
    modifier = 0
    key_parts: list[str] = []

    for part in parts:
        lower = part.lower()
        if lower in MODIFIERS:
            modifier |= MODIFIERS[lower]
        else:
            key_parts.append(part)

    key = "+".join(key_parts) if key_parts else ""
    if not key:
        return None

    return {"modifiers": modifier, "key": key}


# ---------------------------------------------------------------------------
# F-key helpers
# ---------------------------------------------------------------------------

_FKEY_CODEPOINTS: dict[str, int] = {
    "f1": 57364,
    "f2": 57365,
    "f3": 57366,
    "f4": 57367,
    "f5": 57368,
    "f6": 57369,
    "f7": 57370,
    "f8": 57371,
    "f9": 57372,
    "f10": 57373,
    "f11": 57374,
    "f12": 57375,
}

_FKEY_LEGACY_SEQUENCES: dict[str, list[str]] = {
    "f1": ["\x1bOP"],
    "f2": ["\x1bOQ"],
    "f3": ["\x1bOR"],
    "f4": ["\x1bOS"],
    "f5": ["\x1b[15~"],
    "f6": ["\x1b[17~"],
    "f7": ["\x1b[18~"],
    "f8": ["\x1b[19~"],
    "f9": ["\x1b[20~"],
    "f10": ["\x1b[21~"],
    "f11": ["\x1b[23~"],
    "f12": ["\x1b[24~"],
}

_FKEY_SHIFT_LEGACY: dict[str, list[str]] = {
    "f1": ["\x1b[1;2P", "\x1bO2P"],
    "f2": ["\x1b[1;2Q", "\x1bO2Q"],
    "f3": ["\x1b[1;2R", "\x1bO2R"],
    "f4": ["\x1b[1;2S", "\x1bO2S"],
    "f5": ["\x1b[15;2~"],
    "f6": ["\x1b[17;2~"],
    "f7": ["\x1b[18;2~"],
    "f8": ["\x1b[19;2~"],
    "f9": ["\x1b[20;2~"],
    "f10": ["\x1b[21;2~"],
    "f11": ["\x1b[23;2~"],
    "f12": ["\x1b[24;2~"],
}

_FKEY_CTRL_LEGACY: dict[str, list[str]] = {
    "f1": ["\x1b[1;5P", "\x1bO5P"],
    "f2": ["\x1b[1;5Q", "\x1bO5Q"],
    "f3": ["\x1b[1;5R", "\x1bO5R"],
    "f4": ["\x1b[1;5S", "\x1bO5S"],
    "f5": ["\x1b[15;5~"],
    "f6": ["\x1b[17;5~"],
    "f7": ["\x1b[18;5~"],
    "f8": ["\x1b[19;5~"],
    "f9": ["\x1b[20;5~"],
    "f10": ["\x1b[21;5~"],
    "f11": ["\x1b[23;5~"],
    "f12": ["\x1b[24;5~"],
}

_FKEY_ALT_LEGACY: dict[str, list[str]] = {
    "f1": ["\x1b[1;3P"],
    "f2": ["\x1b[1;3Q"],
    "f3": ["\x1b[1;3R"],
    "f4": ["\x1b[1;3S"],
    "f5": ["\x1b[15;3~"],
    "f6": ["\x1b[17;3~"],
    "f7": ["\x1b[18;3~"],
    "f8": ["\x1b[19;3~"],
    "f9": ["\x1b[20;3~"],
    "f10": ["\x1b[21;3~"],
    "f11": ["\x1b[23;3~"],
    "f12": ["\x1b[24;3~"],
}


# ---------------------------------------------------------------------------
# Modifier bitmask computation from modifier set
# ---------------------------------------------------------------------------


def _modifier_bits(ctrl: bool = False, shift: bool = False, alt: bool = False) -> int:
    bits = 0
    if shift:
        bits |= MODIFIERS["shift"]
    if alt:
        bits |= MODIFIERS["alt"]
    if ctrl:
        bits |= MODIFIERS["ctrl"]
    return bits


# ---------------------------------------------------------------------------
# matches_key — the main matching function
# ---------------------------------------------------------------------------


def matches_key(data: str, key_id: str) -> bool:  # noqa: C901 — complex by nature
    """Return ``True`` if *data* (raw terminal input) matches the named *key_id*.

    *key_id* examples: ``"a"``, ``"ctrl+a"``, ``"shift+enter"``, ``"ctrl+shift+alt+f5"``.
    """
    parsed = parse_key_id(key_id)
    if parsed is None:
        return False

    mod: int = parsed["modifiers"]  # type: ignore[assignment]
    key: str = parsed["key"]  # type: ignore[assignment]

    has_ctrl = bool(mod & MODIFIERS["ctrl"])
    has_shift = bool(mod & MODIFIERS["shift"])
    has_alt = bool(mod & MODIFIERS["alt"])

    # --- Escape ---------------------------------------------------------
    if key == "escape" or key == "esc":
        cp = CODEPOINTS["escape"]
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, cp, mod_bits):
            return True
        if matches_modify_other_keys(data, cp, mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data == "\x1b"
        if has_alt and not has_ctrl and not has_shift:
            return data == "\x1b\x1b"
        return False

    # --- Space ----------------------------------------------------------
    if key == "space":
        cp = CODEPOINTS["space"]
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, cp, mod_bits):
            return True
        if matches_modify_other_keys(data, cp, mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data == " "
        if has_ctrl and not has_shift and not has_alt:
            return data == "\x00"
        if has_alt and not has_ctrl and not has_shift:
            return data == "\x1b "
        return False

    # --- Tab ------------------------------------------------------------
    if key == "tab":
        cp = CODEPOINTS["tab"]
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, cp, mod_bits):
            return True
        if matches_modify_other_keys(data, cp, mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data == "\t"
        if has_shift and not has_ctrl and not has_alt:
            return data == "\x1b[Z"
        if has_alt and not has_ctrl and not has_shift:
            return data == "\x1b\t"
        return False

    # --- Enter ----------------------------------------------------------
    if key == "enter" or key == "return":
        cp_enter = CODEPOINTS["enter"]
        cp_kp = CODEPOINTS["kp_enter"]
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, cp_enter, mod_bits):
            return True
        if matches_kitty_sequence(data, cp_kp, mod_bits):
            return True
        if matches_modify_other_keys(data, cp_enter, mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data == "\r" or data == "\n"
        if has_alt and not has_ctrl and not has_shift:
            return data == "\x1b\r" or data == "\x1b\n"
        return False

    # --- Backspace ------------------------------------------------------
    if key == "backspace":
        cp = CODEPOINTS["backspace"]
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, cp, mod_bits):
            return True
        if matches_modify_other_keys(data, cp, mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data == "\x7f" or data == "\x08"
        if has_alt and not has_ctrl and not has_shift:
            return data == "\x1b\x7f" or data == "\x1b\x08"
        if has_ctrl and not has_shift and not has_alt:
            return data == "\x08"
        return False

    # --- Insert ---------------------------------------------------------
    if key == "insert":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["insert"], mod_bits):
            return True
        if not has_ctrl and not has_shift and not has_alt:
            return data in LEGACY_SEQUENCE_KEY_IDS.get("insert", [])
        if has_shift and not has_ctrl and not has_alt:
            return data in LEGACY_SHIFT_SEQUENCES and LEGACY_SHIFT_SEQUENCES[data] == "insert"
        if has_ctrl and not has_shift and not has_alt:
            return data in LEGACY_CTRL_SEQUENCES and LEGACY_CTRL_SEQUENCES[data] == "insert"
        if has_alt and not has_ctrl and not has_shift:
            return data in LEGACY_ALT_SEQUENCES and LEGACY_ALT_SEQUENCES[data] == "insert"
        if has_ctrl and has_shift and not has_alt:
            return data in LEGACY_CTRL_SHIFT_SEQUENCES and LEGACY_CTRL_SHIFT_SEQUENCES[data] == "insert"
        if has_ctrl and has_alt and not has_shift:
            return data in LEGACY_CTRL_ALT_SEQUENCES and LEGACY_CTRL_ALT_SEQUENCES[data] == "insert"
        if has_shift and has_alt and not has_ctrl:
            return data in LEGACY_SHIFT_ALT_SEQUENCES and LEGACY_SHIFT_ALT_SEQUENCES[data] == "insert"
        if has_ctrl and has_shift and has_alt:
            return data in LEGACY_CTRL_SHIFT_ALT_SEQUENCES and LEGACY_CTRL_SHIFT_ALT_SEQUENCES[data] == "insert"
        return False

    # --- Delete ---------------------------------------------------------
    if key == "delete":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["delete"], mod_bits):
            return True
        return _match_legacy_key(data, "delete", has_ctrl, has_shift, has_alt)

    # --- Clear ----------------------------------------------------------
    if key == "clear":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        # Clear doesn't have a kitty codepoint in the standard set, but check legacy
        if not has_ctrl and not has_shift and not has_alt:
            return data in LEGACY_SEQUENCE_KEY_IDS.get("clear", [])
        return False

    # --- Home -----------------------------------------------------------
    if key == "home":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["home"], mod_bits):
            return True
        return _match_legacy_key(data, "home", has_ctrl, has_shift, has_alt)

    # --- End ------------------------------------------------------------
    if key == "end":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["end"], mod_bits):
            return True
        return _match_legacy_key(data, "end", has_ctrl, has_shift, has_alt)

    # --- Page Up --------------------------------------------------------
    if key == "pageUp":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["pageUp"], mod_bits):
            return True
        return _match_legacy_key(data, "pageUp", has_ctrl, has_shift, has_alt)

    # --- Page Down ------------------------------------------------------
    if key == "pageDown":
        mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
        if matches_kitty_sequence(data, FUNCTIONAL_CODEPOINTS["pageDown"], mod_bits):
            return True
        return _match_legacy_key(data, "pageDown", has_ctrl, has_shift, has_alt)

    # --- Arrow keys -----------------------------------------------------
    if key == "up":
        return _match_arrow(data, "up", has_ctrl, has_shift, has_alt)
    if key == "down":
        return _match_arrow(data, "down", has_ctrl, has_shift, has_alt)
    if key == "left":
        return _match_arrow(data, "left", has_ctrl, has_shift, has_alt)
    if key == "right":
        return _match_arrow(data, "right", has_ctrl, has_shift, has_alt)

    # --- Function keys --------------------------------------------------
    fkey_match = re.match(r"^f(\d+)$", key)
    if fkey_match:
        return _match_fkey(data, key, has_ctrl, has_shift, has_alt)

    # --- Single character keys (letters, digits, symbols) ---------------
    return _match_char_key(data, key, has_ctrl, has_shift, has_alt)


# ---------------------------------------------------------------------------
# Internal helpers for matches_key
# ---------------------------------------------------------------------------


def _match_legacy_key(
    data: str, key_name: str, has_ctrl: bool, has_shift: bool, has_alt: bool
) -> bool:
    """Match a key against legacy terminal sequences for all modifier combos."""
    if not has_ctrl and not has_shift and not has_alt:
        return data in LEGACY_SEQUENCE_KEY_IDS.get(key_name, [])
    if has_shift and not has_ctrl and not has_alt:
        return LEGACY_SHIFT_SEQUENCES.get(data) == key_name
    if has_ctrl and not has_shift and not has_alt:
        return LEGACY_CTRL_SEQUENCES.get(data) == key_name
    if has_alt and not has_ctrl and not has_shift:
        return LEGACY_ALT_SEQUENCES.get(data) == key_name
    if has_ctrl and has_shift and not has_alt:
        return LEGACY_CTRL_SHIFT_SEQUENCES.get(data) == key_name
    if has_ctrl and has_alt and not has_shift:
        return LEGACY_CTRL_ALT_SEQUENCES.get(data) == key_name
    if has_shift and has_alt and not has_ctrl:
        return LEGACY_SHIFT_ALT_SEQUENCES.get(data) == key_name
    if has_ctrl and has_shift and has_alt:
        return LEGACY_CTRL_SHIFT_ALT_SEQUENCES.get(data) == key_name
    return False


def _match_arrow(
    data: str, direction: str, has_ctrl: bool, has_shift: bool, has_alt: bool
) -> bool:
    """Match an arrow key with optional modifiers."""
    cp = ARROW_CODEPOINTS[direction]
    mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)
    if matches_kitty_sequence(data, cp, mod_bits):
        return True
    return _match_legacy_key(data, direction, has_ctrl, has_shift, has_alt)


def _match_fkey(
    data: str, key_name: str, has_ctrl: bool, has_shift: bool, has_alt: bool
) -> bool:
    """Match a function key (f1-f12) with optional modifiers."""
    # Kitty protocol: F-keys have dedicated codepoints (57364+)
    kitty_cp = _FKEY_CODEPOINTS.get(key_name)
    mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)

    if kitty_cp is not None:
        if matches_kitty_sequence(data, kitty_cp, mod_bits):
            return True

    # Also check the functional codepoint form used by parse_kitty_sequence
    internal_cp = _KITTY_CODEPOINT_TO_KEY
    for cp_val, name in internal_cp.items():
        if name == key_name:
            if matches_kitty_sequence(data, cp_val, mod_bits):
                return True
            break

    # Legacy sequences
    if not has_ctrl and not has_shift and not has_alt:
        seqs = _FKEY_LEGACY_SEQUENCES.get(key_name, [])
        return data in seqs
    if has_shift and not has_ctrl and not has_alt:
        seqs = _FKEY_SHIFT_LEGACY.get(key_name, [])
        return data in seqs
    if has_ctrl and not has_shift and not has_alt:
        seqs = _FKEY_CTRL_LEGACY.get(key_name, [])
        return data in seqs
    if has_alt and not has_ctrl and not has_shift:
        seqs = _FKEY_ALT_LEGACY.get(key_name, [])
        return data in seqs
    if has_ctrl and has_shift and not has_alt:
        return LEGACY_CTRL_SHIFT_SEQUENCES.get(data) == key_name
    if has_ctrl and has_alt and not has_shift:
        return LEGACY_CTRL_ALT_SEQUENCES.get(data) == key_name
    if has_shift and has_alt and not has_ctrl:
        return LEGACY_SHIFT_ALT_SEQUENCES.get(data) == key_name
    if has_ctrl and has_shift and has_alt:
        return LEGACY_CTRL_SHIFT_ALT_SEQUENCES.get(data) == key_name
    return False


def _match_char_key(
    data: str, key: str, has_ctrl: bool, has_shift: bool, has_alt: bool
) -> bool:
    """Match a single character key (letter, digit, or symbol) with modifiers."""
    # Determine the codepoint for kitty matching
    if len(key) == 1:
        key_lower = key.lower()
        cp = ord(key_lower)
    else:
        return False

    mod_bits = _modifier_bits(ctrl=has_ctrl, shift=has_shift, alt=has_alt)

    # Kitty protocol match
    if matches_kitty_sequence(data, cp, mod_bits):
        return True
    if matches_modify_other_keys(data, cp, mod_bits):
        return True

    # Also try uppercase codepoint for shifted keys in kitty
    if has_shift and key_lower.isalpha():
        upper_cp = ord(key_lower.upper())
        if matches_kitty_sequence(data, upper_cp, mod_bits):
            return True
        if matches_modify_other_keys(data, upper_cp, mod_bits):
            return True

    # Also try shifted symbol codepoint
    if has_shift and key in SHIFTED_KEY_MAP:
        shifted_cp = ord(SHIFTED_KEY_MAP[key])
        if matches_kitty_sequence(data, shifted_cp, _modifier_bits(ctrl=has_ctrl, alt=has_alt)):
            return True

    # --- Plain key (no modifiers) ---
    if not has_ctrl and not has_shift and not has_alt:
        return data == key

    # --- Ctrl only ---
    if has_ctrl and not has_shift and not has_alt:
        ctrl = raw_ctrl_char(key)
        if ctrl is not None and data == ctrl:
            return True
        # Also check for ESC-prefixed version when ctrl produces a char
        return False

    # --- Alt only ---
    if has_alt and not has_ctrl and not has_shift:
        # Alt is typically ESC + key
        if data == "\x1b" + key:
            return True
        # Alt + letter can also come as high-bit set (rare, but some terminals)
        return False

    # --- Shift only ---
    if has_shift and not has_ctrl and not has_alt:
        if key_lower.isalpha():
            return data == key_lower.upper()
        if key in SHIFTED_KEY_MAP:
            return data == SHIFTED_KEY_MAP[key]
        return False

    # --- Ctrl + Alt ---
    if has_ctrl and has_alt and not has_shift:
        ctrl = raw_ctrl_char(key)
        if ctrl is not None and data == "\x1b" + ctrl:
            return True
        return False

    # --- Ctrl + Shift ---
    if has_ctrl and has_shift and not has_alt:
        # ctrl+shift+letter: some terminals send the ctrl char, some send modified sequences
        ctrl = raw_ctrl_char(key)
        if ctrl is not None and data == ctrl:
            return True
        return False

    # --- Shift + Alt ---
    if has_shift and has_alt and not has_ctrl:
        if key_lower.isalpha():
            return data == "\x1b" + key_lower.upper()
        if key in SHIFTED_KEY_MAP:
            return data == "\x1b" + SHIFTED_KEY_MAP[key]
        return False

    # --- Ctrl + Shift + Alt ---
    if has_ctrl and has_shift and has_alt:
        ctrl = raw_ctrl_char(key)
        if ctrl is not None and data == "\x1b" + ctrl:
            return True
        return False

    return False


# ---------------------------------------------------------------------------
# parse_key — determine what key was pressed from raw input
# ---------------------------------------------------------------------------


def parse_key(data: str) -> str | None:  # noqa: C901
    """Parse raw terminal input and return the key identifier, or ``None``.

    The returned string uses the same format as ``matches_key`` expects:
    e.g. ``"a"``, ``"ctrl+a"``, ``"shift+enter"``, ``"f5"``.
    """
    if not data:
        return None

    # --- Try kitty protocol first ---
    parsed = parse_kitty_sequence(data)
    if parsed is not None:
        mod = (parsed.modifier - 1) & ~LOCK_MASK
        has_shift = bool(mod & MODIFIERS["shift"])
        has_alt = bool(mod & MODIFIERS["alt"])
        has_ctrl = bool(mod & MODIFIERS["ctrl"])

        prefix = ""
        if has_ctrl:
            prefix += "ctrl+"
        if has_shift:
            prefix += "shift+"
        if has_alt:
            prefix += "alt+"

        cp = parsed.codepoint

        # Check named codepoints
        for name, code in CODEPOINTS.items():
            if cp == code:
                if name == "kp_enter":
                    return prefix + "enter"
                return prefix + name

        # Check arrow codepoints
        key_name = _KITTY_CODEPOINT_TO_KEY.get(cp)
        if key_name is not None:
            return prefix + key_name

        # Check f-key codepoints (57364+)
        fkey_name = _KITTY_F_KEY_CODEPOINTS.get(cp)
        if fkey_name is not None:
            return prefix + fkey_name

        # Regular character
        if cp > 0:
            ch = chr(cp)
            if ch.isprintable():
                return prefix + ch.lower()

        return None

    # --- modifyOtherKeys format: CSI 27;modifier;keycode ~ ---
    mok_match = re.match(r"\x1b\[27;(\d+);(\d+)~", data)
    if mok_match:
        modifier = int(mok_match.group(1))
        keycode = int(mok_match.group(2))
        mod = (modifier - 1) & ~LOCK_MASK
        has_shift = bool(mod & MODIFIERS["shift"])
        has_alt = bool(mod & MODIFIERS["alt"])
        has_ctrl = bool(mod & MODIFIERS["ctrl"])

        prefix = ""
        if has_ctrl:
            prefix += "ctrl+"
        if has_shift:
            prefix += "shift+"
        if has_alt:
            prefix += "alt+"

        if keycode > 0:
            ch = chr(keycode)
            if ch.isprintable():
                return prefix + ch.lower()

        return None

    # --- Legacy escape sequences ---
    # Check modified sequences first (they're longer / more specific)
    for seq_dict, mod_prefix in [
        (LEGACY_CTRL_SHIFT_ALT_SEQUENCES, "ctrl+shift+alt+"),
        (LEGACY_CTRL_SHIFT_SEQUENCES, "ctrl+shift+"),
        (LEGACY_CTRL_ALT_SEQUENCES, "ctrl+alt+"),
        (LEGACY_SHIFT_ALT_SEQUENCES, "shift+alt+"),
        (LEGACY_CTRL_SEQUENCES, "ctrl+"),
        (LEGACY_SHIFT_SEQUENCES, "shift+"),
        (LEGACY_ALT_SEQUENCES, "alt+"),
        (LEGACY_KEY_SEQUENCES, ""),
    ]:
        if data in seq_dict:
            return mod_prefix + seq_dict[data]

    # --- Simple single-byte keys ---
    if data == "\x1b":
        return "escape"
    if data == "\r" or data == "\n":
        return "enter"
    if data == "\t":
        return "tab"
    if data == " ":
        return "space"
    if data == "\x7f" or data == "\x08":
        return "backspace"
    if data == "\x00":
        return "ctrl+space"
    if data == "\x1b[Z":
        return "shift+tab"

    # --- Ctrl + letter (0x01 - 0x1a) ---
    if len(data) == 1 and 1 <= ord(data) <= 26:
        return "ctrl+" + chr(ord(data) + ord("a") - 1)

    # --- Alt + key (ESC prefix) ---
    if len(data) == 2 and data[0] == "\x1b":
        ch = data[1]
        if ch == "\x1b":
            return "alt+escape"
        if ch == "\r" or ch == "\n":
            return "alt+enter"
        if ch == "\t":
            return "alt+tab"
        if ch == " ":
            return "alt+space"
        if ch == "\x7f" or ch == "\x08":
            return "alt+backspace"
        if 1 <= ord(ch) <= 26:
            return "ctrl+alt+" + chr(ord(ch) + ord("a") - 1)
        if ch.isupper():
            return "shift+alt+" + ch.lower()
        if ch.isprintable():
            return "alt+" + ch.lower()

    # --- Plain printable character ---
    if len(data) == 1 and data.isprintable():
        return data

    return None
