"""Configuration for the web UI server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    db_path: str = field(default_factory=lambda: str(Path.home() / ".pi" / "web-ui.db"))
    static_dir: str = field(default_factory=lambda: str(Path(__file__).parent / "static"))
