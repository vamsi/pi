"""Configuration management for pi-pods. Stores pod config at ~/.pi/pods.json."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pi.pods.types import Config, Pod, config_from_dict, config_to_dict


def _get_config_dir() -> Path:
    config_dir = Path(os.environ.get("PI_CONFIG_DIR", Path.home() / ".pi"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_config_path() -> Path:
    return _get_config_dir() / "pods.json"


def load_config() -> Config:
    config_path = _get_config_path()
    if not config_path.exists():
        return Config()
    try:
        data = json.loads(config_path.read_text())
        return config_from_dict(data)
    except Exception as e:
        print(f"Error reading config: {e}", file=sys.stderr)
        return Config()


def save_config(config: Config) -> None:
    config_path = _get_config_path()
    try:
        config_path.write_text(json.dumps(config_to_dict(config), indent=2))
    except Exception as e:
        print(f"Error saving config: {e}", file=sys.stderr)
        sys.exit(1)


def get_active_pod() -> tuple[str, Pod] | None:
    config = load_config()
    if not config.active or config.active not in config.pods:
        return None
    return (config.active, config.pods[config.active])


def add_pod(name: str, pod: Pod) -> None:
    config = load_config()
    config.pods[name] = pod
    if not config.active:
        config.active = name
    save_config(config)


def remove_pod(name: str) -> None:
    config = load_config()
    config.pods.pop(name, None)
    if config.active == name:
        config.active = None
    save_config(config)


def set_active_pod(name: str) -> None:
    config = load_config()
    if name not in config.pods:
        print(f"Pod '{name}' not found", file=sys.stderr)
        sys.exit(1)
    config.active = name
    save_config(config)
