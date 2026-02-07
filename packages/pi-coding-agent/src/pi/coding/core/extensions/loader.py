"""Extension discovery and loading via importlib.

Discovers extensions from configured paths, global/project directories,
and loads them by importing Python modules that export factory functions.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any

from pi.coding.core.extensions.types import Extension, ExtensionAPI, ExtensionFactory


def _expand_path(path: str) -> str:
    """Expand ~ and resolve path."""
    return str(Path(os.path.expanduser(path)).resolve())


def _discover_in_dir(directory: str) -> list[str]:
    """Discover extension entry points in a directory.

    Looks for:
    1. Direct .py files
    2. Subdirectories with __init__.py
    3. Subdirectories with a pyproject.toml declaring pi.extensions
    """
    if not os.path.isdir(directory):
        return []

    paths: list[str] = []
    try:
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)

            # Direct .py file
            if name.endswith(".py") and os.path.isfile(full):
                paths.append(full)
                continue

            # Subdirectory with __init__.py
            if os.path.isdir(full):
                init = os.path.join(full, "__init__.py")
                if os.path.isfile(init):
                    paths.append(full)
                    continue

                # Check for entry point file
                for entry_name in ("extension.py", "main.py", "index.py"):
                    entry = os.path.join(full, entry_name)
                    if os.path.isfile(entry):
                        paths.append(entry)
                        break
    except OSError:
        pass

    return paths


def discover_extensions_in_dir(directory: str) -> list[str]:
    """Public API for discovering extensions in a directory."""
    return _discover_in_dir(directory)


def _load_module_from_path(path: str) -> Any:
    """Load a Python module from a file path using importlib."""
    resolved = _expand_path(path)

    if os.path.isdir(resolved):
        # Look for __init__.py in directory
        init = os.path.join(resolved, "__init__.py")
        if os.path.isfile(init):
            resolved = init
        else:
            msg = f"Directory has no __init__.py: {resolved}"
            raise ImportError(msg)

    if not os.path.isfile(resolved):
        msg = f"Extension file not found: {resolved}"
        raise ImportError(msg)

    # Generate a unique module name
    module_name = f"pi_extension_{Path(resolved).stem}_{id(resolved)}"

    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        msg = f"Cannot create module spec for: {resolved}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        msg = f"Failed to load extension: {resolved}"
        raise ImportError(msg) from e

    return module


def _find_factory(module: Any) -> ExtensionFactory | None:
    """Find the extension factory function in a loaded module.

    Looks for:
    1. A function named 'extension' or 'activate'
    2. A callable 'default' export
    3. The first callable that takes a single ExtensionAPI parameter
    """
    # Check for known factory names
    for name in ("extension", "activate", "default"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory

    # Look for any callable with ExtensionAPI parameter
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and _accepts_extension_api(obj):
            return obj

    return None


def _accepts_extension_api(func: Any) -> bool:
    """Check if a function accepts an ExtensionAPI parameter."""
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) == 1:
            annotation = params[0].annotation
            if annotation is inspect.Parameter.empty:
                return True  # Assume yes if no annotation
            if annotation is ExtensionAPI or (isinstance(annotation, str) and "ExtensionAPI" in annotation):
                return True
    except ValueError, TypeError:
        pass
    return False


# --- Public API ---


async def load_extensions(
    paths: list[str],
    cwd: str,
) -> tuple[list[Extension], list[str]]:
    """Load extensions from explicit file paths.

    Returns (loaded_extensions, errors).
    """
    extensions: list[Extension] = []
    errors: list[str] = []

    seen: set[str] = set()
    for path in paths:
        resolved = _expand_path(path)
        if resolved in seen:
            continue
        seen.add(resolved)

        try:
            module = _load_module_from_path(path)
            factory = _find_factory(module)
            if factory is None:
                errors.append(f"No extension factory found in: {path}")
                continue

            ext = Extension(path=path, resolved_path=resolved)
            api = ExtensionAPI(ext, cwd)

            result = factory(api)
            if inspect.isawaitable(result):
                await result

            extensions.append(ext)
        except Exception as e:
            errors.append(f"Error loading extension {path}: {e}")

    return extensions, errors


async def discover_and_load_extensions(
    configured_paths: list[str],
    cwd: str,
    agent_dir: str | None = None,
) -> tuple[list[Extension], list[str]]:
    """Discover extensions from all sources and load them.

    Sources (in order):
    1. Global extensions directory (agent_dir/extensions/)
    2. Project-local extensions (cwd/.pi/extensions/)
    3. Explicitly configured paths

    Returns (loaded_extensions, errors).
    """
    all_paths: list[str] = []

    # 1. Global extensions
    if agent_dir:
        global_dir = os.path.join(agent_dir, "extensions")
        all_paths.extend(_discover_in_dir(global_dir))

    # 2. Project-local extensions
    project_dir = os.path.join(cwd, ".pi", "extensions")
    all_paths.extend(_discover_in_dir(project_dir))

    # 3. Configured paths
    for path in configured_paths:
        expanded = _expand_path(path)
        if os.path.isdir(expanded):
            all_paths.extend(_discover_in_dir(expanded))
        elif os.path.isfile(expanded):
            all_paths.append(expanded)

    return await load_extensions(all_paths, cwd)


async def load_extension_from_factory(
    factory: ExtensionFactory,
    cwd: str,
    *,
    path: str = "<inline>",
) -> Extension:
    """Load a single extension from an inline factory function."""
    ext = Extension(path=path, resolved_path=path)
    api = ExtensionAPI(ext, cwd)

    result = factory(api)
    if inspect.isawaitable(result):
        await result

    return ext
